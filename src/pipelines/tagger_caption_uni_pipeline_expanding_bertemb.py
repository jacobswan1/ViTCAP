import json
from src.tools.metric import *
from src.tools.torch_common import *
from torchvision.transforms import transforms
from src.data_layer.builder import collate_fn
from src.pipelines.uni_pipeline import UniPipeline
from src.data_layer.dataset import TransCaptionTensorizer
from src.data_layer.dataset import Tensorizer, pert_collate_fn
from src.layers.bert import ViTCAP
from src.data_layer.dataset import CaptionIdxTSVDataset, ImageIdxTSVDataset
from src.data_layer.transform import (
    LoadLabel,
    LoadHW,
    LoadImage,
    LoadCaption,
    IdentifyTextAB,
    RemoveUselessKeys,
    RenameKey,
)



class ImageCaptioning(nn.Module):
    def __init__(self,
                 model,
                 test_extra_input=None,
                 tokenizer=None,
                 bert_tokenizer=None,
                 image_encoder=None,
                 cfg=None,
                 ):
        super().__init__()
        self.module = model
        self.iter = 0
        self.tokenizer = tokenizer
        self.bert_tokenizer = bert_tokenizer
        self.test_extra_input = test_extra_input
        self.image_encoder = image_encoder
        self.cfg = cfg
        self.acc = MultiLabelAccuracy()
        self.map = mAPMeter()

        if cfg.pert_img_prob is not None and cfg.pert_img_prob > 0:
            # we need an image text matching loss on the pooled output
            # number of relationship is always 1, we use BCE loss
            self.seq_relationship = nn.Linear(model.bert.pooler.dense.weight.shape[0], 1)
            assert self.cfg.mask_type != 'seq2seq', 'matching loss is useless'
        else:
            self.seq_relationship = None

        self.category = cfg.category
        if self.category == 'vinvl':
            self.vocab = self.tokenizer['idx_to_label']
        else:
            self.vocab = self.bert_tokenizer.ids_to_tokens

    def construct_attn_mask(self, data):
        img_feats = data['img_feats']
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        batch_size = img_feats.shape[0]

        num_img_feats = img_feats.shape[1]
        num_token = input_ids.shape[-1]
        device = input_ids.device
        top_right = torch.ones((batch_size, num_token, num_img_feats), device=device)
        if self.cfg.mask_type == 'seqbid':
            mask_type = data.pop('mask_type')
            bottom_left = torch.ones((batch_size, num_img_feats, num_token), device=device)
            # if mask_type is 1, it is seq2seq and we need to zero it out
            bottom_left[mask_type] = 0
        elif self.cfg.mask_type in ['seq2seq', 'seq2seq_off']:
            bottom_left = torch.zeros((batch_size, num_img_feats, num_token), device=device)
        else:
            assert self.cfg.mask_type == 'bidirectional'
            bottom_left = torch.ones((batch_size, num_img_feats, num_token), device=device)
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(dim=1)
                attention_mask = attention_mask.expand(batch_size, num_token, num_token)
        bottom_right = torch.ones((batch_size, num_img_feats, num_img_feats), device=device)
        bottom = torch.cat((bottom_left, bottom_right), dim=2)

        top = torch.cat((attention_mask, top_right), dim=2)
        full_attention_mask = torch.cat((top, bottom), dim=1)
        data['attention_mask'] = full_attention_mask

    def forward(self, data):

        data = dict(data.items())
        # this is required in test, but not in train
        data.pop('key')

        if self.training:
            if self.cfg.gt_tag_train:
                data['gen_tag_ratio'] = torch.tensor(0.05).cuda()
            elif self.cfg.pred_tag_train:
                data['gen_tag_ratio'] = torch.tensor(1).cuda()
            # linearly increase the gen_tag_ratio to the max (1.0)
            else:
                data['gen_tag_ratio'] = torch.tensor(max(self.cfg.gen_tag_ratio, self.iter/self.cfg.max_iter)).cuda() \
                    if getattr(self.cfg, 'gen_tag_ratio', None) else None

        else:
            data['gen_tag_ratio'] = 1

        if self.image_encoder:
            assert 'img_feats' not in data
            data['img_feats'] = self.image_encoder(data)

            self.construct_attn_mask(data)

            if 'image' in data: data.pop('image')

        if self.training:

            matched = data.get('matched')
            result = self.module(**data, return_dict=True)
            loss_dict = {}

            if matched is not None:
                matching_loss = self.calc_image_text_matching_loss(result, matched)
                loss_dict['matching_loss'] = matching_loss

            verbose = (self.iter % 100) == 0

            self.iter += 1

            if verbose:
                self.acc.calc(result['tag_logits'], data['label'])

                masked_ids = data['masked_ids']
                masked_ids = masked_ids[masked_ids != 0]
                if masked_ids.numel() > 0:
                    # when we have image text matching pair, there could be a
                    # chance there is no masked tokens as we ignore it if it is
                    # not matched. One example is batch size = 2, mismatching
                    # rate is 0.5.
                    batch_score = compute_score_with_logits(result['class_logits'], masked_ids)
                    batch_acc = torch.sum(batch_score.float()) / torch.sum(data['masked_pos'])
                    logging.info('caption acc = {}'.format(batch_acc))

                    with torch.no_grad():
                        # compute mAP
                        logging.info('Tag Loss = {}'.format(result['tag_loss'].detach()))
                        logging.info('Tag Precision. = {}'.format(self.acc.prec()))

                        # compute mAP
                        self.map.add(output=torch.nn.functional.sigmoid(result['tag_logits'].detach()), target=data['label'])
                        logging.info('Tag mAP: {}'.format(self.map.value()))

                    if self.iter == 1:
                        logging.info('# of tokens = {}'.format(data['img_feats'].shape[1]))

                    logging.info('# of tokens = {}'.format(data['img_feats'].shape[1]))
                    logging.info('Input ids sample: {}'.format(str(
                        [self.bert_tokenizer.ids_to_tokens[token] for token in data['input_ids'][0].detach().cpu().numpy()])))

                    logging.info('Sample Generation: {}'.format(str(logit_to_label(result['tag_logits'].detach(),
                                                                                   self.vocab,
                                                                                   topk=50,
                                                                                   category=self.category)[0])))

                    logging.info('GT Tags: {}'.format(str(label_to_label(data['label'], self.vocab, category=self.category)[0]), ))

            # for mismatched pair, we ignore the masked token loss in the
            # self.module, where we clear out the masked pos. That logic can be
            # moved to collate function also.

            # FIXME: use nX multiplier to the caption loss.
            loss_dict['masked_loss'] = result['masked_loss']
            return loss_dict

        else:

            if self.cfg.use_cbs:
                data.update({
                    'min_constraints_to_satisfy': 2,
                    'use_cbs': True,
                })

            data.update(self.test_extra_input)
            result = self.module(**data)

            return result

    def calc_image_text_matching_loss(self, result, matched):
        logits = self.seq_relationship(result['pooled_output'])
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, matched.float().reshape(logits.shape))


class CaptionUniPipeline(UniPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._default.update({
            'mask_type': 'seq2seq',
            'max_seq_a_length': 40,
            'max_seq_length': 70,
            'add_od_labels': True,
            'od_label_conf ': 0.2,
            'drop_out': 0.1,
            'tie_weights': True,
            'label_smoothing': 0.1,
            'img_layer_norm_eps': 1e-5,
            'max_img_seq_length': 50,
            'max_gen_length': 20,
            'output_isvalid': False,
            'max_masked_tokens': 3,
            'cider_cached_tokens': 'data/coco_caption/gt/coco-train-words.p',
            'num_beams': 1,
            'mask_prob': 0.15,
            'replace_by_mask_prob': 0.8,
            'replace_by_rand_prob': 0.1,
            'temperature': 1,
            'top_k': 0,
            'top_p': 1,
            'gradient_clip': 1.,
            'optimizer_type': 'MAdamW',
            'bias_no_weight_decay': True,
            'ln_no_weight_decay': True,
            'unique_labels_on': False,
            'scheduler_type': 'linear',
            'pad_to_max': True,
            'no_sort_by_conf': False,
            'ignore_project_image': False,
            'real_text_a_in_test': False,
            'pert_img_prob': None,
        })
        self._tokenizer = None
        self._test_caption_tensorizer = None
        self._train_caption_tensorizer = None

        if self.cfg.pad_to_max:
            from torch.utils.data.dataloader import default_collate
            self.train_collate_fn = default_collate
            self.test_collate_fn = default_collate
        else:
            if self.cfg.pert_img_prob is not None:
                self.train_collate_fn = \
                    lambda x: pert_collate_fn(x, self.cfg.pert_img_prob)
            else:
                self.train_collate_fn = collate_fn
            self.test_collate_fn = collate_fn

    def train(self):
        model = self.get_model(is_train=True)

        self.cfg.max_iter = self.max_iter

        optimizer = self.get_optimizer(model)
        logging.info(optimizer)

        scheduler = self.get_lr_scheduler(optimizer)

        checkpointer = self.create_checkpointer(
            model,
            optimizer,
            scheduler,
        )

        # load base-model or last pre-trained snapshot
        extra_param = checkpointer.recover_or_load(
            self.cfg.basemodel, model_only=True, load_if_has=False)

        # init. the bert.encoder.tag_blocks with the bert.encoder_blocks' params.
        encoder_params = model.module.module.bert.encoder.blocks[-getattr(model.module.module.config, 'split_blocks', 4):].state_dict()
        logging.info(model.module.module.bert.encoder.tag_blocks.load_state_dict(encoder_params))

        start_iter = extra_param.get('iteration', 0)

        # use the mask-rcnn trainer engine
        train_loader = self.get_data_loader(
            is_train=True,
            start_iter=start_iter,
        )

        self.do_train(train_loader, model, optimizer, scheduler, checkpointer, start_iter)
        return checkpointer.get_checkpoint_file()

    def get_parameter_groups(self, model):
        lr = self.cfg.base_lr
        from collections import defaultdict
        decay_to_info = defaultdict(lambda: defaultdict(list))
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            weight_decay = self.cfg.weight_decay
            if self.cfg.bias_no_weight_decay and "bias" in key:
                weight_decay = 0.
            if self.cfg.ln_no_weight_decay and 'LayerNorm.weight' in key:
                weight_decay = 0
            if self.cfg.conv_no_weight_decay and key.endswith('conv.weight'):
                weight_decay = 0.
            logging.info('{}: lr = {}; weight_decay = {}'.format(
                key, lr, weight_decay
            ))
            decay_to_info[weight_decay]['params'].append(value)
            decay_to_info[weight_decay]['param_names'].append(key)
        ps = []
        for w, info in decay_to_info.items():
            p = {'weight_decay': w, 'lr': lr}
            p.update(info)
            ps.append(p)
        return ps

    def get_optimizer(self, model):

        # parameters:
        # model.module.image_encoder                           1e-4
        # model.module.module.bert.embeddings                  1e-4
        # model.module.module.bert.encoder.blocks[:-4]         1e-4
        # model.module.module.bert.encoder.blocks[-4:]         1e-4 * M
        # model.module.module.bert.encoder.tag_blocks          1e-4 * M
        # model.module.module.bert.caption_pooler              1e-4
        # model.module.module.bert.pooler                      1e-4 * M
        # model.module.module.bert.tag_logit                   1e-4 * M
        # model.module.module.bert.decoder                     1e-4

        # if extra embedding:
        # model.module.module.bert.extra_embeddings            1e-4

        # Manually lower the LR for CTN blocks as it's already pre-trained with good initialization, use smaller LR.
        # Tune the LR multiplier yield some performances fluctuations. In my exp, 0.1 leads to optimal results than 1.0 and 0.01.
        l = getattr(model.module.module.config, 'split_blocks', 4)
        image_encoder = self.get_parameter_groups(model.module.image_encoder                   )
        embedding = self.get_parameter_groups(model.module.module.bert.embeddings              )
        share_blocks = self.get_parameter_groups(model.module.module.bert.encoder.blocks[:-l]  )
        caption_blocks = self.get_parameter_groups(model.module.module.bert.encoder.blocks[-l:])
        tag_blocks = self.get_parameter_groups(model.module.module.bert.encoder.tag_blocks     )
        caption_pooler = self.get_parameter_groups(model.module.module.bert.caption_pooler     )
        pooler = self.get_parameter_groups(model.module.module.bert.pooler                     )
        tag_logit = self.get_parameter_groups(model.module.module.bert.tag_logit               )
        decoder = self.get_parameter_groups(model.module.module.bert.decoder                   )
        extra_embedding = self.get_parameter_groups(model.module.module.bert.extra_embeddings  )

        # Apply the LR multiplier
        logging.info('LR Updating...')
        for params in [tag_blocks, share_blocks, pooler, tag_logit]:
            for dic in params:
                dic['lr'] *= self.cfg.lr_multiplier

        parameters = image_encoder + embedding + share_blocks \
                     + caption_blocks + tag_blocks + caption_pooler \
                     + pooler + tag_logit + decoder + extra_embedding

        if self.cfg.optimizer_type in ['MAdamW']:
            from src.solver import AdamW
            print('learning rate {}, {}'.format(self.cfg.base_lr, type(self.cfg.base_lr)))

            optimizer = AdamW(parameters,
                              lr=self.cfg.base_lr,
                              eps=1e-8)

        else:
            raise NotImplementedError(self.cfg.optimizer_type)
        return optimizer

    def get_len_dataset(self, is_train):
        if is_train:
            dataset = CaptionIdxTSVDataset(
                data=self.cfg.data,
                split='train',
                caption_version=self.cfg.train_version,
            )
        else:
            dataset = ImageIdxTSVDataset(
                data=self.cfg.test_data,
                split=self.cfg.test_split,
            )

        return dataset

    def get_transform(self, is_train):
        data = self.cfg.data if is_train else self.cfg.test_data
        split = 'train' if is_train else self.cfg.test_split

        all_trans = []
        cache_policy = None
        hw_loader = LoadHW(
            data=data,
            split=split,
            cache_policy=cache_policy,
        )
        all_trans.append(hw_loader)

        max_img_seq_len = self.cfg.max_img_seq_length
        load_feature = max_img_seq_len > 0

        # by default, we don't load detector features
        assert not load_feature

        # load image and we will extract the features online. This is mainly
        # used for end-to-end training or inference.
        image_loader = LoadImage(data, split)

        from src.pipelines.uni_pipeline import get_transform_image
        image_transform = get_transform_image(self, is_train)

        from src.data_layer.transform import ImageTransform2Dict
        image_transform = ImageTransform2Dict(image_transform)

        feature_loader = transforms.Compose([
            image_loader,
            image_transform,
        ])

        all_trans.append(feature_loader)

        if is_train:
            caption_loader = LoadCaption(
                data=data, split=split, version=None,
                cache_policy=cache_policy,
            )
            all_trans.append(caption_loader)

        # if self.cfg.add_od_labels:
        label_loader = LoadLabel(
            data=data, split=split,
            version=self.cfg.train_label_version)

        all_trans.append(label_loader)

        text_ab = IdentifyTextAB(
            False,
            self.cfg.od_label_conf,
            label_sort_by_conf=not self.cfg.no_sort_by_conf,
            unique_labels_on=self.cfg.unique_labels_on,
            qa2caption=None,
            sep_token=self.tokenizer.sep_token,
        )

        all_trans.append(text_ab)

        tensorizer = (self.train_caption_tensorizer if is_train else
                      self.test_caption_tensorizer)

        # use the customized tensorizer
        if not is_train:
            # in test, we have to do this padding, otherwise it will crash
            self.cfg.pad_to_max = True

        trans_tensorizer = TransCaptionTensorizer(
            tensorizer,
            with_img_feats=load_feature,
            pad_to_max=self.cfg.pad_to_max,
            pad_image_to_max=True,
            real_text_a_in_test=self.cfg.real_text_a_in_test
        )
        all_trans.append(trans_tensorizer)

        # Tag-Tensorize
        trans_tensorizer = Tensorizer(self.tagger_tensorizer)
        all_trans.append(trans_tensorizer)

        # Remove some annotations that won't be used.
        if split == 'train':
            useless_keys = [
                'idx',
                'idx_img',
                'idx_cap',
                'dataset',
                # 'label',
                'caption',
                'text_ab_type',
                'text_a',
                # remove the labels as we don't need them in input_ids
                'text_b',
                'width',
                'height',
                'text_changed',
                'text_a_or_b_changed',
                'img_feat',
                'max_seq_a_len',
                'seq_a_padded_len',
                'feats_conf',
                'feats_class',
                'teacher_feats_conf',
                'teacher_feats_class',
                'vocab_size',
                'feats_class_token_ids',
                'feats_class_tokens',
                'origin_input_ids',
            ]
        else:
            # In inference mode, remove the detector tags to prevent the case when accidentally using detector tag during testing.
            useless_keys = [
                'idx',
                'idx_img',
                'idx_cap',
                'dataset',
                'label',
                'caption',
                'text_ab_type',
                'text_a',
                # remove the labels as we don't need them in input_ids
                'text_b',
                'width',
                'height',
                'text_changed',
                'text_a_or_b_changed',
                'img_feat',
                'max_seq_a_len',
                'seq_a_padded_len',
                'feats_conf',
                'feats_class',
                'teacher_feats_conf',
                'teacher_feats_class',
                'vocab_size',
                'feats_class_token_ids',
                'feats_class_tokens',
                'origin_input_ids',
            ]

        all_trans.extend([
            RemoveUselessKeys(useless_keys),
            RenameKey({'segment_ids': 'token_type_ids'}),
        ])
        return transforms.Compose(all_trans)

    def get_fusion_config(self, is_train):
        from src.layers.bert import BertConfig
        config = BertConfig.from_pretrained(
            # this class does not support a separate text encoder and thus
            # text_encoder_type here means the joint fusion encoder
            self.cfg.text_encoder_type,
            num_labels=2,
            finetuning_task='image_captioning',
        )

        if 'vit' in self.cfg.text_encoder_type:
            # this is just to make sure we are using the right variables for vit model
            assert self.cfg.drop_out == 0

        config.img_feature_type = 'frcnn'
        config.hidden_dropout_prob = self.cfg.drop_out
        config.loss_type = 'classification'
        config.tie_weights = self.cfg.tie_weights
        config.freeze_embedding = False
        config.label_smoothing = self.cfg.label_smoothing
        config.drop_worst_ratio = 0
        config.drop_worst_after = 0
        config.img_feature_dim = self.cfg.img_feature_dim
        config.use_img_layernorm = self.cfg.use_img_layernorm
        config.img_layer_norm_eps = self.cfg.img_layer_norm_eps
        config.net = self.cfg.image_encoder_type[len('VitEmb_'):]
        config.ignore_project_image = self.cfg.ignore_project_image
        config.later_captioning = self.cfg.later_captioning
        config.attn_token_sample = self.cfg.attn_token_sample
        config.vocab = self.tag_vocab
        config.tokenizer = self.tokenizer
        config.loss = getattr(self.cfg, 'loss', 'bce')
        config.split_blocks = getattr(self.cfg, 'split_blocks', '4')
        config.topktagger = getattr(self.cfg, 'topktagger', None)
        config.tagemb = getattr(self.cfg, 'tagemb', 'bert')
        config.tagemb_gradient = getattr(self.cfg, 'tagemb_gradient', None)
        config.category = getattr(self.cfg, 'category', 'vinvl')
        config.tie_tag_weights = getattr(self.cfg, 'tie_tag_weights', False)

        if getattr(self.cfg, 'topk') is None:
            config.threshold = self.cfg.threshold
        else:
            config.topk = self.cfg.topk

        return config

    def get_raw_model(self, is_train):
        config = self.get_fusion_config(is_train)
        model = ViTCAP(config=config) # init from scratch

        image_encoder = self.get_image_encoder_model(is_train,)

        if is_train:
            model = ImageCaptioning(model,
                                    image_encoder=image_encoder,
                                    tokenizer=self.tag_tokenizer,
                                    bert_tokenizer=self.tokenizer,
                                    cfg=self.cfg)
        else:
            tokenizer = self.tokenizer
            cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
                tokenizer.convert_tokens_to_ids([
                    tokenizer.cls_token,
                    tokenizer.sep_token,
                    tokenizer.pad_token,
                    tokenizer.mask_token,
                    '.',
                ])
            test_extra_input = {
                'is_decode': True,
                'do_sample': False,
                'bos_token_id': cls_token_id,
                'pad_token_id': pad_token_id,
                'eos_token_ids': [sep_token_id],
                'mask_token_id': mask_token_id,
                # for adding od labels
                'add_od_labels': self.cfg.add_od_labels,
                'od_labels_start_posid': self.cfg.max_seq_a_length,
                # hyper-parameters of beam search
                'max_length': self.cfg.max_gen_length,
                'num_beams': self.cfg.num_beams,
                "temperature": self.cfg.temperature,
                "top_k": self.cfg.top_k,
                "top_p": self.cfg.top_p,
                "repetition_penalty": 1,
                "length_penalty": 1,
                "num_return_sequences": 1,
                "num_keep_best": 1,
            }
            model = ImageCaptioning(
                model,
                test_extra_input,
                tokenizer=self.tokenizer,
                bert_tokenizer=self.tokenizer,
                image_encoder=image_encoder,
                cfg=self.cfg,
            )

        return model

    def predict_output_to_tsv_row(self, data, output):
        all_caps = output[0]  # batch_size * num_keep_best * max_len
        all_confs = torch.exp(output[1])

        for img_key, caps, confs in zip(data['key'], all_caps, all_confs):
            res = []
            for cap, conf in zip(caps, confs):
                cap = self.tokenizer.decode(
                    cap.tolist(), skip_special_tokens=True)
                res.append({'caption': cap, 'conf': conf.item()})
            yield img_key, json.dumps(res)

    def evaluate(self, predict_file, evaluate_file):
        from src.tools.tsv.tsv_io import TSVDataset
        dataset = TSVDataset(self.cfg.test_data)
        json_caption = op.join(
            dataset._data_root,
            self.cfg.test_split + '.caption_coco_format.json')
        if not op.isfile(json_caption):
            from src.tools.tsv.tsv_io import iter_caption_to_json
            iter_caption_to_json(
                dataset.iter_data(
                    self.cfg.test_split, 'caption'),
                json_caption)
        from src.tools.captioning.utils_caption_evaluate import evaluate_on_coco_caption
        result = evaluate_on_coco_caption(predict_file, json_caption, outfile=evaluate_file)
        logging.info('evaluation result: {}'.format(str(result)))
        logging.info('evaluation result saved to {}'.format(evaluate_file))

    @property
    def taggertokenizer(self):
        if self._taggertokenizer is None:
            assert op.isfile(self.cfg.tokenizer_file)
            taggertokenizer = json.load(open(self.cfg.tokenizer_file))
            self._taggertokenizer = taggertokenizer
        return self._taggertokenizer

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from src.layers.bert import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained(
                self.cfg.text_encoder_type, do_lower_case=True)
            self._tokenizer = tokenizer
        return self._tokenizer

    @property
    def tag_tokenizer(self):
        assert op.isfile(self.cfg.tokenizer_file)
        tokenizer = json.load(open(self.cfg.tokenizer_file))
        self._tag_tokenizer = tokenizer
        return self._tag_tokenizer

    @property
    def tagger_tensorizer(self):

        from src.data_layer.dataset import CaptionTaggerTensorizer
        tensorizer = CaptionTaggerTensorizer(
            self.tag_tokenizer,
            self.tokenizer,
            self.cfg.od_label_conf,
            category=self.cfg.category,
            encode=getattr(self.cfg, 'encode', 'nltk'),
            caption_only=self.cfg.caption_only,
        )
        self._train_tag_tensorizer = tensorizer
        return self._train_tag_tensorizer

    @property
    def tag_vocab(self):
        assert op.isfile(self.cfg.tokenizer_file)
        tag_vocab = json.load(open(self.cfg.tokenizer_file))
        return tag_vocab

    @property
    def train_caption_tensorizer(self):
        if self._train_caption_tensorizer is None:

            from src.data_layer.dataset import CaptionTensorizer
            caption_tensorizer = CaptionTensorizer(
                self.tokenizer,
                max_img_seq_length=self.cfg.max_img_seq_length,
                max_seq_length=self.cfg.max_seq_length,
                max_seq_a_length=self.cfg.max_seq_a_length,
                mask_prob=self.cfg.mask_prob,
                max_masked_tokens=self.cfg.max_masked_tokens,
                mask_type=self.cfg.mask_type,
                is_train=True,
                mask_b=False,
                replace_by_mask_prob=self.cfg.replace_by_mask_prob,
                replace_by_rand_prob=self.cfg.replace_by_rand_prob,
                output_isvalid=self.cfg.output_isvalid,
                mask_token_by_word_in_train=self.cfg.mask_token_by_word_in_train
            )
            self._train_caption_tensorizer = caption_tensorizer
        return self._train_caption_tensorizer

    @property
    def train_tag_tensorizer(self):
        if self._train_caption_tensorizer is None:
            self._train_caption_tensorizer = 1
        return self._train_caption_tensorizer

    @property
    def test_caption_tensorizer(self):
        if self._test_caption_tensorizer is None:
            max_seq_length = self.cfg.max_seq_length if self.cfg.add_od_labels else self.cfg.max_gen_length
            max_od_labels_len = self.cfg.max_seq_length - self.cfg.max_seq_a_length
            max_seq_length = self.cfg.max_gen_length + max_od_labels_len

            from src.data_layer.dataset import CaptionTensorizer
            caption_tensorizer = CaptionTensorizer(
                self.tokenizer,
                max_img_seq_length=self.cfg.max_img_seq_length,
                max_seq_length=max_seq_length,
                max_seq_a_length=self.cfg.max_gen_length,
                is_train=False,
            )
            self._test_caption_tensorizer = caption_tensorizer
        return self._test_caption_tensorizer

    def append_predict_param(self, cc):
        super().append_predict_param(cc)

        if self.cfg.max_img_seq_length not in [0, 50]:
            # if it is 0, normally the strucutre is quite different, and there
            # is no need to specify it here
            cc.append('image_region{}'.format(self.cfg.max_img_seq_length))

    # VitEmb_vit_base_patch32_384
    def get_image_encoder_model(self, is_train):
        if self.cfg.image_encoder_type.startswith('VitEmb_'):
            # VitEmb_base32_384
            net = self.cfg.image_encoder_type[len('VitEmb_'):]

            if self.cfg.image_encoder_pretrained:
                logging.info('VIT image encoder loaded from pre-trained weight!  '
                             'Note that this might be replaced by pre-trained checkpoint later!')
            from src.pytorch_image_models import timm

            logging.info('Non-Patch Selection Mode.')
            model = timm.create_model(
                net,
                output_grid=True,
                pretrained=self.cfg.image_encoder_pretrained,
            )

            # clear out the following two modules
            model.norm = nn.Identity()
            model.blocks = nn.ModuleList()
            if not is_train:
                model.eval()

            from src.tools.torch_common import InputAsDict
            model = InputAsDict(model)

        else:
            raise NotImplementedError(self.cfg.image_encoder_type)
        return model

