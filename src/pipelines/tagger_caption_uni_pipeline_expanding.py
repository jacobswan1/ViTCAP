import json
from src.tools.metric import *
from src.tools.torch_common import *
from torchvision.transforms import transforms
from src.data_layer.builder import collate_fn
from src.pipelines.uni_pipeline import UniPipeline
from src.data_layer.dataset import TransCaptionTensorizer
from src.data_layer.dataset import Tensorizer, pert_collate_fn
from src.layers.bert import TaggerEncDecSplitForImageCaptioning
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


def label_to_label(logit, vocab, category='vinvl'):
    '''convert logit to labels.'''
    labels = []

    if category == 'vinvl':
        # iterate all samples
        for pred in logit:
            p = [vocab['idx_to_label'][str(idx)] for idx, t in enumerate(pred) if t == 1 ]
            labels.append(p)
    else:
        # iterate all samples
        for pred in logit:
            p = [vocab[idx] for idx, t in enumerate(pred) if t == 1 ]
            labels.append(p)

    # alphabetic order for better view
    # labels.sort()
    return labels


def logit_to_label(logit, vocab, topk=20, threshold=None, category='vinvl'):
    '''convert logit to labels.'''
    labels = []

    with torch.no_grad():
        logit = torch.nn.functional.sigmoid(logit)

    if threshold is not None:
        topk = (logit >= threshold).sum()

    topk, pred_topk = logit.topk(topk, dim=1, largest=True)

    # iterate all samples
    for pred in pred_topk:
        if category == 'vinvl':
            p = [vocab[str(int(t.cpu().numpy()))] for t in pred]
        else:
            p = [vocab[int(t.cpu().numpy())] for t in pred]

        labels.append(p)

    # alphabetic order for better view
    # labels.sort()
    return labels


class TokenSample(nn.Module):
    '''
    this code is for random token sampling.
    '''
    def __init__(self):
        super().__init__()
        # self.num = num

    def forward(self, x, num):
        B, N = x.shape[:2]

        if N <= num:
            return x

        # by default, we always include the first one, which is [CLS] token
        zero = torch.zeros((1,), dtype=torch.int64)
        ys = [x[b][torch.cat((zero, torch.randperm(N - 1)[:(num-1)] + 1))] for b in range(B)]

        return torch.stack(ys)


def convert_vit_cls_model_to_caption(cls_model, cap_model):
    # this function is used to convert cls_uni_pipeline model to captioning
    # model where there is no seperate image encoder, so that we can load this
    # pre-trained model
    cls_model = torch_load(cls_model)
    model = cls_model['model']

    prefix = 'module.'
    out = {}
    for k, v in model.items():
        while k.startswith(prefix):
            k = k[len(prefix):]
        if k.startswith('blocks.'):
            k = 'module.bert.encoder.' + k
        else:
            k = 'image_encoder.module.' + k
        out[k] = v
    torch_save(out, cap_model)


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, -1)[1].data # argmax
    return logits == labels


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Accuracy(object):
    """ base class for accuracy calculation
    """

    def __init__(self):
        pass

    def calc(self, output, target):
        pass

    def prec(self):
        pass

    def result_str(self):
        pass


class MultiLabelAccuracy(Accuracy):
    """ class for multi label accuracy calculation
    """

    def __init__(self):
        self.accuracy = AverageMeter()

    # @profile
    def calc(self, output, target, fast=False):
        """Computes the precision of multi label prediction"""
        with torch.no_grad():
            if target.size(1) > 1:
                num_labels = target.sum(dim=1)
                valid_indices = torch.nonzero(num_labels)

                maxk = num_labels.max().int().item()
                if fast:
                    maxk = min(maxk, 10)

                maxk = max(1, maxk)
                topk, pred_topk = output.topk(maxk, dim=1, largest=True)

                n = valid_indices.size(0)
                pred = torch.zeros_like(output).cuda()

                if fast:
                    pred = pred.scatter(1, pred_topk, 1)
                else:
                    for i in range(n):
                        sample_index = valid_indices[i].item()
                        k = num_labels[sample_index].int().item()
                        pred[sample_index, pred_topk[sample_index, :k]] = 1

                pred = pred * target
                correct = pred.sum(dim=1)
                accuracy = correct[valid_indices] * 100. / num_labels[valid_indices]
                accuracy = accuracy.sum(dim=0).item()

                if n > 0:
                    accuracy /= n
                    self.accuracy.update(accuracy, n)
            else:
                pos_positive_count = ((output * target) > 0).sum()
                pos_negative_count = ((output * (torch.ones_like(target).cuda() - target)) < 0).sum()
                good_count = pos_positive_count + pos_negative_count
                n = output.size(0)
                accuracy = good_count * 100. / n
                self.accuracy.update(accuracy, n)

    def prec(self):
        return self.accuracy.avg

    def val_sum(self):
        return self.accuracy.sum

    def total_count(self):
        return self.accuracy.count

    def result_str(self):
        return 'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(acc=self.accuracy)


class ImageCaptioning(nn.Module):
    def __init__(self,
                 model,
                 test_extra_input=None,
                 tokenizer=None,
                 bert_tokenizer=None,
                 image_encoder=None,
                 caption_loader=None,
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
        self.caption_loader = caption_loader

        if self.cfg.scst:
            from src.qd.mask.modeling.captioning.utils_caption_evaluate import ScstRewardCriterion
            self.scst_criterion = ScstRewardCriterion(
                cider_cached_tokens=op.join('.', 'data/coco_caption/gt/coco-train-words.p'),
            )
            logging.info("  SCST training...")

        if cfg.pert_img_prob is not None and cfg.pert_img_prob > 0:
            # we need an image text matching loss on the pooled output
            # number of relationship is always 1, we use BCE loss
            self.seq_relationship = nn.Linear(model.bert.pooler.dense.weight.shape[0], 1)
            assert self.cfg.mask_type != 'seq2seq', 'matching loss is useless'
        else:
            self.seq_relationship = None

        # random drop visual tokens for faster training
        self.random_sampler = None
        if cfg.random_token_sample is not None:
            self.random_sampler = TokenSample()

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

        # check if checkpoint param is changed
        # init_param = torch.load('./output/TaxCocoCaption_B_Vilt_captioning_testing/snapshot/vitbfocal10.pt')['model']
        # now_param = self.module.state_dict()
        # keys = now_param.keys()
        # for key in keys:
        #     if 'module.module.'+key in init_param:
        #         print(key, (init_param['module.module.'+key] - now_param[key]).sum())
        #     else:
        #         print(key, 'not initialized.')
        #
        #
        # now_param = self.image_encoder.state_dict()
        # keys = now_param.keys()
        # for key in keys:
        #     if 'module.image_encoder.'+key in init_param:
        #         print(key, (init_param['module.image_encoder.'+key] - now_param[key]).sum())
        #     else:
        #         print(key, 'not initialized.')

        data = dict(data.items())
        # this is required in test, but not in train
        data.pop('key')
        image_keys = data.pop('idx_img')

        if self.image_encoder:
            assert 'img_feats' not in data
            data['img_feats'] = self.image_encoder(data)

            # Random token drop
            if self.random_sampler and self.cfg.random_token_sample != 1.0:
                if self.training:
                    data['img_feats'] = self.random_sampler(data['img_feats'],
                                                            num=int(data['img_feats'].shape[1]
                                                                    *self.cfg.random_token_sample))
                else:
                    if not self.cfg.train_random_sample_only:
                        data['img_feats'] = self.random_sampler(data['img_feats'],
                                                                num=int(data['img_feats'].shape[1]
                                                                        * self.cfg.random_token_sample))

            self.construct_attn_mask(data)

            if 'image' in data: data.pop('image')
            if 'image2' in data: data.pop('image2')
            if 'image3' in data: data.pop('image3')
            if 'image_ori' in data: data.pop('image_ori')
            if 'image2_ori' in data: data.pop('image2_ori')
            if 'image3_ori' in data: data.pop('image3_ori')

        if self.training:

            if not self.cfg.scst:

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

                        logging.info('Input ids sample: {}'.format(str([self.bert_tokenizer.ids_to_tokens[token]
                                                                        for token in data['input_ids'][0]
                                                                       .detach().cpu().numpy()])))

                        logging.info('Sample Generation: {}'.format(str(logit_to_label(result['tag_logits'].detach(),
                                                                                       self.vocab,
                                                                                       topk=50,
                                                                                       category=self.category)[0])))

                        logging.info('GT Tags: {}'.format(str(label_to_label(data['label'], self.vocab, category=self.category)[0]), ))

                # for mismatched pair, we ignore the masked token loss in the
                # self.module, where we clear out the masked pos. That logic can be
                # moved to collate function also.

                # FIXME: use 10X multiplier to the caption loss.
                loss_dict['masked_loss'] = result['masked_loss']
                loss_dict['tag_loss'] = result['tag_loss']
                return loss_dict

            # SCST loss
            else:
                cls_token_id, sep_token_id, pad_token_id, mask_token_id = \
                    self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token,
                                                          self.tokenizer.sep_token,
                                                          self.tokenizer.pad_token,
                                                          self.tokenizer.mask_token]
                                                         )

                inputs = {'is_decode': True,
                          'input_ids': data['input_ids'],
                          'attention_mask': data['attention_mask'],
                          'token_type_ids': data['token_type_ids'],
                          'img_feats': data['img_feats'],
                          'masked_pos': data['masked_pos'],
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
                          'num_beams': 1,
                          "temperature": 1,
                          "top_k": 0,
                          "top_p": 1,
                          "repetition_penalty": 1,
                          "length_penalty": 1,
                          "num_return_sequences": 1,
                          "num_keep_best": 1,
                          "label": data['label'],
                          }

                def _ids_to_captions(all_ids):
                    captions = []
                    for ids in all_ids:
                        c = self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
                        captions.append(c)
                    return captions

                # if self.cfg.sc_baseline_type == 'greedy':
                self.module.eval()
                with torch.no_grad():
                    greedy_res_raw, _ = self.module(**inputs)
                    greedy_res_raw.squeeze_(1)  # batch_size * max_len

                greedy_res = _ids_to_captions(greedy_res_raw)

                self.module.train()
                inputs['do_sample'] = True
                assert self.cfg.scst_num_return is not None
                inputs['num_return_sequences'] = self.cfg.scst_num_return
                sample_res_raw, sample_logprobs = self.module(**inputs)
                sample_res_raw.squeeze_(1)
                sample_logprobs.squeeze_(1)
                assert sample_logprobs.requires_grad == True
                assert sample_res_raw.requires_grad == False
                sample_res = _ids_to_captions(sample_res_raw)

                # sample K GT captions from the annotations
                gt_res = [self.caption_loader.get_captions_by_key(k) for k in image_keys]
                loss = self.scst_criterion(gt_res, greedy_res, sample_res, sample_logprobs)

                verbose = (self.iter % 100) == 0
                self.iter += 1

                if verbose:
                    batch_acc = self.scst_criterion.get_score()
                    logging.info('acc = {}, sample generation: {}'.format(batch_acc, greedy_res))

                loss_dict = {'scst_loss': loss}
            return loss_dict

        else:
            data.update(self.test_extra_input)
            result = self.module(**data)


            return result

    def calc_image_text_matching_loss(self, result, matched):
        logits = self.seq_relationship(result['pooled_output'])
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, matched.float().reshape(logits.shape))


def pert_collate_fn(batch, prob):
    batch = collate_fn(batch)
    num_image = batch['image'].shape[0]
    # by expectation, there is one image which stays the same, and thus, we add
    # 1 here to compensate
    shuffle_len = int(num_image * prob) + 1
    idx = torch.cat((torch.randperm(shuffle_len),
               torch.arange(shuffle_len, num_image)))
    batch['image'] = batch['image'][idx]
    batch['matched'] = idx == torch.arange(num_image)
    return batch


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
        # self._berttokenizer = None
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

        max_seq_length = self.cfg.max_seq_length

        # leave some space for decoded tags
        # if not self.cfg.add_od_labels:
        #     assert self.cfg.max_seq_a_length == max_seq_length
        # else:
            # assert self.cfg.max_seq_a_length == 40

    def train(self):
        model = self.get_model(is_train=True)

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
        encoder_params = model.module.module.bert.encoder.blocks[-getattr(model.module.module.config,
                                                                          'split_blocks', 4):].state_dict()
        logging.info(model.module.module.bert.encoder.tag_blocks.load_state_dict(encoder_params))

        start_iter = extra_param.get('iteration', 0)

        # logging.info('tag split branch initialization result:', scheduler)

        # use the maskrcnn trainer engine
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


        # apply the LR multiplier
        logging.info('LR Updating...')
        for params in [tag_blocks, share_blocks, pooler, tag_logit]:
            for dic in params:
                dic['lr'] *= self.cfg.lr_multiplier

        parameters = image_encoder + embedding + share_blocks \
                     + caption_blocks + tag_blocks + caption_pooler \
                     + pooler + tag_logit + decoder

        if self.cfg.optimizer_type in ['MAdamW']:
            from src.qd.mask.solver import AdamW
            print('learning rate {}, {}'.format(self.cfg.base_lr, type(self.cfg.base_lr)))

            optimizer = AdamW(parameters,
                              lr=self.cfg.base_lr,
                              eps=1e-8)

        else:
            raise NotImplementedError(self.cfg.optimizer_type)

        return optimizer

    # def get_lr_scheduler(self, optimizer):
    #     # only support linear mode, expand it if needed.
    #     scheduler_type = self.cfg.scheduler_type
    #     if scheduler_type == "linear":
    #         from src.qd.mask.solver import WarmupLinearSchedule
    #         scheduler = WarmupLinearSchedule(
    #             optimizer,
    #             warmup_steps=self.parse_iter(self.cfg.warmup_steps),
    #             t_total=self.max_iter,
    #         )
    #     else:
    #         raise NotImplementedError(scheduler_type)
    #     return scheduler


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

        # load image and we will extract the features online. This is mainly
        # used for end-to-end training or inference.
        image_loader = LoadImage(data, split)
        from src.pipelines.uni_pipeline import get_transform_image

        #FIXME: !!!!!
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

        useless_keys = [
            'idx',
            # 'idx_img',
            'idx_cap',
            'dataset',
            # 'label',
            'caption',
            'text_ab_type',
            'text_a',
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
            # this class does not support a seperate text encoder and thus
            # text_encoder_type here means the joint fusion encoder
            self.cfg.text_encoder_type,
            num_labels=2,
            finetuning_task='image_captioning',
        )

        if 'vit' in self.cfg.text_encoder_type:
            # this is just to make sure we are using the right variables for
            # vit model
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
        config.bertembtagger = getattr(self.cfg, 'bertembtagger', None)
        config.category = getattr(self.cfg, 'category', 'vinvl')
        config.tie_tag_weights = getattr(self.cfg, 'tie_tag_weights', False)
        config.tagemb = getattr(self.cfg, 'tagemb', 'bert')
        config.tagemb_gradient = getattr(self.cfg, 'tagemb_gradient', None)

        if getattr(self.cfg, 'topk') is None:
            config.threshold = self.cfg.threshold
        else:
            config.topk = self.cfg.topk

        return config

    def get_raw_model(self, is_train):
        config = self.get_fusion_config(is_train)
        model = TaggerEncDecSplitForImageCaptioning(config=config) # init from scratch

        image_encoder = None
        # if self.cfg.max_img_seq_length == 0:
        image_encoder = self.get_image_encoder_model(is_train,)

        if is_train:

            if self.cfg.scst:
                data = self.cfg.data if is_train else self.cfg.test_data
                split = 'train' if is_train else self.cfg.test_split
                caption_loader = LoadCaption(
                    data=data,
                    split=split,
                    version=None,
                    cache_policy=None,
                )
                model = ImageCaptioning(model,
                                    image_encoder=image_encoder,
                                    caption_loader=caption_loader,
                                    tokenizer=self.tokenizer,
                                    bert_tokenizer=self.tokenizer,
                                    cfg=self.cfg)
            else:
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
            from src.qd.process_tsv import iter_caption_to_json
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
            from src.qd.mask.layers.bert import BertTokenizer
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
        )
        self._train_tag_tensorizer = tensorizer
        return self._train_tag_tensorizer

    @property
    def tag_vocab(self):
        assert op.isfile(self.cfg.tokenizer_file)
        tag_vocab = json.load(open(self.cfg.tokenizer_file))
        # self._tag_vocab = tag_vocab
        return tag_vocab

    @property
    def train_caption_tensorizer(self):
        if self._train_caption_tensorizer is None:
            from src.qd.mask.data.datasets.caption_tensorizer import CaptionTensorizer
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
            from src.qd.mask.data.datasets.caption_tensorizer import CaptionTensorizer
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

            # print(net)
            # print('timm.create_model ->')
            # print(model.state_dict().keys())
            # print(model.state_dict()['patch_embed.proj.weight'])

            # clear out the following two modules
            model.norm = nn.Identity()
            model.blocks = nn.ModuleList()
            if not is_train:
                model.eval()

            from src.qd.torch_common import InputAsDict
            model = InputAsDict(model)

        else:
            raise NotImplementedError(self.cfg.image_encoder_type)
        return model


class Tensorizer(object):
    def __init__(self, tensorizer):
        self.tensorizer = tensorizer

    def __call__(self, data):
        # VinVL predictions dict is like: {"image_h": 480, "image_w": 640, "num_boxes": 43, "objects": [{"class": "sky", "conf": 0.774634838104248,
        # "rect": [39.00554275512695, 31.809423446655273, 639.2000122070312, 327.2917175292969]}], "predicates": [], "relations": []}
        if 'objects' in data['label']:
            labels = data['label']['objects']
        # others are like: {"class": "sky", "conf": 0.774634838104248,
        # "rect": [39.00554275512695, 31.809423446655273, 639.2000122070312, 327.2917175292969]}
        else:
            labels = data['label']

        if 'caption' in data.keys():
            x = self.tensorizer.tensorize(labels, data['caption']['caption'])
        else:
            x = self.tensorizer.tensorize(labels)

        data.update(x)
        return data


import math


class Meter(object):
    '''Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    '''

    def reset(self):
        '''Resets the meter to default settings.'''
        pass

    def add(self, value):
        '''Log a new value to the meter
        Args:
            value: Next restult to include.
        '''
        pass

    def value(self):
        '''Get the value of the meter in the current state.'''
        pass


class APMeter(Meter):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self):
        super(APMeter, self).__init__()
        self.reset()

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())
        self.weights = torch.FloatTensor(torch.FloatStorage())

    def add(self, output, target, weight=None):
        """Add a new observation
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                (eg: a row [0, 1, 0, 1] indicates that the example is
                associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if weight is not None:
            if not torch.is_tensor(weight):
                weight = torch.from_numpy(weight)
            weight = weight.squeeze()
        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if weight is not None:
            assert weight.dim() == 1, 'Weight dimension should be 1'
            assert weight.numel() == target.size(0), \
                'Weight dimension 1 should be the same as that of target'
            assert torch.min(weight) >= 0, 'Weight should be non-negative only'
        assert torch.equal(target**2, target), \
            'targets should be binary (0 or 1)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            new_weight_size = math.ceil(self.weights.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))
            if weight is not None:
                self.weights.storage().resize_(int(new_weight_size + output.size(0)))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

        if weight is not None:
            self.weights.resize_(offset + weight.size(0))
            self.weights.narrow(0, offset, weight.size(0)).copy_(weight)

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        if hasattr(torch, "arange"):
            rg = torch.arange(1, self.scores.size(0) + 1).float()
        else:
            rg = torch.range(1, self.scores.size(0)).float()
        if self.weights.numel() > 0:
            weight = self.weights.new(self.weights.size())
            weighted_truth = self.weights.new(self.weights.size())

        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            _, sortind = torch.sort(scores, 0, True)
            truth = targets[sortind]
            if self.weights.numel() > 0:
                weight = self.weights[sortind]
                weighted_truth = truth.float() * weight
                rg = weight.cumsum(0)

            # compute true positive sums
            if self.weights.numel() > 0:
                tp = weighted_truth.cumsum(0)
            else:
                tp = truth.float().cumsum(0)

            # compute precision curve
            precision = tp.div(rg)

            # compute average precision
            ap[k] = precision[truth.bool()].sum() / max(float(truth.sum()), 1)
        return ap


class mAPMeter(Meter):
    """
    The mAPMeter measures the mean average precision over all classes.
    The mAPMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self):
        super(mAPMeter, self).__init__()
        self.apmeter = APMeter()

    def reset(self):
        self.apmeter.reset()

    def add(self, output, target, weight=None):
        self.apmeter.add(output, target, weight)

    def value(self):
        return self.apmeter.value().mean()