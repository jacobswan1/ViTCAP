import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.functional import pad
from src.tools.common import get_mpi_size as get_world_size

from .scan_utils import EncoderFeature, EncoderText, ContrastiveLoss


class SCANEmbedding(object):
    """
    Structure that holds SCAN embeddings and provides .to function to
    be able to move all necessary tensors between gpu and cpu.
    """

    def __init__(self, img_emb, img_length, cap_emb, cap_length):
        self.img_emb = img_emb
        self.img_length = img_length
        self.cap_emb = cap_emb
        self.cap_length = cap_length

    def to(self, *args, **kwargs):
        cast_img_emb = self.img_emb.to(*args, **kwargs)
        cast_cap_emb = self.cap_emb.to(*args, **kwargs)
        cast_img_length = self.img_length.to(*args, **kwargs)
        cast_cap_length = self.cap_length.to(*args, **kwargs)
        return SCANEmbedding(cast_img_emb, cast_img_length,
                             cast_cap_emb, cast_cap_length)


class SCAN(nn.Module):
    def __init__(self, cfg, bbox_proposal_model=None):
        super(SCAN, self).__init__()

        if cfg.MODEL.RPN_ONLY:
            raise ValueError("SCAN model can't operate in RPN_ONLY regime, "
                             "since it requires an object detection head")
        if bbox_proposal_model:
            self.img_dim = bbox_proposal_model.roi_heads.box.feature_dim
        else:
            self.img_dim = cfg.MODEL.SCAN.IMG_FEATURES_DIM

        self.img_enc = EncoderFeature(
            self.img_dim, cfg.MODEL.SCAN.EMBED_SIZE,
            precomp_enc_type=cfg.MODEL.SCAN.PRECOMP_ENC_TYPE,
            no_featnorm=cfg.MODEL.SCAN.NO_IMG_NORM,
        )

        self.text_features_as_input = cfg.MODEL.SCAN.TEXT_FEATURES_AS_INPUT
        if self.text_features_as_input:
            self.text_dim = cfg.MODEL.SCAN.TEXT_FEATURES_DIM
        else:
            self.text_dim = cfg.MODEL.SCAN.VOCAB_SIZE

        self.txt_enc = EncoderText(
            self.text_dim, cfg.MODEL.SCAN.WORD_DIM,
            cfg.MODEL.SCAN.EMBED_SIZE, cfg.MODEL.SCAN.NUM_LAYERS,
            use_bi_gru=cfg.MODEL.SCAN.BI_GRU,
            no_txtnorm=cfg.MODEL.SCAN.NO_TXT_NORM,
            features_as_input=cfg.MODEL.SCAN.TEXT_FEATURES_AS_INPUT,
        )
        self.criterion = ContrastiveLoss(
            opt=cfg.MODEL.SCAN,
            margin=cfg.MODEL.SCAN.MARGIN,
            max_violation=cfg.MODEL.SCAN.MAX_VIOLATION,
        )
        self.bbox_proposal_model = bbox_proposal_model
        self.freeze_backbone = cfg.MODEL.SCAN.FREEZE_BACKBONE
        self.use_precomputed_boxes = cfg.MODEL.SCAN.BBOX_AS_INPUT
        self.device = cfg.MODEL.DEVICE
        self.average_loss = cfg.MODEL.SCAN.AVERAGE_LOSS
        self.random_boxes = cfg.MODEL.SCAN.RANDOM_BOXES

    def forward(self, images, targets):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets: Tensor containing padded captions information
                (and maybe also precomputed bounding boxes)
        Returns:
            contrastive_loss in the training regime and tuple of image and
            caption embeddings in the test/evaluation regime.
        """
        images = [image.to(self.device) for image in images]
        if self.use_precomputed_boxes:
            targets_transposed = list(zip(*targets))
            captions = targets_transposed[0]
            boxes = targets_transposed[1]
            if self.random_boxes:
                for boxlist in boxes:
                    for idx, box in enumerate(boxlist.bbox):
                        x = np.random.random() * boxlist.size[0]
                        y = np.random.random() * boxlist.size[1]
                        w = np.random.random() * (boxlist.size[0] - x)
                        h = np.random.random() * (boxlist.size[1] - y)
                        boxlist.bbox[idx] = torch.tensor([x, y, x + w, y + h])
            boxes = [box.to(self.device) for box in boxes]

            force_boxes = True
        else:
            captions = targets
            boxes = None
            force_boxes = False

        captions = [caption.to(self.device) for caption in captions]
        if self.text_features_as_input:
            captions = [caption.reshape(-1, self.text_dim) for caption in captions]
        # ideally this processing should be moved to collate_fn in datalayer,
        # but that would involve changing main dataset building code
        lengths = np.array([len(cap) for cap in captions])
        max_tokens = lengths.max()
        if self.text_features_as_input:
            def pad_features(features, max_length):
                length = features.shape[0]
                if length == 0:
                    return torch.zeros((max_length, self.text_dim),
                                       dtype=torch.float32, device=self.device)
                return pad(features, [0, 0, 0, max_length - length], 'constant', 0)

            captions = torch.stack(tuple(
                pad_features(caption, max_tokens) for caption in captions
            ))
        else:
            captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True)
        # sorting data on caption length to use pack_padded_sequence
        sorted_indices = np.argsort(-lengths)
        lengths = lengths[sorted_indices]
        # no matter the input is token or feature,
        # captions now is a torch tensor with batch_first=True
        captions = captions[sorted_indices]
        images = [images[i] for i in sorted_indices]
        if self.use_precomputed_boxes:
            boxes = [boxes[i] for i in sorted_indices]

        cap_emb, cap_lens = self.txt_enc(
            captions,
            torch.tensor(lengths, dtype=torch.int64, device=self.device),
        )
        cap_lens = cap_lens.to(self.device)

        if self.bbox_proposal_model is not None:
            # remembering the current mode to restore it later
            detection_training = self.bbox_proposal_model.training

            force_boxes_model = self.bbox_proposal_model.force_boxes
            force_boxes_box = self.bbox_proposal_model.roi_heads.box.force_boxes
            self.bbox_proposal_model.force_boxes = force_boxes
            self.bbox_proposal_model.roi_heads.box.force_boxes = force_boxes
            self.bbox_proposal_model.roi_heads.box.post_processor.force_boxes = force_boxes

            if self.freeze_backbone:
                self.bbox_proposal_model.eval()
                with torch.no_grad():
                    predictions = self.bbox_proposal_model(
                        images, targets=boxes
                    )
                # restoring the mode of detection model
                if detection_training is True:
                    self.bbox_proposal_model.train()
            else:
                # TODO: consider making a separate parameter to run model in
                #       inference mode
                # setting the whole model to eval mode to ensure we get test-regime
                # proposals. However, since we are going to train backbone and
                # box head feature extractor, we want to keep them in the training
                # regime (to ensure that e.g. batch norm behaves correctly).
                backbone_training = self.bbox_proposal_model.backbone.training
                self.bbox_proposal_model.eval()
                if self.training:
                    self.bbox_proposal_model.backbone.train()
                    self.bbox_proposal_model.roi_heads.box.feature_extractor.train()
                predictions = self.bbox_proposal_model(images, targets=boxes)
                # restoring the mode of detection model
                if detection_training is True:
                    self.bbox_proposal_model.train()
                # restoring the mode of backbone and box head
                if backbone_training is False:
                    self.bbox_proposal_model.backbone.eval()
                    self.bbox_proposal_model.roi_heads.box.feature_extractor.eval()
            predictions = [pred.get_field('box_features') for pred in predictions]

            self.bbox_proposal_model.force_boxes = force_boxes_model
            self.bbox_proposal_model.roi_heads.box.force_boxes = force_boxes_box
            self.bbox_proposal_model.roi_heads.box.post_processor.force_boxes = force_boxes_box
        else:
            # if bbox_proposal_model is None, dataset has to yield features
            # for rpn proposals instead of images
            predictions = [features.reshape(-1, self.img_dim)
                           for features in images]
        num_proposals = torch.tensor([len(pred) for pred in predictions],
                                     dtype=torch.int64, device=self.device)
        max_proposals = num_proposals.max()

        def pad_features(features, max_length):
            length = features.shape[0]
            if length == 0:
                return torch.zeros((max_length, self.img_dim),
                                   dtype=torch.float32, device=self.device)
            return pad(features, [0, 0, 0, max_length - length], 'constant', 0)

        image_features = torch.stack(tuple(
            pad_features(pred, max_proposals) for pred in predictions
        ))

        img_emb = self.img_enc(image_features, num_proposals)

        if self.training:
            # in distributed setting, we need to aggregate all embeddings
            # before computing loss, since SCAN loss depends on all elements
            # in the batch.
            # note that this code will compute exactly the same loss on each
            # GPU. This can potentially be optimized to parallel the computation
            # of the loss, but since that's not the bottleneck of the model
            # we do not try to do that for now.
            world_size = get_world_size()

            if world_size > 1:
                # need to make sure batch size is the same on all processes,
                # since for the last batch in epoch it might be different;
                # if it's different, we will cut everything to the smallest
                # batch size across all processes
                batch_size = torch.tensor(img_emb.shape[0], device=self.device)
                batch_size_full = [torch.zeros_like(batch_size)
                                   for _ in range(world_size)]
                dist.all_gather(batch_size_full, batch_size)

                # cutting all data to min batch size across all GPUs
                min_bs = min([bs.item() for bs in batch_size_full])
                if min_bs < batch_size:
                    num_proposals = num_proposals[:min_bs]
                    cap_lens = cap_lens[:min_bs]
                    img_emb = img_emb[:min_bs]
                    cap_emb = cap_emb[:min_bs]

                # exchanging proposals
                cap_lens_full = [torch.zeros_like(cap_lens)
                                 for _ in range(world_size)]
                num_proposals_full = [torch.zeros_like(num_proposals)
                                      for _ in range(world_size)]

                dist.all_gather(cap_lens_full, cap_lens)
                dist.all_gather(num_proposals_full, num_proposals)
                cap_lens = torch.cat(cap_lens_full, dim=0)
                num_proposals = torch.cat(num_proposals_full, dim=0)

                # before exchanging embeddings, need to pad them
                # to be of the same size
                def pad_features(features, max_length):
                    length = features.shape[1]
                    return pad(features, [0, 0, 0, max_length - length, 0, 0],
                               'constant', 0)

                img_emb = pad_features(img_emb, num_proposals.max().item())
                cap_emb = pad_features(cap_emb, cap_lens.max().item())
                img_emb_full = [torch.zeros_like(img_emb)
                                for _ in range(world_size)]
                cap_emb_full = [torch.zeros_like(cap_emb)
                                for _ in range(world_size)]
                dist.all_gather(img_emb_full, img_emb)
                dist.all_gather(cap_emb_full, cap_emb)
                # need to do this to restore propagation of the gradients
                rank = dist.get_rank()
                img_emb_full[rank] = img_emb
                cap_emb_full[rank] = cap_emb
                img_emb = torch.cat(img_emb_full, dim=0)
                cap_emb = torch.cat(cap_emb_full, dim=0)

            losses = {
                'contrastive_loss': self.criterion(img_emb, num_proposals,
                                                   cap_emb, cap_lens),
            }
            if self.average_loss:
                losses['contrastive_loss'] /= img_emb.shape[0]
            return losses
        # in the evaluation we need to return things in the correct order
        # in addition we restructure everything to return all results for
        # each image separately
        orig_indices = np.argsort(sorted_indices)
        return [
            SCANEmbedding(img_emb, img_len, cap_emb, cap_len)
            for img_emb, img_len, cap_emb, cap_len in zip(
                img_emb[orig_indices], num_proposals[orig_indices],
                cap_emb[orig_indices], cap_lens[orig_indices],
            )
        ]

    def unused_params(self):
        pms = {}
        if self.bbox_proposal_model is None:
            return pms
        for key, param in self.bbox_proposal_model.named_parameters():
            if self.freeze_backbone:
                pms['bbox_proposal_model.{}'.format(key)] = param
            if not self.freeze_backbone and 'backbone' not in key \
                    and 'box.feature_extractor' not in key:
                pms['bbox_proposal_model.{}'.format(key)] = param
        return pms
