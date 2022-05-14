import os.path as op
import torch
import logging

from src.qd.mask.data import samplers
from src.qd.mask.utils.comm import get_world_size
from src.qd.mask.utils.build import make_data_sampler
from src.qd.mask.data.datasets import (CaptionTSVDataset,
                                       VilTPretrainCaptionTSVDataset,)
from src.qd.mask.data.datasets.caption_tensorizer import build_tensorizer


def build_dataset(yaml_file, tokenizer, args, is_train=True, transform=None):
    if not op.isfile(yaml_file):
        yaml_file = op.join(args.data_dir, yaml_file)
        assert op.isfile(yaml_file)
    tensorizer = build_tensorizer(args, tokenizer, is_train=is_train)
    return VilTPretrainCaptionTSVDataset(
        yaml_file,
        tensorizer,
        tokenizer,
        is_train=is_train,
        mask_loss_for_unmatched=args.mask_loss_for_unmatched,
        on_memory=args.on_memory,
        qa2caption=args.qa2caption,
        transform=transform,
        pert_caption_prob=args.pert_caption_prob,
        pert_labels_prob=args.pert_labels_prob,
    )


def make_batch_data_sampler(sampler, images_per_gpu, num_iters=None, start_iter=0):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_gpu, drop_last=False
    )
    if num_iters is not None and num_iters >= 0:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(args, yaml_file, tokenizer, is_distributed=True, 
        is_train=True, start_iter=0, is_pretrain=False, transform=None):
    if is_pretrain:
        assert is_train
        dataset = build_dataset(yaml_file, tokenizer, args, is_train, transform)
    else:
        dataset = build_dataset(yaml_file, tokenizer, args,
                                is_train=(is_train and not args.scst),
                                transform=transform)
    logger = logging.getLogger(__name__)
    if is_train:
        shuffle = args.train_shuffle
        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(dataset) // images_per_batch
        num_iters = iters_per_batch * args.num_train_epochs
        logger.info("Train with {} images per GPU.".format(images_per_gpu))
        logger.info("Total batch size {}".format(images_per_batch))
        logger.info("Total training steps {}".format(num_iters))
        logging.info('shuffle = {}'.format(shuffle))
    else:
        shuffle = False
        images_per_gpu = args.per_gpu_eval_batch_size
        num_iters = None
        start_iter = 0

    sampler = make_data_sampler(
        dataset, shuffle, is_distributed, images_per_gpu)
    batch_sampler = make_batch_data_sampler(
        sampler, images_per_gpu, num_iters, start_iter
    )

    from src.data_layer.builder import collate_fn
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, batch_sampler=batch_sampler,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return data_loader


