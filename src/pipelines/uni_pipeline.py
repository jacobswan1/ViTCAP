from src.tools.torch_common import evaluate_topk
from src.tools.tsv.tsv_io import reorder_tsv_keys
import src.data_layer.samplers as samplers
from src.tools.torch_common import recursive_to_device
from src.tools.common import save_parameters
from src.tools.opt.trainer import do_train_dict
from tqdm import tqdm
from src.tools.common import write_to_yaml_file
from src.tools.common import load_from_yaml_file
from src.tools.common import worth_create
from src.tools.common import read_to_buffer
from src.tools.common import write_to_file
from src.tools.common import plot_to_file
from src.tools.common import ensure_remove_dir
from src.tools.tsv.tsv_io import tsv_reader
from src.tools.tsv.tsv_io import TSVDataset
from shutil import copyfile
import os
import copy
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms
try:
    from itertools import izip as zip
except:
    # python 3
    pass

import math
import PIL
from src.tools.torch_common import ensure_init_process_group
from src.tools.torch_common import init_random_seed
from src.tools.torch_common import attach_module_name_
from src.data_layer.transform import ImageTransform2Dict
from src.tools.torch_common import InputAsDict
import sys
from datetime import datetime
from pprint import pformat
from src.tools.common import ensure_directory
from src.data_layer.dataset import DatasetPlusTransform
import json
from src.tools.common import get_mpi_size
from src.tools.common import get_mpi_local_rank, get_mpi_local_size
import logging
import os.path as op
import torch
from torch import nn
# from src.qd.layers.loss import ModelLoss
from src.tools.common import get_mpi_rank
from src.tools.tsv.tsv_io import tsv_writer
from src.tools.common import synchronize
import time
from src.tools.common import json_dump
from src.tools.common import qd_tqdm as tqdm
from src.tools.opt.checkpoint import Checkpointer
from src.tools.common import get_all_path, dict_get_path_value, dict_update_path_value
from src.pytorch_image_models import timm


class Config(object):
    def __init__(self, default, overwrite):
        self.default = default
        self.overwrite = overwrite

    def get(self, k):
        from src.tools.common import dict_has_path, dict_get_path_value
        if dict_has_path(self.overwrite, k):
            return dict_get_path_value(self.overwrite, k)
        if dict_has_path(self.default, k):
            return dict_get_path_value(self.default, k)

    def __getattr__(self, k):
        return self.get(k)

    def get_dict(self):
        import copy
        default = copy.deepcopy(self.default)
        for p in get_all_path(self.overwrite):
            v = dict_get_path_value(self.overwrite, p)
            dict_update_path_value(default, p, v)
        return default


def get_model_sub_name(i):
    return 'model_iter_{:07d}'.format(i)


class UniPipeline(object):
    def __init__(self, **kwargs):
        self._default = {
            'snapshot_steps': 5000,
            'find_unused_parameters': True,
            'test_batch_size': 1,
            'effective_batch_size': 8,
            'find_unused_parameters': True,
            'data': 'Unknown',
            'net': 'Unknown',
            'expid': 'Unknown',
            'dist_backend': 'nccl',
            'init_method_type': 'tcp',
            'log_step': 100,
            'evaluate_method': 'map',
            'test_split': 'test',
            'num_workers': 8,
            'ovthresh': [-1],
            'step_lr': 30,
            'base_lr': 0.1,
            'max_iter': 10,
            # the default value was 5e-4, which is the default for yolo. We
            # add the default as 5e-4 in yolo_by_mask, and set it 1e-4 for
            # classification.
            'random_seed': 88,
            'apply_nms_gt': True,
            'cudnn_benchmark': False,
            'test_mergebn': False,
            'bgr2rgb': False, # this should be True, but set it False for back compatibility
            'coco_eval_max_det': 100,
            # init
            'dist_url_tcp_port': 12345,
            # data layer
            'train_crop_size': 224,
            'test_crop_size': 224,
            'train_shuffle': True,
            # optimizer
            'momentum': 0.9,
            'weight_decay': 1e-4,
            # lr scheduler
            'scheduler_type': 'cosine',
            'min_rel_lr_in_cosine': 0.,
            'cosine_warmup_factor': 1. / 3,
            'cosine_restart_after_warmup': True,
            'train_transform': 'inception',
            'cosine_warmup_iters': 500,
            'warmup_steps': 0,
            'rms_alpha': 0.99,
            'smooth_label_eps': 0.1,
            'pred_tsv_to_json_extra': 1,
            'mobilenetv3_dropout_ratio': 0.2,
            'cutout_factor': 4,
            'dist_weight': 1.,
            'max_gen_length': 20,
            'splitbysplitsample_buffer_size': 1,
            'splitbysplitsample_group_size': 1,
            'device': 'cuda',
        }
        self.cfg = Config(self._default, kwargs)

        # output folder
        self.full_expid = self.cfg.full_expid or '_'.join(
            map(str, [self.cfg.data, self.cfg.net, self.cfg.expid]))
        self.output_folder = op.join('output', self.full_expid)
        self.model_folder = op.join(self.output_folder, 'snapshot')
        ensure_directory(self.model_folder)

        self.mpi_rank = get_mpi_rank()
        self.mpi_size = get_mpi_size()
        self.mpi_local_rank = get_mpi_local_rank()
        self.mpi_local_size = get_mpi_local_size()

        self.device_id = (self.mpi_local_rank if not self.cfg.debug_train else 0)

        # data related
        self.train_collate_fn = None
        self.test_collate_fn = None

        # adapt the batch size based on the mpi_size
        self.is_master = self.mpi_rank == 0

        self.max_iter = self.parse_iter(self.cfg.max_iter)
        # self.max_iter = 0

        self.initialized = False

    def get_len_dataset(self, is_train):
        raise NotImplementedError('defined in sub class')

    def get_transform(self, is_train):
        raise NotImplementedError('defined in sub classes')

    def get_raw_model(self, is_train):
        raise NotImplementedError('sub class to implement')

    def predict_output_to_tsv_row(self, data, output):
        raise NotImplementedError('sub class to implement')

    def append_predict_param(self, cc):
        if self.cfg.test_normalize_module:
            cc.append('NormBy{}'.format(self.cfg.test_normalize_module))
        if self.cfg.predict_extract:
            s = self.predict_extract
            if isinstance(self.predict_extract, list):
                s = '.'.join(self.predict_extract)
            cc.append('Extract{}'.format(s))
        if self.cfg.test_crop_position:
            cc.append(self.test_crop_position)
        if self.cfg.test_resize_size and self.cfg.test_resize_size != 224:
            cc.append('r{}'.format(self.cfg.test_resize_size))
        if self.cfg.predict_ema_decay:
            cc.append('ema{}'.format(self.predict_ema_decay))
        if self.cfg.test_max_iter is not None:
            # this is used for speed test
            if self.test_mergebn:
                cc.append('mergebn')
            cc.append('max_iter{}'.format(self.test_max_iter))
            # we explicitly log the batch size here so that we can make sure it
            # is 1 or batch processing
            cc.append('BS{}'.format(self.test_batch_size))
            cc.append(self.device)
            if self.device == 'cpu' and self.cpu_num_threads:
                torch.set_num_threads(self.cpu_num_threads)
                cc.append('thread{}'.format(self.cpu_num_threads))
        if self.cfg.flush_denormal and self.device == 'cpu':
            # gpu is not supported
            r = torch.set_flush_denormal(True)
            assert r, 'not supported'
            cc.append('flush_denormal')
        if self.cfg.pred_file_hint is not None:
            cc.append(self.pred_file_hint)
        if self.cfg.test_crop_size != 224 and self.cfg.test_crop_size:
            cc.append('crop{}'.format(self.cfg.test_crop_size))

        # in vision-laugnage
        if self.cfg.max_gen_length != 20:
            cc.append('max_token{}'.format(self.cfg.max_gen_length))

        if self.cfg.test_respect_ratio_max is not None:
            cc.append('testMax{}'.format(self.cfg.test_respect_ratio_max))

    def get_model(self, is_train):
        model = self.get_raw_model(is_train)
        model = self.model_surgery(model, is_train)
        return model

    def model_surgery(self, model, is_train):
        # assign a name to each module so that we can use it in each module to
        # print debug information
        attach_module_name_(model)
        if is_train:
            if self.cfg.device == 'cuda':
                if self.cfg.trainer != 'pl':
                    model = self.data_parallel_wrap(model)
                else:
                    assert self.cfg.trainer is None
                    model = model.cuda()
        else:
            model.eval()

        return model

    def parse_iter(self, i):
        def to_iter(e):
            if type(e) is str and e.endswith('e'):
                num_train_images = len(self.get_len_dataset(is_train=True))
                iter_each_epoch = 1. * num_train_images / self.cfg.effective_batch_size
                return int(float(e[:-1]) * iter_each_epoch)
            else:
                return int(e)
        return to_iter(i)

    def get_dataset(self, is_train):
        len_dataset = self.get_len_dataset(is_train)
        trans = self.get_transform(is_train)
        dataset = DatasetPlusTransform(len_dataset, trans)
        return dataset

    def get_sampler(self, is_train, dataset):
        # elif stage == 'train' and self.composite_rank_aware_sampler:
        if is_train:
            length_divisible = self.cfg.effective_batch_size // self.mpi_size
        else:
            length_divisible = 1

        if is_train and self.cfg.sampler_type == 'splitBysplit':
            sampler = samplers.SplitBySplitSampler(
                dataset,
                shuffle=self.cfg.train_shuffle,
                random_seed=self.cfg.random_seed,
                group_size=self.cfg.splitbysplitsample_group_size,
                prepare_t=self.get_splitbysplit_sampler_prepare_t(),
                prepare_version=self.get_splitbysplit_sampler_prepare_version(),
                prepare_buffer=self.cfg.splitbysplitsample_buffer_size,
            )
        elif is_train and self.cfg.sampler_type == 'ranksplit':
            from src.data_layer.samplers import RankSplitSampler
            sampler = RankSplitSampler(dataset, shuffle=self.cfg.train_shuffle,
                                       random_seed=self.cfg.random_seed)
        elif is_train and self.cfg.sampler_type == 'nodesplit':
            from src.data_layer.samplers import NodeSplitSampler
            sampler = NodeSplitSampler(dataset, shuffle=self.cfg.train_shuffle,
                                       random_seed=self.cfg.random_seed)
        else:
            sampler = samplers.DistributedSampler(
                dataset,
                shuffle=self.cfg.train_shuffle if is_train else False,
                length_divisible=length_divisible)
        return sampler

    def get_splitbysplit_sampler_prepare_t(self):
        return None

    def get_splitbysplit_sampler_prepare_version(self):
        return None

    def get_batch_sampler(self, is_train, sampler, start_iter):
        bs = (self.cfg.effective_batch_size // self.mpi_size if is_train else
            self.cfg.test_batch_size)
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler,
            bs,
            drop_last=False,
        )
        if is_train:
            batch_sampler = samplers.IterationBasedBatchSampler(
                batch_sampler, self.max_iter, start_iter
            )
        return batch_sampler

    def get_data_loader(self, is_train, start_iter):
        dataset = self.get_dataset(is_train)
        sampler = self.get_sampler(is_train, dataset)
        logging.info('sampler = {}'.format(sampler))
        batch_sampler = self.get_batch_sampler(is_train, sampler, start_iter)
        collate_fn = None

        if is_train:
            collate_fn = self.train_collate_fn
        else:
            collate_fn = self.test_collate_fn

        loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
        )
        return loader

    def ensure_train(self):
        self._ensure_initialized(
            init_ddp=(self.cfg.trainer not in ['pl', 'ds']),
            init_ds=(self.cfg.trainer == 'ds')
        )

        last_model_file = self.get_checkpoint_file()
        logging.info('last model file = {}'.format(last_model_file))
        if op.isfile(last_model_file) and not self.cfg.force_train:
            logging.info('skip to train')
            return

        if self.mpi_rank == 0:
            save_parameters(self.cfg.overwrite, self.output_folder)

        logging.info(pformat(self.cfg.get_dict()))
        from src.tools.torch_common import get_torch_version_info
        logging.info('torch info = {}'.format(
            pformat(get_torch_version_info())))

        synchronize()

        self._setup_logging()
        train_result = self.train()

        if self.mpi_rank == 0 and not self.cfg.debug_train:
            # save the code after training
            from src.tools.common import zip_qd, try_delete
            # we'd better to delete it since it seems like zip will read/write if there is
            source_code = op.join(self.output_folder, 'source_code.zip')
            if op.isfile(source_code):
                try_delete(source_code)
            zip_qd(op.join(self.output_folder, 'source_code'))

        synchronize()

        return train_result

    def _setup_logging(self):
        # all ranker outputs the log to a file
        # only rank 0 print the log to console
        log_file = op.join(self.output_folder,
            'log_{}_rank{}.txt'.format(
                datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),
                self.mpi_rank))
        ensure_directory(op.dirname(log_file))
        file_handle = logging.FileHandler(log_file)
        logger_fmt = logging.Formatter('%(asctime)s.%(msecs)03d %(process)d:%(filename)s:%(lineno)s %(funcName)10s(): %(message)s')
        file_handle.setFormatter(fmt=logger_fmt)

        root = logging.getLogger()
        root.handlers = []
        root.setLevel(logging.INFO)
        root.addHandler(file_handle)

        if self.mpi_rank == 0:
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(logger_fmt)
            root.addHandler(ch)

    def get_optimizer(self, model):
        parameters = get_parameter_groups(self, model)

        if self.cfg.optimizer_type in [None, 'SGD', 'LARS']:
            from src.tools.opt.sgd import SGDVerbose
            optimizer = SGDVerbose(parameters,
                                   self.cfg.base_lr,
                                   momentum=self.cfg.momentum,
                                   # this is default decay, and will be
                                   # overwritten if we specified it in
                                   # parameters.
                                   weight_decay=self.cfg.weight_decay,
                                   nesterov=self.cfg.sgd_nesterov,
                                   )
        elif self.cfg.optimizer_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(
                parameters,
                self.cfg.base_lr,
                momentum=self.cfg.momentum,
                alpha=self.cfg.rms_alpha,
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.optimizer_type in ['Adam']:
            optimizer = torch.optim.Adam(
                parameters,
                self.cfg.base_lr,
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.optimizer_type in ['AdamW']:
            optimizer = torch.optim.AdamW(
                parameters,
                self.cfg.base_lr,
                weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer_type in ['MAdamW']:
            from src.solver import AdamW
            print('learning rate {}, {}'.format(self.cfg.base_lr, type(self.cfg.base_lr)))
            optimizer = AdamW(parameters,
                              lr=self.cfg.base_lr,
                              eps=1e-8)
        else:
            raise NotImplementedError(self.cfg.optimizer_type)
        # if self.cfg.optimizer_type in ['LARS']:
        #     from torchlars import LARS
        #     optimizer = LARS(optimizer=optimizer)
        if self.cfg.ema_optimizer:
            from src.tools.opt.ema_optimizer import EMAOptimizer
            optimizer = EMAOptimizer(optimizer=optimizer)
        return optimizer

    def get_lr_scheduler(self, optimizer):
        scheduler_type = self.cfg.scheduler_type
        if scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.parse_iter(self.cfg.step_lr),
            )
        elif scheduler_type == 'multi_step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[self.parse_iter(i) for i in self.cfg.stageiter],
                gamma=0.1,
            )
        elif scheduler_type == 'cosine':
            from src.tools.opt.WarmupCosineAnnealingLR import WarmupCosineAnnealingLR
            assert isinstance(self.max_iter, int)
            scheduler = WarmupCosineAnnealingLR(
                optimizer,
                max_iter=self.max_iter,
                min_lr=self.cfg.min_rel_lr_in_cosine * self.cfg.base_lr,
                warmup_factor=self.cfg.cosine_warmup_factor,
                warmup_iters=self.parse_iter(self.cfg.cosine_warmup_iters),
                cosine_restart_after_warmup=self.cfg.cosine_restart_after_warmup
            )
        elif scheduler_type == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.max_iter,
            )
        elif scheduler_type == 'ReduceLROnPlateau':
            assert isinstance(self.max_iter, int)
            patience = 3 * self.max_iter // self.effective_batch_size
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=patience, verbose=True)
        elif scheduler_type == "linear":
            from src.solver import WarmupLinearSchedule
            scheduler = WarmupLinearSchedule(
                optimizer,
                warmup_steps=self.parse_iter(self.cfg.warmup_steps),
                t_total=self.max_iter,
            )
        else:
            raise NotImplementedError(scheduler_type)
        return scheduler

    def data_parallel_wrap(self, model):
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.device_id],
            # used for effiicient-net + faster-rcnn
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def create_checkpointer(self, model, optimizer, scheduler):
        save_to_disk = get_mpi_rank() == 0
        checkpointer = Checkpointer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            save_dir=op.join(self.output_folder, 'snapshot'),
            save_to_disk=save_to_disk,
            suffix='pt',
        )
        return checkpointer

    def lightning_train(self):
        pass

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

        start_iter = extra_param.get('iteration', 0)

        logging.info(scheduler)

        # use the maskrcnn trainer engine
        train_loader = self.get_data_loader(
            is_train=True,
            start_iter=start_iter,
        )

        self.do_train(train_loader, model, optimizer, scheduler, checkpointer, start_iter)
        return checkpointer.get_checkpoint_file()

    def do_train(self, loader, model, optimizer, scheduler, checkpointer,
                 start_iter):
        device = torch.device(self.cfg.device)
        logging.info(model)
        for n, m in model.named_modules():
            logging.info('{}: training={}'.format(n, m.training))
        logging.info('dataset = \n{}'.format(loader.dataset))

        if self.cfg.trainer == 'pl':
            raise NotImplementedError('not work')
            from src.qd.layers.lightning_wrapper import LightningModule
            model = LightningModule(model, optimizer, scheduler)
            args = {}

            args['max_epochs'] = 1
            if self.cfg.use_amp:
                args['precision'] = 16
            if self.cfg.gradient_clip is not None:
                args['gradient_clip_val'] = self.cfg.gradient_clip
            args['replace_sampler_ddp'] = False
            args['accelerator'] = 'horovod'

            trainer = pl.Trainer(**args)

            trainer.fit(model, loader)

            # save the result
            checkpointer.save(get_model_sub_name(self.max_iter))

        elif self.cfg.trainer == 'ds':
            from src.tools.opt.trainer import do_train_by_deepspeed
            do_train_by_deepspeed(
                model=model,
                data_loader=loader,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpointer=checkpointer,
                device=device,
                checkpoint_period=self.cfg.snapshot_steps,
                arguments={'iteration': start_iter},
                log_step=self.cfg.log_step,
                use_amp=self.cfg.use_amp,
                gradient_clip=self.cfg.gradient_clip,
                model_sub_name_fn=get_model_sub_name
            )
        else:
            do_train_dict(
                model=model,
                data_loader=loader,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpointer=checkpointer,
                device=device,
                checkpoint_period=self.cfg.snapshot_steps,
                arguments={'iteration': start_iter},
                log_step=self.cfg.log_step,
                use_amp=self.cfg.use_amp,
                gradient_clip=self.cfg.gradient_clip,
                model_sub_name_fn=get_model_sub_name
            )

    def get_checkpoint_file(self, iteration=None):
        if iteration is None and self.cfg.model_file is not None:
            return self.cfg.model_file
        if iteration is None:
            iteration = self.max_iter
        iteration = self.parse_iter(iteration)
        return op.join(
            self.model_folder,
            get_model_sub_name(iteration) + '.pt')

    def init_ddp(self):
        ensure_init_process_group(
            device_id=self.device_id,
            port=self.cfg.dist_url_tcp_port,
        )

    def init_ds(self):
        import deepspeed
        deepspeed.init_distributed()

    def _ensure_initialized(self, init_ddp=True, init_ds=False):
        if self.initialized:
            return

        if self.cfg.file_system_sharing:
            logging.info('using file system for tensor sharing')
            torch.multiprocessing.set_sharing_strategy('file_system')

        if self.cfg.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        torch.cuda.set_device(self.mpi_local_rank)

        if init_ddp:
            self.init_ddp()

        if init_ds:
            self.init_ds()

        # sometimes, the init hangs, and thus we print some logs for
        # verification
        logging.info('initialized')
        # we need to synchronise before exit here so that all workers can
        # finish init_process_group(). If not, worker A might exit the
        # whole program first, but worker B still needs to talk with A. In
        # that case, worker B will never return and will hang there
        synchronize()
        init_random_seed(self.cfg.random_seed)
        self.initialized = True

    def get_predict_file(self, model_file=None):
        if model_file is None:
            model_file = self.get_checkpoint_file(iteration=self.max_iter)
        cc = [model_file, self.cfg.test_data, self.cfg.test_split]
        self.append_predict_param(cc)
        cc.append('predict')
        cc.append('tsv')
        return '.'.join(cc)

    def ensure_predict(self, model_file=None):
        if self.cfg.ignore_predict:
            logging.info('ignore to predict as instructed')
            return

        self._ensure_initialized()

        if model_file is None:
            model_file = self.get_checkpoint_file()
            assert model_file is not None
        predict_result_file = self.get_predict_file(model_file)

        # load the pre-trained checkpoint. this assume we must load something for evaluation.
        if not op.isfile(model_file) and not op.isdir(model_file):
            logging.info('ignore to run predict since {} does not exist'.format(
                model_file))
            return predict_result_file

        if not worth_create(model_file, predict_result_file) and not self.cfg.force_predict:
            logging.info('ignore to do prediction {}'.format(predict_result_file))
            return predict_result_file

        self.predict(model_file, predict_result_file)

        return predict_result_file

    def load_test_model(self, model, model_file):
        checkpointer = Checkpointer(
            model=model,
            save_dir=self.output_folder,
        )
        checkpointer.load(model_file, load_if_has=False)

    def get_rank_specific_tsv(self, f, rank):
        return '{}_{}_{}.tsv'.format(f, rank, self.mpi_size)

    def predict_iter(self, dataloader, model, meters):
        start = time.time()
        logging.info(dataloader.dataset)
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            if self.cfg.test_max_iter is not None and i >= self.cfg.test_max_iter:
                # this is used for speed test, where we only would like to run a
                # few images
                break
            meters.update(data=time.time() - start)
            start = time.time()
            data = recursive_to_device(data, self.cfg.device)
            meters.update(input_to_cuda=time.time() - start)
            start = time.time()
            output = self.predict_iter_forward(model, data)
            meters.update(model=time.time() - start)
            start = time.time()
            for row in self.predict_output_to_tsv_row(data, output):
                yield row
            if self.cfg.debug_feature:
                model.sumarize_feature()
            meters.update(write=time.time() - start)
            start = time.time()

    def predict_iter_forward(self, model, inputs):

        # FIXME: CHECK here!! Sometimes the training fails to load the initialization
        # check if checkpoint param is changed
        # init_param = torch.load('./output/TaxCocoCaption_B_Vilt_tagger/snapshot/model_iter_0050000.pt')['model']
        # now_param = model.state_dict()
        # keys = now_param.keys()
        # for key in keys:
        #     if key in init_param:
        #         print(key, (init_param[key] - now_param[key]).sum())
        #     else:
        #         print(key, 'not initialized.')

        with torch.no_grad():
            return model(inputs)

    def feature_to_tsv_row(self, features, feature_names, keys):
        if isinstance(feature_names, str):
            feature_names = [feature_names]
        if isinstance(keys, dict):
            keys = keys['key']
        for i, key in enumerate(keys):
            info = []
            for f, f_name in zip(features, feature_names):
                info.append({'feature': f[i].tolist(), 'name': f_name})
            yield key, json_dump(info)

    def post_load_model_surgery(self, model, model_file):
        if self.cfg.test_mergebn:
            from src.qd.layers import MergeBatchNorm
            model = MergeBatchNorm(model)
            logging.info('after merging bn = {}'.format(model))
        from src.layers import ForwardPassTimeChecker
        model = ForwardPassTimeChecker(model)
        model = model.to(self.cfg.device)
        model.eval()

        return model

    def is_train_finished(self):
        last_model = self.get_checkpoint_file()
        if not op.isfile(last_model) and \
                not op.islink(last_model) and \
                not op.isdir(last_model):
            logging.info('{} is not a file and not a folder'.format(
                last_model
            ))
            return False
        return True

    def predict(self, model_file, predict_result_file):
        if self.mpi_size > 1:
            sub_predict_file = self.get_rank_specific_tsv(predict_result_file,
                    self.mpi_rank)
        else:
            sub_predict_file = predict_result_file

        model = self.get_model(is_train=False)

        # FIXME: do not load checkpoint.
        self.load_test_model(model, model_file)

        model = self.post_load_model_surgery(model, model_file)
        dataloader = self.get_data_loader(is_train=False, start_iter=0)

        # from maskrcnn_benchmark.utils.metric_logger import MetricLogger
        from src.tools.logger import MetricLogger
        meters = MetricLogger(delimiter="  ")
        logging.info('writing {}'.format(sub_predict_file))
        tsv_writer(self.predict_iter(dataloader, model, meters),
                   sub_predict_file)

        speed_yaml = sub_predict_file + '.speed.yaml'
        write_to_yaml_file(model.get_time_info(), speed_yaml)

        logging.info(str(meters))

        if self.mpi_rank == 0:
            info_file = predict_result_file + '.info.yaml'
            write_to_yaml_file(self.cfg.overwrite, info_file)

        # we need to sync before merging all to make sure each rank finish its
        # own task
        synchronize()
        if self.mpi_size > 1 and get_mpi_rank() == 0:
            cache_files = [self.get_rank_specific_tsv(predict_result_file, i)
                for i in range(self.mpi_size)]
            before_reorder = predict_result_file + '.before.reorder.tsv'
            from src.tools.tsv.tsv_io import concat_tsv_files, delete_tsv_files
            concat_tsv_files(cache_files, before_reorder)
            # in distributed testing, some images might be predicted by
            # more than one worker since the distributed sampler only
            # garrantee each image will be processed at least once, not
            # exactly once. Thus, we have to remove the duplicate
            # predictions.
            ordered_keys = dataloader.dataset.get_keys()
            reorder_tsv_keys(before_reorder, ordered_keys, predict_result_file)

            delete_tsv_files(cache_files)
            delete_tsv_files([before_reorder])

            # during prediction, we also computed the time cost. Here we
            # merge the time cost
            speed_cache_files = [c + '.speed.yaml' for c in cache_files]
            speed_yaml = predict_result_file + '.speed.yaml'
            from src.tools.common import merge_speed_info
            merge_speed_info(speed_cache_files, speed_yaml)
            from src.tools.common import try_delete
            for x in speed_cache_files:
                try_delete(x)
            vis_files = [op.splitext(c)[0] + '.vis.txt' for c in speed_cache_files]
            from src.tools.common import merge_speed_vis
            merge_speed_vis(vis_files,
                    op.splitext(speed_yaml)[0] + '.vis.txt')
            for x in vis_files:
                try_delete(x)

        synchronize()
        return predict_result_file

    def get_evaluate_file(self, predict_file=None):
        if predict_file is None:
            predict_file = self.get_predict_file()
        assert predict_file.endswith('.tsv')
        cc = [op.splitext(predict_file)[0]]
        if self.cfg.evaluate_method != 'map':
            if self.cfg.evaluate_method is None:
                return
            cc.append(self.cfg.evaluate_method)
        if self.cfg.evaluate_method == 'neg_aware_gmap':
            if not self.apply_nms_gt:
                cc.append('noNMSGt')
            if not self.apply_nms_det:
                cc.append('noNMSDet')
            if not self.expand_label_det:
                cc.append('noExpandDet')
        if self.cfg.test_version:
            if self.cfg.test_version == -1:
                latest_version = TSVDataset(self.test_data).get_latest_version(
                        self.test_split, 'label')
                self.test_version = latest_version
                logging.info('inferred the latest version is {}'.format(
                    latest_version))
            cc.append('v{}'.format(self.cfg.test_version))
        if self.cfg.coco_eval_max_det is not None and self.cfg.coco_eval_max_det != 100:
            cc.append('MaxDet{}'.format(self.coco_eval_max_det))
        if self.cfg.pred_tsv_to_json_extra != 1 and \
                self.cfg.evaluate_method == 'coco_box':
            cc.append('{}'.format(self.pred_tsv_to_json_extra))
        cc.append('report')
        return '.'.join(cc)

    def ensure_evaluate(self, predict_file=None):
        if self.mpi_rank != 0:
            logging.info('skip because the rank {} != 0'.format(self.mpi_rank))
            return

        # if prediction is disabled, we will not proceed either.
        if self.cfg.ignore_evaluate or self.cfg.ignore_predict:
            logging.info('ignore evaluate as instructed')
            return

        # not other rank will exit and initalizing distributed will not go
        # through. No need to run initilaization here actually.
        # self._ensure_initialized()
        if not predict_file:
            model_file = self.get_checkpoint_file()
            predict_file = self.get_predict_file(model_file)
        evaluate_file = self.get_evaluate_file(predict_file)
        if evaluate_file is None:
            return
        if not worth_create(predict_file, evaluate_file) and not self.cfg.force_evaluate:
            logging.info('ignore {}'.format(evaluate_file))
        else:
            self.evaluate(predict_file, evaluate_file)

        # create index
        # Zhiyuan: Comment it as not using this.
        # self.ensure_create_evaluate_meta_file(evaluate_file)
        return evaluate_file

    def evaluate(self, predict_file, evaluate_file):
        dataset = TSVDataset(self.cfg.test_data)

        if self.cfg.evaluate_method == 'map':
            from src.qd.deteval import deteval_iter
            other_param = copy.deepcopy(self.cfg.overwrite)
            if 'ovthresh' in other_param:
                del other_param['ovthresh']
            deteval_iter(
                    dataset.iter_data(self.cfg.test_split, 'label',
                        version=self.cfg.test_version),
                    predict_file,
                    report_file=evaluate_file,
                    ovthresh=self.cfg.ovthresh, # this is in self.kwargs already
                    **other_param)
        elif self.cfg.evaluate_method == 'attr':
            # only for visualgenome
            def gen_rows():
                for key, str_rects in tsv_reader(predict_file):
                    rects = json.loads(str_rects)
                    rects2 = []
                    for r in rects:
                        for l, s in zip(r['attr_labels'], r['attr_scores']):
                            rects2.append({'rect': r['rect'], 'class': str(l), 'conf': s})
                    yield key, json_dump(rects2)
            out_tsv = op.splitext(predict_file)[0] + '.attr.tsv'
            tsv_writer(gen_rows(), out_tsv)
            from src.qd.deteval import deteval_iter
            deteval_iter(
                    dataset.iter_data('test', 'attr',
                        version=None),
                    out_tsv,
                    report_file=evaluate_file,
                    ovthresh=[0.5],
                    force_evaluate=True)
        elif self.cfg.evaluate_method == 'coco_box':
            from src.qd.cocoeval import convert_gt_to_cocoformat
            from src.qd.cocoeval import convert_to_cocoformat
            from src.qd.cocoeval import coco_eval_json
            pred_tsv_to_json_extra = self.cfg.pred_tsv_to_json_extra
            gt_json = dataset.get_data(self.cfg.test_split, 'label.cocoformat',
                    version=self.cfg.test_version) + '.json'
            gt_iter = dataset.iter_data(self.cfg.test_split, 'label',
                        version=self.cfg.test_version)

            if not op.isfile(gt_json) or self.cfg.force_evaluate:
                convert_gt_to_cocoformat(gt_iter, gt_json)
            if pred_tsv_to_json_extra == 1:
                predict_json = predict_file + '.cocoformat.json'
            else:
                assert pred_tsv_to_json_extra == 0
                predict_json = predict_file + '.cocoformat.0.json'
            is_empty = False
            if worth_create(predict_file, predict_json) or self.cfg.force_evaluate:
                annotations = convert_to_cocoformat(predict_file, predict_json,
                                                    extra=pred_tsv_to_json_extra)
                if len(annotations) == 0:
                    is_empty = True
            else:
                from src.tools.common import get_file_size
                if get_file_size(predict_json) < 100 and \
                        len(json.loads(read_to_buffer(predict_json))) == 0:
                    is_empty = True
            if is_empty:
                result = {'0.5-all': 0,
                        '0.75-all': 0,
                        'AR-all': 0,
                        'AR-all-1': 0,
                        'AR-all-10': 0,
                        'AR-large': 0,
                        'AR-medium': 0,
                        'AR-small': 0,
                        'all-all': 0,
                        'all-large': 0,
                        'all-medium': 0,
                        'all-small': 0}
            else:
                result = coco_eval_json(predict_json, gt_json,
                        maxDet=self.cfg.coco_eval_max_det)

            write_to_yaml_file(result, evaluate_file)
        elif self.cfg.evaluate_method == 'top1':
            iter_label = dataset.iter_data(self.cfg.test_split, 'label',
                    self.cfg.test_version)
            top1 = evaluate_topk(tsv_reader(predict_file), iter_label)
            logging.info('top1 = {}'.format(top1))
            write_to_yaml_file({'top1': top1}, evaluate_file)
        elif self.cfg.evaluate_method == 'neg_aware_gmap':
            from src.qd.evaluate.evaluate_openimages_google import evaluate
            truths = dataset.get_data(self.cfg.test_split, 'label')
            imagelabel_truths = dataset.get_data(self.cfg.test_split, 'imagelabel')
            assert op.isfile(truths), truths
            assert op.isfile(imagelabel_truths)
            result = evaluate(truths, imagelabel_truths, predict_file,
                    json_hierarchy_file=op.join(dataset._data_root, 'hierarchy.json'),
                    apply_nms_det=self.cfg.apply_nms_det,
                    expand_label_det=self.cfg.expand_label_det,
                    expand_label_gt=True,
                    apply_nms_gt=self.apply_nms_gt,
                    )
            from src.tools.common import convert_to_yaml_friendly
            result = convert_to_yaml_friendly(result)
            logging.info(pformat(result))
            logging.info('mAP = {}'.format(result['map']))
            write_to_yaml_file(result, evaluate_file)
        else:
            logging.info('unknown evaluate method = {}'.format(self.cfg.evaluate_method))

    def monitor_train(self):
        self._ensure_initialized()
        while True:
            need_wait_models = self.pred_eval_intermediate_models()
            all_step = self.get_all_steps()
            all_eval_file = [self.get_evaluate_file(self.get_predict_file(self.get_checkpoint_file(iteration=i)))
                for i in all_step]
            iter_to_eval = dict((i, get_acc_for_plot(eval_file))
                    for i, eval_file in zip(all_step, all_eval_file) if
                        op.isfile(eval_file))
            self.update_acc_iter(iter_to_eval)
            if need_wait_models == 0:
                break
            time.sleep(5)

        if self.mpi_rank == 0:
            self.save_to_tensorboard()
        synchronize()

    def update_acc_iter(self, iter_to_eval):
        if self.mpi_rank == 0:
            xys = list(iter_to_eval.items())
            xys = sorted(xys, key=lambda x: x[0])
            xs = [x for x, _ in xys]
            if len(xys) > 0:
                keys = xys[0][1].keys()
                for k in keys:
                    # coco accuracy
                    ys = [y[k] for _, y in xys]
                    out_file = os.path.join(
                        self.output_folder,
                        'map_{}_{}_{}.png'.format(self.cfg.test_data,
                            self.cfg.test_split, k.replace('$', '_')))
                    logging.info('create {}'.format(out_file))
                    if op.isfile(out_file):
                        os.remove(out_file)
                    plot_to_file(xs, ys, out_file)
            else:
                logging.info('nothing plotted')
        synchronize()

    def save_to_tensorboard(self):
        all_step = self.get_all_steps()
        all_eval_file = [self.get_evaluate_file(self.get_predict_file(self.get_checkpoint_file(iteration=s)))
            for s in all_step]
        all_step_eval_result = [(s, get_acc_for_plot(e)) for s, e in zip(all_step,
            all_eval_file) if op.isfile(e)]

        tensorboard_folder = op.join('output', self.full_expid, 'tensorboard_data')
        from torch.utils.tensorboard import SummaryWriter
        ensure_remove_dir(tensorboard_folder)
        wt = SummaryWriter(log_dir=tensorboard_folder)
        tag_prefix = '{}_{}'.format(self.cfg.test_data, self.cfg.test_split)
        for step, eval_result in all_step_eval_result:
            for k in eval_result:
                wt.add_scalar(tag='{}_{}'.format(tag_prefix, k),
                        scalar_value=eval_result[k],
                        global_step=step)
        wt.close()

    def pred_eval_intermediate_models(self):
        ready_predict, all_step = self.get_intermediate_model_status()
        all_ready_predict_step = [step for step, status in zip(all_step, ready_predict) if status == 1]
        for step in all_ready_predict_step:
            model_file = self.get_checkpoint_file(iteration=step)
            pred = self.ensure_predict(model_file=model_file)
            self.ensure_evaluate(pred)
            synchronize()
        not_exist_steps = [step for step, status in zip(all_step, ready_predict) if status == 0]
        logging.info('not exist steps = {}'.format(not_exist_steps))
        need_wait_models = [x for x in ready_predict if x == 0]
        return len(need_wait_models)

    def get_intermediate_model_status(self):
        ready_predict = []
        all_step = self.get_all_steps()
        for step in all_step[:-1]:
            model_file = self.get_checkpoint_file(iteration=step)
            if not op.isfile(model_file) and not op.isdir(model_file):
                ready_predict.append(0)
                continue
            predict_result_file = self.get_predict_file(model_file)
            eval_file = self.get_evaluate_file(predict_result_file)
            if not worth_create(model_file, predict_result_file) and \
                    not worth_create(predict_result_file, eval_file):
                ready_predict.append(-1)
                continue
            ready_predict.append(1)
        if self.mpi_size > 1:
            # by default, we use nccl backend, which only supports gpu. Thus,
            # we should not use cpu here.
            ready_predict = torch.tensor(ready_predict).cuda()
            dist.broadcast(ready_predict, src=0)
            ready_predict = ready_predict.tolist()
        return ready_predict, all_step[:-1]

    def get_snapshot_steps(self):
        return self.cfg.snapshot_steps

    def get_all_steps(self):
        steps = self.get_snapshot_steps()
        curr = 0
        all_step = []
        while True:
            curr += steps
            if curr >= self.max_iter:
                all_step.append(self.max_iter)
                break
            all_step.append(curr)
        return all_step


    def ensure_create_evaluate_meta_file(self, evaluate_file):
        if self.cfg.evaluate_method == 'map':
            ensure_create_evaluate_meta_file(evaluate_file)


def get_acc_for_plot(eval_file):
    if 'coco_box' in eval_file:
        return load_from_yaml_file(eval_file)
    elif 'top1' in eval_file:
        return load_from_yaml_file(eval_file)
    elif 'vqa_acc' in eval_file:
        return load_from_yaml_file(eval_file)
    elif 'caption' in eval_file:
        return load_from_yaml_file(eval_file)
    else:
        if op.isfile(eval_file + '.map.json'):
            x = json.loads(read_to_buffer(eval_file + '.map.json'))
            from src.tools.common import dict_get_all_path, dict_get_path_value
            return {p: dict_get_path_value(x, p) for p in dict_get_all_path(x)}
        return load_from_yaml_file(eval_file)


def ensure_create_evaluate_meta_file(evaluate_file):
    result = None
    simple_file = evaluate_file + '.map.json'
    if worth_create(evaluate_file, simple_file):
        if result is None:
            logging.info('data reading...')
            eval_result= read_to_buffer(evaluate_file)
            logging.info('json parsing...')
            result = json.loads(eval_result)
        s = {}
        for size_type in result:
            if size_type not in s:
                s[size_type] = {}
            for thresh in result[size_type]:
                if thresh not in s[size_type]:
                    s[size_type][thresh] = {}
                s[size_type][thresh]['map'] = \
                        result[size_type][thresh]['map']
        write_to_file(json.dumps(s, indent=4, sort_keys=True), simple_file)

    simple_file = evaluate_file + '.class_ap.json'
    if worth_create(evaluate_file, simple_file):
        if result is None:
            eval_result= read_to_buffer(evaluate_file)
            result = json.loads(eval_result)
        s = {}
        for size_type in result:
            if size_type not in s:
                s[size_type] = {}
            for thresh in result[size_type]:
                if thresh not in s[size_type]:
                    s[size_type][thresh] = {}
                s[size_type][thresh]['class_ap'] = \
                        result[size_type][thresh]['class_ap']
        write_to_file(json.dumps(s, indent=4, sort_keys=True), simple_file)

    simple_file = '{}.prec.threshold.tsv'.format(evaluate_file)
    if worth_create(evaluate_file, simple_file):
        if result is None:
            logging.info('data reading...')
            eval_result= read_to_buffer(evaluate_file)
            logging.info('json parsing...')
            result = json.loads(eval_result)
        _, max_key = max([(float(k), k) for k in result['overall']],
                key=lambda x: x[0])
        class_thresh = result['overall'][max_key]['class_thresh']
        precision_ths = None
        for l in class_thresh:
            precision_ths = class_thresh[l].keys()
            break
        if precision_ths:
            for precision_th in precision_ths:
                sub_simple_file = '{}.{}.prec{}.threshold.tsv'.format(
                        evaluate_file, max_key, precision_th)
                def gen_rows():
                    for l in class_thresh:
                        th_recall = class_thresh[l].get(precision_th, [1, 0])
                        yield l, th_recall[0], th_recall[1]
                tsv_writer(gen_rows(), sub_simple_file)
        from_file = '{}.{}.prec{}.threshold.tsv'.format(evaluate_file, max_key, 0.5)
        if op.isfile(from_file) and worth_create(from_file, simple_file):
            copyfile(from_file, simple_file)

from src.data_layer.transform import BGR2RGB


def get_transform_image(self, is_train):
    # used by cls_uni_pipeline and caption_uni_pipeline (image encoder)
    train_transform = self.cfg.train_transform
    if train_transform == 'vit':
        # TIMM style
        from src.pipelines.uni_pipeline import get_transform_vit_default
        transform = get_transform_vit_default(self, is_train=is_train)
    else:
        raise NotImplementedError(train_transform)
    return transform


def get_transform_vit_default(self, is_train):
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    if not is_train:
        trans = [
            BGR2RGB(),
            transforms.ToPILImage(),
        ]
        if self.cfg.test_respect_ratio_max:
            from src.data_layer.transform import MinMaxResizeForTest
            trans.extend([
                MinMaxResizeForTest(self.cfg.test_crop_size, self.cfg.test_respect_ratio_max)
            ])
        else:
            trans.extend([
                transforms.Resize(int(math.floor(self.cfg.test_crop_size / self.cfg.crop_pct)), PIL.Image.BICUBIC),
                transforms.CenterCrop(self.cfg.test_crop_size),
            ]),
        trans.extend([
            transforms.ToTensor(),
            normalize,
        ])
        transform = transforms.Compose(trans)
    else:
        from src.data_layer.transform import get_inception_train_transform
        transform = get_inception_train_transform(
            bgr2rgb=True,
            crop_size=self.cfg.train_crop_size,
            normalize=normalize,
            small_scale=self.cfg.input_small_scale,
        )

    return transform



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


def get_image_encoder(self, is_train, hidden_size):
    encoder_type = self.cfg.image_encoder_type
    out_dim = hidden_size
    pretrained = self.cfg.image_encoder_pretrained
    if encoder_type.startswith('resnet'):
        param = {
            'pretrained': pretrained,
            'num_classes': out_dim,
            'out_adaptive_pools': self.cfg.out_adaptive_pools,
            'out_pools': self.cfg.out_pools,
        }
        return {
            'from': 'qd.layers.resnet_vl',
            'import': encoder_type,
            'param': param,
        }
    elif encoder_type.startswith('CLIP'):
        if encoder_type == 'CLIPresnet50':
            input_resolution = (self.cfg.train_crop_size if is_train else
                                self.cfg.test_crop_size)
            return {
                'from': 'qd.layers.CLIP.model',
                'import': 'ModifiedResNet',
                'param': {
                    'layers': (3, 4, 6, 3),
                    'output_dim': 1024,
                    'heads': 32,
                    'input_resolution': input_resolution,
                    'width': 64,
                },
            }
        elif encoder_type == 'CLIPViT_B_32':
            input_resolution = (self.cfg.train_crop_size if is_train else
                                self.cfg.test_crop_size)
            return {
                'from': 'qd.layers.CLIP.model',
                'import': 'VisualTransformer',
                'param': {
                    'input_resolution': input_resolution,
                    'patch_size': 32,
                    'width': 768,
                    'layers': 12,
                    'heads': 12,
                    'output_dim': self.cfg.embed_dim or 512,
                },
            }
        else:
            raise NotImplementedError
    elif encoder_type.startswith('timm_'):
        net = encoder_type[5:]
        return {
            'from': 'qd.pipelines.clip_uni_pipeline',
            'import': 'create_timm_image_encoder',
            'param': {
                'output_dim': self.cfg.embed_dim,
                'model_name': net,
                'pretrained': pretrained,
                'output_grid': True,
            }
        }
    else:
        raise NotImplementedError


# VitEmb_vit_base_patch32_384
def get_image_encoder_model(self, is_train):
    if self.cfg.image_encoder_type.startswith('timm_'):
        net = self.cfg.image_encoder_type[5:]
        model = timm.create_model(
            net,
            output_grid=True,
            pretrained=False,
        )
        if not is_train:
            model.eval()
        from src.tools.torch_common import InputAsDict
        model = InputAsDict(model)
    elif self.cfg.image_encoder_type.startswith('VitEmb_'):
        # VitEmb_base32_384
        net = self.cfg.image_encoder_type[len('VitEmb_'):]

        if self.cfg.image_encoder_pretrained:
            logging.info('VIT image encoder loaded from pre-trained weight!  '
                         'Note that this might be replaced by pre-trained checkpoint later!')

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
    elif self.cfg.image_encoder_type.startswith('vit'):
        # prefer to use VitEmb_; as this is too flexible and it is easy to
        # make mistakes.
        parts = list(self.cfg.image_encoder_type.split('_'))[1:]
        depth, embed_dim, patch_size, num_heads = 12, 386, 16, 12
        for p in parts:
            if p.startswith('d'):
                depth = int(p[1:])
            elif p.startswith('h'):
                embed_dim = int(p[1:])
            elif p.startswith('p'):
                patch_size = int(p[1:])
            elif p.startswith('a'):
                num_heads = int(p[1:])
            else:
                raise NotImplementedError
        if depth == 0:
            # image encoder has done projection
            assert self.cfg.ignore_project_image
            assert not self.cfg.use_img_layernorm
        model_kwargs = dict(patch_size=patch_size, embed_dim=embed_dim, depth=depth,
                            num_heads=num_heads)
        img_size = self.cfg.train_crop_size if is_train else self.cfg.test_crop_size
        from timm.models.vision_transformer import VisionTransformer
        if self.cfg.image_encoder_ignore_norm:
            # use case, we ignore norm here. In joint fusion, it will be
            # passed to the ViT, which requires the input not be normed.
            model_kwargs['norm_layer'] = lambda x: nn.Identity()
        model = VisionTransformer(
            img_size=img_size, num_classes=-1, output_grid=True, **model_kwargs)
        if not is_train:
            model.eval()
        from src.tools.torch_common import InputAsDict
        model = InputAsDict(model)
    else:
        raise NotImplementedError(self.cfg.image_encoder_type)
    return model
