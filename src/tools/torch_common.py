from pprint import pformat
import torch
from src.tools.common import get_mpi_rank, get_mpi_size
from src.tools.common import is_hvd_initialized
import torch.distributed as dist
import os
import os.path as op
from torch import nn
import logging
from src.tools.common import get_mpi_local_rank
from src.tools.common import ensure_directory


def init_random_seed(random_seed):
    import random
    random.seed(random_seed)
    import numpy as np
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def get_torch_version_info():
    return {
        'version': torch.__version__,
        'cuda': torch.version.cuda,
        'nccl': torch.cuda.nccl.version(),
        'cudnn': torch.backends.cudnn.version(),
        'device_count': torch.cuda.device_count(),
        'current_device': torch.cuda.current_device(),
    }


def attach_module_name_(model):
    for n, m in model.named_modules():
        m.name_from_root = n


def replace_module_by_name(module, module_part_name, creator_func):
    attach_module_name_(module)
    return replace_module(module,
                   lambda m: m.name_from_root.endswith(module_part_name),
                   creator_func)


def replace_module(module, condition_func, creator_func):
    module_output = module
    if condition_func(module):
        module_output = creator_func(module)
    for name, child in module.named_children():
        child = replace_module(child, condition_func, creator_func)
        module_output.add_module(name, child)
    del module
    return module_output


def torch_save(t, f, **kwargs):
    ensure_directory(op.dirname(f))
    tmp_f = f + '.tmp'
    torch.save(t, tmp_f, **kwargs)
    os.rename(tmp_f, f)


def torch_load(filename):
    from src.tools.common import get_user_name
    user_name = get_user_name()
    from src.tools.common import acquireLock, releaseLock
    from src.tools.common import hash_sha1
    lock_fd = acquireLock(op.join('/tmp',
        '{}_lock_{}'.format(user_name, hash_sha1(filename))))

    result = torch.load(filename, map_location=lambda storage, loc: storage)
    releaseLock(lock_fd)
    return result


def recursive_to_device(d, device, **kwargs):
    if isinstance(d, tuple) or isinstance(d, list):
        return [recursive_to_device(x, device, **kwargs) for x in d]
    elif isinstance(d, dict):
        return dict((k, recursive_to_device(v, device)) for k, v in d.items())
    elif isinstance(d, torch.Tensor) or hasattr(d, 'to'):
        #return d.to(device, non_blocking=True)
        return d.to(device, **kwargs)
    else:
        return d


# use recrusive_to_device, this naming is too short
def to(d, device):
    return recursive_to_device(d, device)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if get_mpi_size() == 1:
        return tensor
    if not is_hvd_initialized():
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(get_mpi_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output
    else:
        import horovod as hvd
        output = hvd.torch.allgather(tensor)
        return output


def get_master_node_ip():

    if 'MASTER_IP' in os.environ:
        return os.environ['MASTER_IP']
    else:
        # in local machine, sometimes, we do not have that file, and here
        # we resort to localhost
        return 'localhost'


def ensure_init_process_group(device_id=None, port=12345):
    if not dist.is_initialized():
        dist_url = 'tcp://{}:{}'.format(get_master_node_ip(),
                port)
        from datetime import timedelta
        init_param = {
            'backend': 'nccl',
            'init_method': dist_url,
            'rank': get_mpi_rank(),
            'world_size': get_mpi_size(),
            # 'world_size': torch.distributed.get_world_size(),
            'timeout': timedelta(days=10),
        }
        if device_id is None:
            device_id = get_mpi_local_rank()
        torch.cuda.set_device(device_id)
        logging.info(init_param)
        dist.init_process_group(**init_param)


def calc_num_node_in_grad_fn(grad_fn):
    result = 0
    if grad_fn is not None:
        result += 1
        if hasattr(grad_fn, 'next_functions'):
            for f in grad_fn.next_functions:
                result += calc_num_node_in_grad_fn(f)
    return result


def evaluate_topk(iter_pred_tsv, iter_label_tsv):
    import json
    correct = 0
    total = 0
    for (key, str_rects), (key_pred, str_pred) in zip(iter_label_tsv, iter_pred_tsv):
        total = total + 1
        assert key == key_pred
        curr_predict = json.loads(str_pred)
        if len(curr_predict) == 0:
            continue
        curr_gt_rects = json.loads(str_rects)
        if type(curr_gt_rects) is int:
            # imagenet data
            curr_gt_rects = [{'class': str(curr_gt_rects)}]
        curr_pred_best = curr_predict[max(range(len(curr_predict)), key=lambda i: curr_predict[i]['conf'])]['class']
        if any(g['class'] == str(curr_pred_best) for g in curr_gt_rects):
            correct = correct + 1
    return 1. * correct / total


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    use_hvd = is_hvd_initialized()
    if not use_hvd:
        if not dist.is_available():
            return
        if not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        if world_size == 1:
            return
        dist.barrier()
    else:
        from src.tools.common import get_mpi_size
        if get_mpi_size() > 1:
            import horovod.torch as hvd
            hvd.allreduce(torch.tensor(0), name='barrier')


def all_gather_grad_curr(x):
    if get_mpi_size() == 1:
        return x
    else:
        with torch.no_grad():
            all_x = [torch.zeros_like(x) for _ in range(get_mpi_size())]
            # note, all_rep should be treated as constent, which means no grad
            # will be propagated back through all_rep
            torch.distributed.all_gather(all_x, x)
        all_x[get_mpi_rank()] = x
        return torch.cat(all_x, dim=0)


def all_gather_curr_first_grad(x):
    if get_mpi_size() == 1:
        return x
    else:
        with torch.no_grad():
            all_x = [torch.zeros_like(x) for _ in range(get_mpi_size())]
            # note, all_rep should be treated as constent, which means no grad
            # will be propagated back through all_rep
            torch.distributed.all_gather(all_x, x)
        rank = get_mpi_rank()
        if rank == 0:
            all_x[0] = x
        else:
            all_x[rank] = all_x[0]
            all_x[0] = x
        return torch.cat(all_x, dim=0)


def sum_reduce_(x):
    if get_mpi_size() > 1:
        torch.distributed.all_reduce(x)


def sum_single_reduce_(x, dst):
    if get_mpi_size() > 1:
        torch.distributed.reduce(x, dst)


def max_reduce_(x):
    if get_mpi_size() > 1:
        torch.distributed.all_reduce(x, torch.distributed.ReduceOp.MAX)



def set_seed(seed, n_gpu):
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


class InputAsDict(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model

    def forward(self, data_dict):
        if isinstance(data_dict, torch.Tensor):
            im = data_dict
        else:
            im = data_dict['image']
        return self.module(im)


def load_model_state_ignore_mismatch(model, init_dict):
    real_init_dict = {}
    name_to_param = dict(model.named_parameters())
    name_to_param.update(dict(model.named_buffers()))

    def same_shape(a, b):
        return len(a.shape) == len(b.shape) and \
                all(x == y for x, y in zip(a.shape, b.shape))

    num_ignored = 0
    unique_key_in_init_dict = []
    keys_shape_mismatch = []
    for k in init_dict:
        if k in name_to_param:
            if same_shape(init_dict[k], name_to_param[k]):
                # a = model.state_dict()[k]
                # b = init_dict[k]
                real_init_dict[k] = init_dict[k]
            else:
                logging.info('{} shape is not consistent, expected: {}; got '
                             '{}'.format(k, name_to_param[k].shape, init_dict[k].shape))
                keys_shape_mismatch.append(k)
        else:
            unique_key_in_init_dict.append(k)
            num_ignored = num_ignored + 1

    logging.info('unique keys in init dict = {}; total = {}'.format(
        pformat(unique_key_in_init_dict), len(unique_key_in_init_dict),
    ))

    result = model.load_state_dict(real_init_dict, strict=False)
    logging.info('unique key (not initialized) in current model = {}'.format(
        pformat(result.missing_keys),
    ))


def remove_prefix(model, prefix):
    out = {}
    for k, v in model.items():
        while k.startswith(prefix):
            k = k[len(prefix): ]
        out[k] = v
    return out


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


