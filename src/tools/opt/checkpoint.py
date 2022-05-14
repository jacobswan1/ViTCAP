import logging
import os
import sys
from src.tools.common import get_mpi_rank

try:
    from torch.hub import _download_url_to_file
    from torch.hub import urlparse
    from torch.hub import HASH_REGEX
except ImportError:
    from torch.utils.model_zoo import _download_url_to_file
    from torch.utils.model_zoo import urlparse
    from torch.utils.model_zoo import HASH_REGEX

import torch


def align_and_update_state_dicts(model_state_dict, loaded_state_dict):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    target_source_name_matched = 0
    all_key_old = set()
    updated_keys = []
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        model_state_dict[key] = loaded_state_dict[key_old]
        updated_keys.append(key)
        all_key_old.add(key_old)
        target_source_name_matched += 1
        logging.info(
            log_str_template.format(
                key,
                max_size,
                key_old,
                max_size_loaded,
                tuple(loaded_state_dict[key_old].shape),
            )
        )
    logging.info('target model param = {}; name matched = {}; loaded = {}'.format(
        len(model_state_dict), target_source_name_matched,
        len(loaded_state_dict)))
    from pprint import pformat
    logging.info('from loaded; ignore = {}'.format(
        pformat([k for k in loaded_state_dict if k not in all_key_old])))
    # at the end, we remove these keys since model_state_dict will be treated
    # as loaded dict
    updated_keys = set(updated_keys)
    no_update_keys = [k for k in model_state_dict.keys() if k not in updated_keys]
    for k in no_update_keys:
        del model_state_dict[k]


def strip_prefix_if_present(state_dict, prefix):
    from src.tools.torch_common import remove_prefix
    return remove_prefix(state_dict, prefix)


def load_state_dict(model, loaded_state_dict):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict)

    # use strict loading
    # model.load_state_dict(model_state_dict)
    from src.tools.qd_pytorch import load_model_state_ignore_mismatch
    load_model_state_ignore_mismatch(model, model_state_dict)


# very similar to https://github.com/pytorch/pytorch/blob/master/torch/utils/model_zoo.py
# but with a few improvements and modifications
def cache_url(url, model_dir=None, progress=True):
    r"""Loads the Torch serialized object at the given URL.
    If the object is already present in `model_dir`, it's deserialized and
    returned. The filename part of the URL should follow the naming convention
    ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
    digits of the SHA256 hash of the contents of the file. The hash is used to
    ensure unique names and to verify the contents of the file.
    The default value of `model_dir` is ``$TORCH_HOME/models`` where
    ``$TORCH_HOME`` defaults to ``~/.torch``. The default directory can be
    overridden with the ``$TORCH_MODEL_ZOO`` environment variable.
    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        progress (bool, optional): whether or not to display a progress bar to stderr
    Example:
        >>> cached_file = maskrcnn_benchmark.utils.model_zoo.cache_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
    """
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv("TORCH_HOME", "~/.torch"))
        model_dir = os.getenv("TORCH_MODEL_ZOO", os.path.join(torch_home, "models"))
    from src.tools.common import ensure_directory
    ensure_directory(model_dir)
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if filename == "model_final.pkl":
        # workaround as pre-trained Caffe2 models from Detectron have all the same filename
        # so make the full path the filename by replacing / with _
        filename = parts.path.replace("/", "_")
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file) and get_mpi_rank() == 0:
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = HASH_REGEX.search(filename)
        if hash_prefix is not None:
            hash_prefix = hash_prefix.group(1)
            # workaround: Caffe2 models don't have a hash, but follow the R-50 convention,
            # which matches the hash PyTorch uses. So we skip the hash matching
            # if the hash_prefix is less than 6 characters
            if len(hash_prefix) < 6:
                hash_prefix = None
        _download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    from src.tools.torch_common import synchronize
    synchronize()
    return cached_file


class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        suffix='pth',
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.suffix = suffix
        self.save_to_disk = save_to_disk
        self.suffix = suffix

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.{}".format(name, self.suffix))
        logging.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        from src.tools.common import get_mpi_rank
        if get_mpi_rank() == 0:
            self.tag_last_checkpoint(save_file)

    def recover_or_load(self, f=None, model_only=False, load_if_has=True):
        return self.load(f, model_only, load_if_has)

    # use recover_or_load
    def load(self, f=None, model_only=False, load_if_has=True):
        if self.has_checkpoint() and load_if_has:
            f = self.get_checkpoint_file()
            model_only = False
        if not f:
            # no checkpoint could be found
            logging.info("No checkpoint found. Initializing model from scratch")
            return {}
        logging.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            if not model_only:
                logging.info("Loading optimizer from {}".format(f))
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
            else:
                checkpoint.pop("optimizer")
        if "scheduler" in checkpoint and self.scheduler:
            if not model_only:
                logging.info("Loading scheduler from {}".format(f))
                self.scheduler.load_state_dict(checkpoint.pop('scheduler'))
            else:
                checkpoint.pop("scheduler")
        if model_only:
            # if it is continuous training, this value of model_only will be false at the
            # very beginning
            if len(checkpoint) == 1:
                assert 'iteration' in checkpoint
            else:
                # the following two are used in fb_swav
                for x in ['epoch', 'amp', 'arch']:
                    if x in checkpoint:
                        del checkpoint[x]
                if len(checkpoint) > 0:
                    from pprint import pformat
                    logging.info('ignore keys = {}'.format(pformat(checkpoint.keys())))
            checkpoint = {}

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved.strip()

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        if isinstance(f, str):
            from src.tools.torch_common import torch_load
            model = torch_load(f)
            if 'model' not in model:
                return {'model': model}
            return model

        else:
            raise ValueError('should not be here')
            assert isinstance(f, dict)
            # this is a pre-loaded checkpoint dict
            return f

    def _load_model(self, checkpoint):
        load_state_dict(self.model, checkpoint.pop("model"))


