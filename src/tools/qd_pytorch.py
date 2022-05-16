import re
import glob
import logging
from pprint import pformat
from datetime import datetime
from torch.utils.data import Dataset
from src.tools.tsv.tsv_io import TSVDataset, load_list_file
from src.tools.common import load_from_yaml_file, img_from_base64

try:
    from itertools import izip as zip
except:
    # python 3
    pass


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


def load_latest_parameters(folder):
    yaml_file = get_latest_parameter_file(folder)
    logging.info('using {}'.format(yaml_file))
    param = load_from_yaml_file(yaml_file)
    return param


def get_latest_parameter_file(folder):
    import os.path as op
    yaml_pattern = op.join(folder,
            'parameters_*.yaml')
    yaml_files = glob.glob(yaml_pattern)
    assert len(yaml_files) > 0, folder
    def parse_time(f):
        m = re.search('.*parameters_(.*)\.yaml', f)
        t = datetime.strptime(m.group(1), '%Y_%m_%d_%H_%M_%S')
        return t
    times = [parse_time(f) for f in yaml_files]
    fts = [(f, t) for f, t in zip(yaml_files, times)]
    fts.sort(key=lambda x: x[1], reverse=True)
    yaml_file = fts[0][0]
    return yaml_file


