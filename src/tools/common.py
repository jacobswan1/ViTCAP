import os
import base64
import cv2
from tqdm import tqdm
import sys
import torch
import yaml
import inspect
import logging
import argparse
import shutil
import numpy as np
import os.path as op
from datetime import datetime
from future.utils import viewitems
from collections import OrderedDict
from importlib import import_module
import torch.distributed as dist
import matplotlib.pyplot as plt


# Data related helper
def img_from_base64(imagestring):
    try:
        jpgbytestring = base64.b64decode(imagestring)
        nparr = np.frombuffer(jpgbytestring, np.uint8)
        r = cv2.imdecode(nparr, cv2.IMREAD_COLOR);
        return r
    except:
        return None;


def pilimg_from_base64(imagestring):
    try:
        from PIL import Image
        import io
        jpgbytestring = base64.b64decode(imagestring)
        image = Image.open(io.BytesIO(jpgbytestring))
        return image
    except:
        return None;


def is_hvd_initialized():
    try:
        import horovod.torch as hvd
        hvd.size()
        return True
    except ImportError:
        return False
    except ValueError:
        return False


# YAML related helpers
def json_dump(obj):
    # order the keys so that each operation is deterministic though it might be
    # slower
    import json
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))



def qd_tqdm(*args, **kwargs):
    desc = kwargs.get('desc', '')
    import inspect
    frame = inspect.currentframe()
    frames = inspect.getouterframes(frame)
    frame = frames[1].frame
    line_number = frame.f_lineno
    fname = op.basename(frame.f_code.co_filename)
    message = '{}:{}'.format(fname, line_number)

    if 'desc' in kwargs:
        kwargs['desc'] = message + ' ' + desc
    else:
        kwargs['desc'] = message

    if 'mininterval' not in kwargs:
        # every 2 secons; default is 0.1 second which is too frequent
        kwargs['mininterval'] = 2

    return tqdm(*args, **kwargs)


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
            try:
                import horovod.torch as hvd
                hvd.allreduce(torch.tensor(0), name='barrier')
            except ImportError:
                return False


def dict_has_path(d, p, with_type=False):
    ps = p.split('$')
    cur_dict = d
    while True:
        if len(ps) > 0:
            k = dict_parse_key(ps[0], with_type)
            if isinstance(cur_dict, dict) and k in cur_dict:
                cur_dict = cur_dict[k]
                ps = ps[1:]
            elif isinstance(cur_dict, list):
                try:
                    k = int(k)
                except:
                    return False
                cur_dict = cur_dict[k]
                ps = ps[1:]
            else:
                return False
        else:
            return True


def execute_func(info):
    # info = {'from': module; 'import': func_name, 'param': dict}
    modules = import_module(info['from'])
    if 'param' not in info:
        return getattr(modules, info['import'])()
    else:
        return getattr(modules, info['import'])(**info['param'])


def unicode_representer(uni):
    node = yaml.ScalarNode(tag=u'tag:yaml.org,2002:str', value=uni)
    return node


def setup_yaml():
    """ https://stackoverflow.com/a/8661021 """
    represent_dict_order = lambda self, data:  self.represent_mapping('tag:yaml.org,2002:map', data.items())
    yaml.add_representer(OrderedDict, represent_dict_order)
    try:
        yaml.add_representer(unicode, unicode_representer)
    except NameError:
        logging.info('python 3 env')


def init_logging():
    np.seterr(divide = "raise", over="warn", under="warn",  invalid="raise")

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    logger_fmt = logging.Formatter('%(asctime)s.%(msecs)03d %(process)d:%(filename)s:%(lineno)s %(funcName)10s(): %(message)s')
    ch.setFormatter(logger_fmt)

    root = logging.getLogger()
    root.handlers = []
    root.addHandler(ch)
    root.setLevel(logging.INFO)
    setup_yaml()


def dict_parse_key(k, with_type):
    if with_type:
        if k[0] == 'i':
            return int(k[1:])
        else:
            return k[1:]
    return k


def dict_remove_path(d, p):
    ps = p.split('$')
    assert len(ps) > 0
    cur_dict = d
    need_delete = ()
    while True:
        if len(ps) == 1:
            if len(need_delete) > 0 and len(cur_dict) == 1:
                del need_delete[0][need_delete[1]]
            else:
                del cur_dict[ps[0]]
            return
        else:
            if len(cur_dict) == 1:
                if len(need_delete) == 0:
                    need_delete = (cur_dict, ps[0])
            else:
                need_delete = (cur_dict, ps[0])
            cur_dict = cur_dict[ps[0]]
            ps = ps[1:]


def get_all_path(d, with_type=False, leaf_only=True, with_list=True):
    assert not with_type, 'will not support'
    all_path = []

    if isinstance(d, dict):
        for k, v in d.items():
            all_sub_path = get_all_path(
                v, with_type, leaf_only=leaf_only, with_list=with_list)
            all_path.extend([k + '$' + p for p in all_sub_path])
            if not leaf_only or len(all_sub_path) == 0:
                all_path.append(k)
    elif (isinstance(d, tuple) or isinstance(d, list)) and with_list:
        for i, _v in enumerate(d):
            all_sub_path = get_all_path(
                _v, with_type,
                leaf_only=leaf_only,
                with_list=with_list,
            )
            all_path.extend(['{}$'.format(i) + p for p in all_sub_path])
            if not leaf_only or len(all_sub_path) == 0:
                all_path.append('{}'.format(i))
    return all_path


def load_from_yaml_file(file_name):
    with open(file_name, 'r') as fp:
        data = load_from_yaml_str(fp)
    while isinstance(data, dict) and '_base_' in data:
        b = op.join(op.dirname(file_name), data['_base_'])
        result = load_from_yaml_file(b)
        assert isinstance(result, dict)
        del data['_base_']
        all_key = get_all_path(data, with_list=False)
        for k in all_key:
            v = dict_get_path_value(data, k)
            dict_update_path_value(result, k, v)
        data = result
    return data


def ensure_directory(path):
    if path == '' or path == '.':
        return
    if path != None and len(path) > 0:
        assert not op.isfile(path), '{} is a file'.format(path)
        if not os.path.exists(path) and not op.islink(path):
            try:
                os.makedirs(path)
            except:
                if os.path.isdir(path):
                    # another process has done makedir
                    pass
                else:
                    raise


def save_parameters(param, folder):

    time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    write_to_yaml_file(param, op.join(folder,
        'parameters_{}.yaml'.format(time_str)))
    # save the env parameters
    # convert it to dict for py3
    write_to_yaml_file(dict(os.environ), op.join(folder,
        'env_{}.yaml'.format(time_str)))


def write_to_yaml_file(context, file_name):
    ensure_directory(op.dirname(file_name))
    with open(file_name, 'w') as fp:
        yaml.dump(context, fp, default_flow_style=False,
                encoding='utf-8', allow_unicode=True)


def load_from_yaml_str(s):
    return yaml.load(s, Loader=yaml.UnsafeLoader)


def parse_general_args():
    parser = argparse.ArgumentParser(description='General Parser')
    parser.add_argument('-c', '--config_file', help='config file',
            type=str)
    parser.add_argument('-p', '--param', help='parameter string, yaml format',
            type=str)
    parser.add_argument('-bp', '--base64_param', help='base64 encoded yaml format',
            type=str)
    args = parser.parse_args()
    kwargs =  {}
    if args.config_file:
        logging.info('loading parameter from {}'.format(args.config_file))
        configs = load_from_yaml_file(args.config_file)
        for k in configs:
            kwargs[k] = configs[k]
    if args.base64_param:
        configs = load_from_yaml_str(base64.b64decode(args.base64_param))
        for k in configs:
            if k not in kwargs:
                kwargs[k] = configs[k]
            elif kwargs[k] == configs[k]:
                continue
            else:
                logging.info('overwriting {} to {} for {}'.format(kwargs[k],
                    configs[k], k))
                kwargs[k] = configs[k]
    if args.param:
        configs = load_from_yaml_str(args.param)
        dict_ensure_path_key_converted(configs)
        for k in configs:
            if k not in kwargs:
                kwargs[k] = configs[k]
            elif kwargs[k] == configs[k]:
                continue
            else:
                logging.info('overwriting {} to {} for {}'.format(kwargs[k],
                    configs[k], k))
                kwargs[k] = configs[k]
    return kwargs


def dict_get_path_value(d, p, with_type=False):
    ps = p.split('$')
    cur_dict = d
    while True:
        if len(ps) > 0:
            k = dict_parse_key(ps[0], with_type)
            if isinstance(cur_dict, (tuple, list)):
                cur_dict = cur_dict[int(k)]
            else:
                cur_dict = cur_dict[k]
            ps = ps[1:]
        else:
            return cur_dict


def dict_update_path_value(d, p, v):
    ps = p.split('$')
    while True:
        if len(ps) == 1:
            d[ps[0]] = v
            break
        else:
            if ps[0] not in d:
                d[ps[0]] = {}
            d = d[ps[0]]
            ps = ps[1:]


def dict_update_nested_dict(a, b, overwrite=True):
    for k, v in viewitems(b):
        if k not in a:
            dict_update_path_value(a, k, v)
        else:
            if isinstance(dict_get_path_value(a, k), dict) and isinstance(v, dict):
                dict_update_nested_dict(dict_get_path_value(a, k), v, overwrite)
            else:
                if overwrite:
                    dict_update_path_value(a, k, v)


def dict_ensure_path_key_converted(a):
    for k in list(a.keys()):
        v = a[k]
        if '$' in k:
            parts = k.split('$')
            x = {}
            x_curr = x
            for p in parts[:-1]:
                x_curr[p] = {}
                x_curr = x_curr[p]
            if isinstance(v, dict):
                dict_ensure_path_key_converted(v)
            x_curr[parts[-1]] = v
            dict_update_nested_dict(a, x)
            del a[k]
        else:
            if isinstance(v, dict):
                dict_ensure_path_key_converted(v)


def dict_ensure_path_key_converted(a):
    for k in list(a.keys()):
        v = a[k]
        if '$' in k:
            parts = k.split('$')
            x = {}
            x_curr = x
            for p in parts[:-1]:
                x_curr[p] = {}
                x_curr = x_curr[p]
            if isinstance(v, dict):
                dict_ensure_path_key_converted(v)
            x_curr[parts[-1]] = v
            dict_update_nested_dict(a, x)
            del a[k]
        else:
            if isinstance(v, dict):
                dict_ensure_path_key_converted(v)


def print_frame_info():
    frame = inspect.currentframe()
    frames = inspect.getouterframes(frame)
    frame = frames[1].frame
    args, _, _, vs = inspect.getargvalues(frame)
    info = []
    info.append('func name = {}'.format(inspect.getframeinfo(frame)[2]))
    for i in args:
        try:
            info.append('{} = {}'.format(i, vs[i]))
        except:
            info.append('type({}) = {}'.format(i, type(vs[i])))
            continue
    logging.info('; '.join(info))


def worth_create(base, derived, buf_second=0):
    if not op.isfile(base) and \
            not op.islink(base) and \
            not op.isdir(base):
        return False
    if os.path.isfile(derived) and \
            os.path.getmtime(derived) > os.path.getmtime(base) - buf_second:
        return False
    else:
        return True


def read_to_buffer(file_name):
    with open(file_name, 'rb') as fp:
        all_line = fp.read()
    return all_line


def write_to_file(contxt, file_name, append=False):
    p = os.path.dirname(file_name)
    ensure_directory(p)
    if type(contxt) is str:
        contxt = contxt.encode()
    flag = 'wb'
    if append:
        flag = 'ab'
    with open(file_name, flag) as fp:
        fp.write(contxt)


def plot_to_file(xs, ys, file_name, **kwargs):
    fig = plt.figure()
    semilogy = kwargs.get('semilogy')
    if all(isinstance(x, str) for x in xs):
        xs2 = range(len(xs))
        plt.xticks(xs2, xs, rotation='vertical')
        xs = xs2
    if type(ys) is dict:
        for key in ys:
            if semilogy:
                plt.semilogy(xs, ys[key], '-o')
            else:
                plt.plot(xs, ys[key], '-o')
    else:
        if semilogy:
            plt.semilogy(xs, ys, '-o')
        else:
            plt.plot(xs, ys, '-o')
    plt.grid()
    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'])
    ensure_directory(op.dirname(file_name))
    plt.tight_layout()
    # explicitly remove the file because philly does not support overwrite
    if op.isfile(file_name):
        try:
            os.remove(file_name)
        except:
            logging.info('{} exists but could not be deleted'.format(
                file_name))
    fig.savefig(file_name)
    plt.close(fig)


def ensure_remove_dir(d):
    is_dir = op.isdir(d)
    is_link = op.islink(d)
    if is_dir:
        if not is_link:
            shutil.rmtree(d)
        else:
            os.unlink(d)

# DDP related helpers

def try_once(func):
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.info('ignore error \n{}'.format(str(e)))
            print_trace()
    return func_wrapper


def master_process_run(func):
    def func_wrapper(*args, **kwargs):
        if get_mpi_rank() == 0:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.info('ignore error \n{}'.format(str(e)))
                print_trace()
    return func_wrapper


def acquireLock(lock_f='/tmp/lockfile.LOCK'):
    ''' acquire exclusive lock file access '''
    import fcntl
    locked_file_descriptor = open(lock_f, 'w+')
    fcntl.lockf(locked_file_descriptor, fcntl.LOCK_EX)
    return locked_file_descriptor


def releaseLock(locked_file_descriptor):
    ''' release exclusive lock file access '''
    locked_file_descriptor.close()


def get_mpi_rank():
    if 'RANK' in os.environ:
        return int(os.environ['RANK'])
    return int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))

def get_mpi_size():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE'])
    return int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1'))


def get_mpi_local_rank():
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))


def get_mpi_local_size():
    if 'LOCAL_SIZE' in os.environ:
        return int(os.environ['LOCAL_SIZE'])
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE', '1'))


# TSV relate
def get_file_size(f):
    if not op.isfile(f):
        return 0
    return os.stat(f).st_size


def get_user_name():
    import getpass
    return getpass.getuser()


def print_trace():
    import traceback
    traceback.print_exc()


def limited_retry_agent(num, func, *args, **kwargs):
    for i in range(num):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.info('fails with \n{}: tried {}-th time'.format(
                e,
                i + 1))
            import time
            print_trace()
            if i == num - 1:
                raise
            time.sleep(5)


def hash_sha1(s):
    import hashlib
    from pprint import pformat
    if type(s) is not str:
        s = pformat(s)
    return hashlib.sha1(s.encode('utf-8')).hexdigest()


def exclusive_open_to_read(fname, mode='r'):
    disable_lock = os.environ.get('QD_DISABLE_EXCLUSIVE_READ_BY_LOCK')
    if disable_lock is not None:
        disable_lock = int(disable_lock)
    if not disable_lock:
        user_name = get_user_name()
        from src.tools.common import acquireLock, releaseLock
        # from src.tools.common import acquireLock, releaseLock
        lock_fd = acquireLock(op.join('/tmp',
            '{}_lock_{}'.format(user_name, hash_sha1(fname))))
    # in AML, it could fail with Input/Output error. If it fails, we will
    # use azcopy as a fall back solution for reading
    fp = limited_retry_agent(10, open, fname, mode)

    if not disable_lock:
        releaseLock(lock_fd)
    return fp
