from src.tools.common import exclusive_open_to_read
import time
import logging
import glob
import json
import random
from src.tools.common import ensure_directory
import six
import os
import os.path as op
import shutil
import re
try:
    from itertools import izip as zip
except ImportError:
    pass

import progressbar
from src.tools.common import qd_tqdm as tqdm


def get_default_splits():
    return ['train', 'trainval', 'test', 'val']


def get_tsv_lineidx(tsv_file):
    return tsv_file[:-3] + 'lineidx'


def get_tsv_lineidx_8b(tsv_file):
    return tsv_file[:-3] + 'lineidx.8b'


def rm_tsv(tsv_file):
    if op.isfile(tsv_file):
        os.remove(tsv_file)
        line_idx = op.splitext(tsv_file)[0] + '.lineidx'
        if op.isfile(line_idx):
            os.remove(line_idx)


def tsv_rm(tsv_file):
    rm_tsv(tsv_file)


def tsv_mv(src_file, dst_file):
    shutil.move(src_file, dst_file)
    src_idx = op.splitext(src_file)[0] + '.lineidx'
    if op.isfile(src_idx):
        dst_idx = op.splitext(dst_file)[0] + '.lineidx'
        shutil.move(src_idx, dst_idx)


def reorder_tsv_keys(in_tsv_file, ordered_keys, out_tsv_file):
    tsv = TSVFile(in_tsv_file)
    logging.info('loading keys in input')
    keys = [tsv.seek_first_column(i) for i in tqdm(range(len(tsv)), mininterval=2)]
    key_to_idx = {key: i for i, key in enumerate(keys)}
    def gen_rows():
        logging.info('writing')
        for key in tqdm(ordered_keys, mininterval=2):
            idx = key_to_idx[key]
            yield tsv.seek(idx)
    tsv_writer(gen_rows(), out_tsv_file)


def read_to_character(fp, c):
    result = []
    while True:
        s = fp.read(32)
        assert s != ''
        if c in s:
            result.append(s[: s.index(c)])
            break
        else:
            result.append(s)
    return ''.join(result)


class CompositeTSVFile(object):
    def __init__(self, list_file, seq_file, cache_policy=False,
                 hold_buffer=0,
                 ):
        # list_file can be a loaded or constructed pair of index, rather than a
        # filename to load. In this case, seq_file will be a list of dataset,
        # which should implement len() and __getitem__() so that we can
        # reference it.
        self.seq_file = seq_file
        self.list_file = list_file
        self.cache_policy = cache_policy
        self.seq = None
        self.tsvs = []
        # please do ont call ensure_initialized here. we wil always do it
        # lazily. we may load a huge amount of seq, which could be super slow
        # when spawning multiple processes.

        # this means, how many tsv fp pointer we will hold. If it is 0 or less
        # than 0, we will hold all fp pointers we need. If it is larger than 0,
        # we only hold some, which are kept in self.hold_sources
        self.hold_buffer = hold_buffer
        self.hold_sources = []

    def __repr__(self):
        return 'CompositeTSVFile(list_file={}, seq_file={})'.format(
            self.seq_file,
            self.list_file
        )

    def __getitem__(self, index):
        self.ensure_initialized()
        idx_source, idx_row = map(int, self.seq[index])
        start = time.time()
        result = self.tsvs[idx_source].seek(idx_row)
        end = time.time()
        if end - start > 10:
            logging.info('too long to load fname = {}, source={}, row={}'.format(
                self.tsvs[idx_source],
                idx_source,
                idx_row,
            ))

        if self.hold_buffer > 0:
            if idx_source not in self.hold_sources:
                if len(self.hold_sources) >= self.hold_buffer:
                    close_idx_source = self.hold_sources.pop(0)
                    self.tsvs[close_idx_source].close_fp()
                self.hold_sources.append(idx_source)

        return result

    def __len__(self):
        self.ensure_initialized()
        return len(self.seq)

    def __iter__(self):
        self.ensure_initialized()
        self.next_row = 0
        self.seq = iter(self.seq)
        return self

    def __next__(self):
        # this function is not well tested. let's have a breakpoint here
        import ipdb;ipdb.set_trace(context=15)
        if self.next_row >= len(self):
            raise StopIteration
        idx_source, idx_row = map(int, next(self.seq))
        return self.tsvs[idx_source][idx_row]

    def release(self):
        # this is to ensure we released all the resources
        self.seq = None
        for t in self.tsvs:
            t.close()

    def seek_first_column(self, index):
        self.ensure_initialized()
        idx_source, idx_row = map(int, self.seq[index])
        return self.tsvs[idx_source].seek_first_column(idx_row)

    def get_composite_source_idx(self):
        return [int(i) for i, _ in self.seq]

    def ensure_initialized(self):
        if self.seq is None:
            if isinstance(self.list_file, str) and \
                    isinstance(self.seq_file, str):
                self.seq = TSVFile(self.seq_file)
                self.tsvs = [TSVFile(f, self.cache_policy) for f in load_list_file(self.list_file)]
            else:
                self.seq = self.list_file
                self.tsvs = self.seq_file


class TSVFile(object):
    def __init__(self, tsv_file, cache_policy=None):
        self.tsv_file = tsv_file
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx'
        self.lineidx_8b = self.lineidx + '.8b'
        self.has_lineidx_8b = op.isfile(self.lineidx_8b)
        # FIXME: Note this param is forcefully set to false to avoid additional opening files. But this saves lots of
        # memory and is preferred when there's memory leaking.
        # self.has_lineidx_8b = False
        self._fp = None
        self._lineidx = None
        self.fp8b = None
        self.cache_policy = cache_policy
        self.close_fp_after_read = False
        if os.environ.get('QD_TSV_CLOSE_FP_AFTER_READ'):
            self.close_fp_after_read = bool(os.environ['QD_TSV_CLOSE_FP_AFTER_READ'])
        # the process always keeps the process which opens the
        # file. If the pid is not equal to the currrent pid, we will re-open
        # teh file.
        self.pid = None
        self.lineidx_8b_pid = None

        self._cache()
        self._len = None

    def close_fp(self):
        if self._fp:
            self._fp.close()
            self._fp = None

    def release(self):
        self.close_fp()
        self._lineidx = None

    def close(self):
        # @deprecated('use close_fp to make it more clear not to release lineidx')
        self.close_fp()
        # we should clear out all resource.
        self._lineidx = None

    def __del__(self):
        self.release()

    def __str__(self):
        return "TSVFile(tsv_file='{}')".format(self.tsv_file)

    def __repr__(self):
        return str(self)

    def __iter__(self):
        self._ensure_tsv_opened()
        self._fp.seek(0)
        self.next_row = 0
        return self

    def __next__(self):
        if self.next_row >= len(self):
            raise StopIteration
        self.next_row += 1
        result = [s.strip() for s in self._fp.readline().split('\t')]
        return result

    def num_rows(self):
        if self._len is None:
            if op.isfile(self.lineidx_8b):
                from src.tools.common import get_file_size
                self._len = get_file_size(self.lineidx_8b) // 8
            else:
                self._ensure_lineidx_loaded()
                self._len = len(self._lineidx)
        return self._len

    def get_key(self, idx):
        return self.seek_first_column(idx)

    def seek(self, idx):
        self._ensure_tsv_opened()
        pos = self.get_offset(idx)

        self._fp.seek(pos)
        result = [s.strip() for s in self._fp.readline().split('\t')]

        if self.close_fp_after_read:
            self.close_fp()

        return result

    def seek_first_column(self, idx):
        self._ensure_tsv_opened()
        pos = self.get_offset(idx)
        self._fp.seek(pos)
        return read_to_character(self._fp, '\t')

    def get_offset(self, idx):
        # do not use op.isfile() to check whether lineidx_8b exists as it may
        # incur API call for blobfuse, which will be super slow if we enumerate
        # a bunch of data
        if self.has_lineidx_8b:
            if self.fp8b is None:
                self.fp8b = exclusive_open_to_read(self.lineidx_8b, 'rb')
                self.lineidx_8b_pid = os.getpid()
            if self.lineidx_8b_pid != os.getpid():
                self.fp8b.close()
                logging.info('re-open {} because the process id changed'.format(
                    self.lineidx_8b))
                self.fp8b= exclusive_open_to_read(self.lineidx_8b, 'rb')
                self.lineidx_8b_pid = os.getpid()
            self.fp8b.seek(idx * 8)
            return int.from_bytes(self.fp8b.read(8), 'little')
        else:
            self._ensure_lineidx_loaded()
            pos = self._lineidx[idx]
            return pos

    def __getitem__(self, index):
        return self.seek(index)

    def __len__(self):
        return self.num_rows()

    def _ensure_lineidx_loaded(self):
        if self._lineidx is None:
            # if not op.isfile(self.lineidx) and not op.islink(self.lineidx):
                # generate_lineidx(self.tsv_file, self.lineidx)
            # with open(self.lineidx, 'r') as fp:

            # with limited_retry_agent(10, open, self.lineidx, 'r') as fp:
            with exclusive_open_to_read(self.lineidx) as fp:
                self._lineidx = tuple([int(i.strip()) for i in fp.readlines()])

            # Comment this as its keep POPPING UP WHICH MADE ME CRAZY! FUUUUCK!
            logging.info('loaded {} from {}'.format(
                len(self._lineidx),
                self.lineidx
            ))

    def _cache(self):
        if self.cache_policy == 'memory':
            # make sure the tsv is opened here. don't put it in seek. If we put
            # it in the first call of seek(), it is loading all the content
            # there. With multi-workers in pytorch, each worker has to read all
            # the files and cache it to memory. If we load it here in the main
            # thread, it won't copy it to each worker
            logging.info('caching {} to memory'.format(self.tsv_file))
            from io import StringIO
            result = StringIO()
            total = op.getsize(self.tsv_file)
            import psutil
            avail = psutil.virtual_memory().available
            if avail < total:
                logging.info('not enough memory to cache {} < {}. fall back'.format(
                    avail, total))
            else:
                pbar = tqdm(total=total/1024./1024.)
                with open(self.tsv_file, 'r') as fp:
                    while True:
                        x = fp.read(1024*1024*100)
                        if len(x) == 0:
                            break
                        pbar.update(len(x) / 1024./1024.)
                        result.write(x)
                self._fp = result

        elif self.cache_policy == 'tmp':
            tmp_tsvfile = op.join('/tmp', self.tsv_file)
            tmp_lineidx = op.join('/tmp', self.lineidx)
            ensure_directory(op.dirname(tmp_tsvfile))

            from src.tools.common import ensure_copy_file
            ensure_copy_file(self.tsv_file, tmp_tsvfile)
            ensure_copy_file(self.lineidx, tmp_lineidx)

            self.tsv_file = tmp_tsvfile
            self.lineidx = tmp_lineidx
            # do not run the following. Supposedly, this function is called in
            # init function. If we use multiprocess, the file handler will be
            # duplicated and thus the seek will have some race condition if we
            # have the following.
        elif self.cache_policy is not None:
            raise ValueError('unkwown cache policy {}'.format(self.cache_policy))

    def _ensure_tsv_opened(self):
        if self.cache_policy == 'memory':
            assert self._fp is not None
            return

        if self._fp is None:
            self._fp = exclusive_open_to_read(self.tsv_file)
            self.pid = os.getpid()

        if self.pid != os.getpid():
            self._fp.close()
            logging.info('re-open {} because the process id changed'.format(self.tsv_file))
            from src.tools.common import print_opened_files
            print_opened_files()
            self._fp = exclusive_open_to_read(self.tsv_file)
            self.pid = os.getpid()


class TSVDataset(object):
    def __init__(self, name, data_root=None):
        self.name = name
        if data_root is None:
            if os.environ.get('QD_DATA_ROOT') is not None:
                data_root = os.environ['QD_DATA_ROOT']
            else:
                from pathlib import Path
                proj_root = op.realpath(Path('.'))
                # proj_root = op.dirname(op.dirname(op.dirname(op.realpath(__file__).parent)))
                data_root = op.join(proj_root, 'data')
        data_root = op.join(data_root, name)
        self._data_root = op.relpath(data_root)
        self._fname_to_tsv = {}

        self._split_to_key_to_idx = {}

    def __repr__(self):
        return 'TSVDataset({})'.format(self.name)

    def __str__(self):
        return 'TSVDataset({})'.format(self.name)

    def seek_by_key(self, key, split, t=None, version=None):
        idx = self.get_idx_by_key(key, split)
        return next(self.iter_data(split, t, version, filter_idx=[idx]))

    def seek_by_idx(self, idx, split, t=None, version=None):
        return next(self.iter_data(split, t, version, filter_idx=[idx]))

    def load_labelmap(self):
        return load_list_file(self.get_labelmap_file())

    def load_pos_labelmap(self):
        return load_list_file(self.get_pos_labelmap_file())

    def get_tree_file(self):
        return op.join(self._data_root, 'tree.txt')

    def get_labelmap_file(self):
        return op.join(self._data_root, 'labelmap.txt')

    def load_txt(self, t='labelmap'):
        return load_list_file(self.get_txt(t))

    # labelmap or attribute map
    def get_txt(self, t='labelmap'):
        return op.join(self._data_root, '{}.txt'.format(t))

    def get_pos_labelmap_file(self):
        return op.join(self._data_root, 'labelmap.pos.txt')

    def get_train_shuffle_file(self):
        return self.get_shuffle_file('train')

    def get_shuffle_file(self, split_name):
        return op.join(self._data_root, '{}.shuffle.txt'.format(split_name))

    def get_labelmap_of_noffset_file(self):
        return op.join(self._data_root, 'noffsets.label.txt')

    def get_idx_by_key(self, key, split):
        if split in self._split_to_key_to_idx:
            key_to_idx = self._split_to_key_to_idx[split]
        else:
            key_to_idx = {k: i for i, k in enumerate(self.load_keys(split))}
            self._split_to_key_to_idx[split] = key_to_idx
        idx = key_to_idx[key]
        return idx

    def load_key_to_idx(self, split):
        result = {}
        for i, row in enumerate(self.iter_data(split, 'label')):
            key = row[0]
            assert key not in result
            result[key] = i
        return result

    def load_keys(self, split):
        assert self.has(split, 'label')
        result = []
        for row in tqdm(self.iter_data(split, 'label'), mininterval=2):
            result.append(row[0])
        return result

    def dynamic_update(self, dataset_ops):
        '''
        sometimes, we update the dataset, and here, we should update the file
        path
        '''
        if len(dataset_ops) >= 1 and dataset_ops[0]['op'] == 'sample':
            self._data_root = op.join('./output/data/',
                    '{}_{}_{}'.format(self.name,
                        dataset_ops[0]['sample_label'],
                        dataset_ops[0]['sample_image']))
        elif len(dataset_ops) >= 1 and dataset_ops[0]['op'] == 'mask_background':
            target_folder = op.join('./output/data',
                    '{}_{}_{}'.format(self.name,
                        '.'.join(map(str, dataset_ops[0]['old_label_idx'])),
                        dataset_ops[0]['new_label_idx']))
            self._data_root = target_folder

    def get_test_tsv_file(self, t=None):
        return self.get_data('test', t)

    def get_test_tsv_lineidx_file(self):
        return op.join(self._data_root, 'test.lineidx')

    def get_train_tsvs(self, t=None):
        if op.isfile(self.get_data('train', t)):
            return [self.get_data('train', t)]
        trainx_file = op.join(self._data_root, 'trainX.tsv')
        if not op.isfile(trainx_file):
            return []
        train_x = load_list_file(trainx_file)
        if t is None:
            return train_x
        elif t =='label':
            if op.isfile(self.get_data('trainX', 'label')):
                return load_list_file(self.get_data('trainX', 'label'))
            else:
                files = [op.splitext(f)[0] + '.label.tsv' for f in train_x]
                return files

    def get_train_tsv(self, t=None):
        return self.get_data('train', t)

    def get_lineidx(self, split_name):
        return op.join(self._data_root, '{}.lineidx'.format(split_name))

    def get_latest_version(self, split, t=None):
        assert t is not None, 'if it is none, it is always 0'
        v = 0
        if t is None:
            pattern = op.join(self._data_root, '{}.v*.tsv'.format(split))
            re_pattern = '{}\.v([0-9]*)\.tsv'.format(split)
        else:
            pattern = op.join(self._data_root, '{}.{}.v*.tsv'.format(
                split, t))
            re_pattern = '{}\.{}\.v([0-9]*)\.tsv'.format(split, t)
        all_file = glob.glob(pattern)
        import re
        re_results = [re.match(re_pattern, op.basename(f)) for f in all_file]
        candidates = ([int(re_result.groups()[0]) for re_result, f in
            zip(re_results, all_file) if re_result])
        if len(candidates) > 0:
            v = max(candidates)
        assert v >= 0
        return v

    def get_gen_info_data(self, split, t=None, version=None):
        return self.get_data(split, '{}.generate.info'.format(t), version=version)

    def get_file(self, fname):
        return op.join(self._data_root, fname)

    def get_data(self, split_name, t=None, version=None):
        '''
        e.g. split_name = train, t = label
        if version = None or 0,  return train.label.tsv
        we don't have train.label.v0.tsv
        if version = 3 > 0, return train.label.v3.tsv
        if version = -1, return the highest version
        '''
        if t is None:
            # in this case, it is an image split, which has no version
            version = None
        if version is None or version in [0,'None','0']:
            if t is None:
                return op.join(self._data_root, '{}.tsv'.format(split_name))
            else:
                return op.join(self._data_root, '{}.{}.tsv'.format(split_name,
                    t))
        elif version == -1:
            if not op.isfile(self.get_data(split_name, t)):
                return self.get_data(split_name, t)
            v = self.get_latest_version(split_name, t)
            return self.get_data(split_name, t, v)
        else:
            return op.join(self._data_root, '{}.{}.v{}.tsv'.format(split_name,
                t, version))

    def get_num_train_image(self):
        if op.isfile(self.get_data('trainX')):
            if op.isfile(self.get_shuffle_file('train')):
                return len(load_list_file(self.get_shuffle_file('train')))
            else:
                return 0
        else:
            return len(load_list_file(op.join(self._data_root, 'train.lineidx')))

    def get_trainval_tsv(self, t=None):
        return self.get_data('trainval', t)

    def get_noffsets_file(self):
        return op.join(self._data_root, 'noffsets.txt')

    def load_noffsets(self):
        logging.info('deprecated: pls generate it on the fly')
        return load_list_file(self.get_noffsets_file())

    def load_inverted_label(self, split, version=None, label=None):
        fname = self.get_data(split, 'inverted.label', version)
        if not op.isfile(fname):
            return {}
        elif label is None:
            tsv = TSVFile(fname)
            num_rows = len(tsv)
            result = {}
            for row in tqdm(tsv, total=num_rows, mininterval=2):
                assert row[0] not in result
                assert len(row) == 2
                ss = row[1].split(' ')
                if len(ss) == 1 and ss[0] == '':
                    result[row[0]] = []
                else:
                    result[row[0]] = list(map(int, ss))
            return result
        else:
            all_label = load_list_file(self.get_data(split, 'labelmap', version))
            if label not in all_label:
                return {}
            result = {}
            idx = all_label.index(label)
            tsv = self._retrieve_tsv(fname)
            row = tsv.seek(idx)
            assert row[0] == label
            ss = row[1].split(' ')
            if len(ss) == 1 and ss[0] == '':
                result[row[0]] = []
            else:
                result[row[0]] = list(map(int, ss))
            return result

    def load_inverted_label_as_list(self, split, version=None, label=None):
        fname = self.get_data(split, 'inverted.label', version)
        if not op.isfile(fname):
            return []
        elif label is None:
            rows = tsv_reader(fname)
            result = []
            for row in rows:
                assert len(row) == 2
                ss = row[1].split(' ')
                if len(ss) == 1 and ss[0] == '':
                    result.append((row[0], []))
                else:
                    result.append((row[0], list(map(int, ss))))
            return result
        else:
            all_label = self.load_labelmap()
            result = []
            idx = all_label.index(label)
            tsv = self._retrieve_tsv(fname)
            row = tsv.seek(idx)
            assert row[0] == label
            ss = row[1].split(' ')
            if len(ss) == 1 and ss[0] == '':
                result.append((row[0], []))
            else:
                result.append((row[0], list(map(int, ss))))
            return result

    def has(self, split, t=None, version=None):
        return op.isfile(self.get_data(split, t, version)) or (
                op.isfile(self.get_data('{}X'.format(split), t, version)) and
                op.isfile(self.get_shuffle_file(split)))

    def last_update_time(self, split, t=None, version=None):
        tsv_file = self.get_data(split, t, version)
        if op.isfile(tsv_file):
            return os.path.getmtime(tsv_file)
        assert version is None or version == 0, 'composite dataset always v=0'
        tsv_file = self.get_data('{}X'.format(split), t, version)
        assert op.isfile(tsv_file)
        return os.path.getmtime(tsv_file)

    def iter_composite(self, split, t, version, filter_idx=None):
        splitX = split + 'X'
        file_list = load_list_file(self.get_data(splitX, t, version))
        tsvs = [self._retrieve_tsv(f) for f in file_list]
        shuffle_file = self.get_shuffle_file(split)
        if filter_idx is None:
            shuffle_tsv_rows = tsv_reader(shuffle_file)
            for idx_source, idx_row in shuffle_tsv_rows:
                idx_source, idx_row = int(idx_source), int(idx_row)
                row = tsvs[idx_source].seek(idx_row)
                if len(row) == 3:
                    row[1] == 'dont use'
                yield row
        else:
            shuffle_tsv = self._retrieve_tsv(shuffle_file)
            for i in filter_idx:
                idx_source, idx_row = shuffle_tsv.seek(i)
                idx_source, idx_row = int(idx_source), int(idx_row)
                row = tsvs[idx_source].seek(idx_row)
                if len(row) == 3:
                    row[1] == 'dont use'
                yield row

    def num_rows(self, split, t=None, version=None):
        f = self.get_data(split, t, version)
        if op.isfile(f) or op.islink(f):
            return TSVFile(f).num_rows()
        else:
            f = self.get_data(split + 'X', version=version)
            assert op.isfile(f), f
            return len(load_list_file(self.get_shuffle_file(split)))

    def iter_data(self, split, t=None, version=None,
            unique=False, filter_idx=None, progress=False):
        if progress:
            if filter_idx is None:
                num_rows = self.num_rows(split)
            else:
                num_rows = len(filter_idx)
            pbar = progressbar.ProgressBar(maxval=num_rows).start()
        splitX = split + 'X'
        if not op.isfile(self.get_data(split, t, version)) and \
                op.isfile(self.get_data(splitX, t, version)):
            if t is not None:
                if unique:
                    returned = set()
                for i, row in enumerate(self.iter_composite(split, t, version,
                        filter_idx=filter_idx)):
                    if unique and row[0] in returned:
                        continue
                    else:
                        yield row
                        if unique:
                            returned.add(row[0])
                    if progress:
                        pbar.update(i)
            else:
                rows_data = self.iter_composite(split, None, version=version,
                        filter_idx=filter_idx)
                logging.info('breaking change: label is ignore for t=None')
                #rows_label = self.iter_data(split, 'label', version=version,
                        #filter_idx=filter_idx)
                if unique:
                    returned = set()
                for i, r in enumerate(rows_data):
                    if unique and r[0] in returned:
                        continue
                    else:
                        yield r
                        if unique:
                            returned.add(r[0])
                    if progress:
                        pbar.update(i)
        else:
            fname = self.get_data(split, t, version)
            if not op.isfile(fname):
                logging.info('no {}'.format(fname))
                return
            if filter_idx is None:
                for i, row in enumerate(tsv_reader(self.get_data(
                    split, t, version))):
                    yield row
                    if progress:
                        pbar.update(i)
            else:
                fname = self.get_data(split, t, version)
                tsv = self._retrieve_tsv(fname)
                if progress:
                    for i in tqdm(filter_idx):
                        yield tsv.seek(i)
                else:
                    for i in filter_idx:
                        yield tsv.seek(i)


    def _retrieve_tsv(self, fname):
        if fname in self._fname_to_tsv:
            tsv = self._fname_to_tsv[fname]
        else:
            tsv = TSVFile(fname)
            self._fname_to_tsv[fname] = tsv
        return tsv

    def safe_write_data(self, rows, split, t=None, version=None,
                        generate_info=None, force=False):
        assert force or not self.has(split, t, version)
        if generate_info is None:
            from src.tools.common import get_frame_info
            info = get_frame_info(last=1)
            def gen_info():
                for k, v in info.items():
                    if isinstance(v, str):
                        yield k, v
            generate_info = gen_info()
        self.write_data(rows, split, t, version,
                        generate_info=generate_info)

    def write_data(self, rows, split, t=None, version=None, generate_info=None):
        out_tsv = self.get_data(split, t, version)
        tsv_writer(rows, out_tsv)
        if generate_info is not None:
            out_tsv = self.get_data(split, '{}.generate.info'.format(t), version=version)
            tsv_writer(generate_info, out_tsv)

    def update_data(self, rows, split, t, generate_info=None):
        '''
        if the data are the same, we will not do anything.
        '''
        assert t is not None
        v = self.get_latest_version(split, t)
        if self.has(split, t, v):
            is_equal = True
            # we first save it to a tmp tsv file
            self.write_data(rows, split, t + '.tmp', v + 1)
            for origin_row, new_row in zip(self.iter_data(split, t, v),
                    self.iter_data(split, t + '.tmp', v + 1)):
                if len(origin_row) != len(new_row):
                    is_equal = False
                    break
                for o, n in zip(origin_row, new_row):
                    if o != n:
                        is_equal = False
                        break
                if not is_equal:
                    break
            if not is_equal:
                logging.info('creating {} for {}'.format(v + 1, self.name))
                if generate_info:
                    self.write_data(generate_info, split, '{}.generate.info'.format(t), v + 1)
                tsv_mv(self.get_data(split, t + '.tmp', v + 1),
                        self.get_data(split, t, v + 1))
                return v + 1
            else:
                logging.info('ignore to create since the label matches the latest')
        else:
            assert v == 0
            v = -1
            logging.info('creating {} for {}'.format(v + 1, self.name))
            if generate_info:
                self.write_data(generate_info, split, '{}.generate.info'.format(t), v + 1)
            self.write_data(rows, split, t, version=v + 1)
            return v + 1

    def load_composite_source_data_split(self, split):
        splitX = split + 'X'
        pattern = 'data/(.*)/(.*)\.tsv'
        tsv_sources = [l for l, in tsv_reader(self.get_data(splitX))]
        matched_result = [re.match(pattern, l).groups()
                for l in tsv_sources]

        return [(d, s) for d, s in matched_result]

    def load_composite_source_data_split_versions(self, split):
        # this function is only valid if we generated the composite dataset
        # from tsv, not from db. if it is from db, there is no file of
        # origin.label. use load_composite_source_data_split, instead.
        splitX = split + 'X'
        pattern = 'data/(.*)/(train|trainval|test)\.label\.v(.*)\.tsv'
        tsv_sources = [l for l, in tsv_reader(self.get_data(splitX,
            'origin.label'))]
        matched_result = [re.match(pattern, l).groups()
                for l in tsv_sources]

        return [(d, s, int(v)) for d, s, v in matched_result]


class TSVSplitProperty(object):
    '''
        one instance of this class mean one tsv file or one composite tsv, it could
        be label tsv, or hw tsv, or image tsv
    '''
    def __init__(self, data, split, t=None, version=0, cache_policy=None,
                 hold_buffer=0):
        self.data = data
        self.split = split
        self.t = t
        self.version = version
        dataset = TSVDataset(data)
        single_tsv = dataset.get_data(split, t, version)
        is_single_tsv = op.isfile(single_tsv)
        if is_single_tsv:
            self.tsv = TSVFile(dataset.get_data(split, t, version),
                    cache_policy)
        else:
            splitX = split + 'X'
            list_file = dataset.get_data(splitX, t, version=version)
            seq_file = dataset.get_shuffle_file(split)
            assert op.isfile(list_file) and op.isfile(seq_file), (
                '{}, {}/{} not available'.format(single_tsv, list_file, seq_file)
            )
            self.tsv = CompositeTSVFile(list_file, seq_file, cache_policy,
                                        hold_buffer=hold_buffer)

    def __repr__(self):
        return 'TSVSplitProperty(tsv={})'.format(
            self.tsv
        )

    def __getitem__(self, index):
        row = self.tsv[index]
        return row

    def __len__(self):
        return len(self.tsv)

    def num_rows(self):
        return len(self)

    def __iter__(self):
        return iter(self.tsv)

    def get_key(self, i):
        return self.tsv.seek_first_column(i)

    def seek_first_column(self, idx):
        return self.tsv.seek_first_column(idx)

    def get_composite_source_idx(self):
        return self.tsv.get_composite_source_idx()


def tsv_writers(all_values, tsv_file_names, sep='\t'):
    # values: a list of [row1, row2]. each row goes to each tsv_file_name
    for tsv_file_name in tsv_file_names:
        ensure_directory(os.path.dirname(tsv_file_name))
    tsv_lineidx_files = [os.path.splitext(tsv_file_name)[0] + '.lineidx'
        for tsv_file_name in tsv_file_names]
    tsv_file_name_tmps = [tsv_file_name + '.tmp' for tsv_file_name in
            tsv_file_names]
    tsv_lineidx_file_tmps = [tsv_lineidx_file + '.tmp' for tsv_lineidx_file in
            tsv_lineidx_files]
    sep = sep.encode()
    assert all_values is not None
    fps = [open(tsv_file_name_tmp, 'wb') for tsv_file_name_tmp in
            tsv_file_name_tmps]
    fpidxs = [open(tsv_lineidx_file_tmp, 'w') for tsv_lineidx_file_tmp in
            tsv_lineidx_file_tmps]
    idxs = [0 for _ in fps]
    for values in all_values:
        assert values is not None
        for i, (value, fp, fpidx) in enumerate(zip(values, fps, fpidxs)):
            value = map(lambda v: v if type(v) == bytes else str(v).encode(),
                    value)
            v = sep.join(value) + b'\n'
            fp.write(v)
            fpidx.write(str(idxs[i]) + '\n')
            idxs[i] = idxs[i]+ len(v)
    for f in fps:
        f.close()
    for f in fpidxs:
        f.close()
    # the following might crash if there are two processes which are writing at
    # the same time. One process finishes the renaming first and the second one
    # will crash. In this case, we know there must be some errors when you run
    # the code, and it should be a bug to fix rather than to use try-catch to
    # protect it here.
    for tsv_file_name_tmp, tsv_file_name in zip(tsv_file_name_tmps,
            tsv_file_names):
        os.rename(tsv_file_name_tmp, tsv_file_name)
    for tsv_lineidx_file_tmp, tsv_lineidx_file in zip(tsv_lineidx_file_tmps,
            tsv_lineidx_files):
        os.rename(tsv_lineidx_file_tmp, tsv_lineidx_file)


def iter_caption_to_json(iter_caption, json_file):
    # save gt caption to json format so thet we can call the api
    key_captions = [(key, json.loads(p)) for key, p in iter_caption]

    info = {
        'info': 'dummy',
        'licenses': 'dummy',
        'type': 'captions',
    }
    info['images'] = [{'file_name': k, 'id': k} for k, _ in key_captions]
    n = 0
    annotations = []
    for k, cs in key_captions:
        for c in cs:
            annotations.append({
                'image_id': k,
                'caption': c['caption'],
                'id': n
            })
            n += 1
    info['annotations'] = annotations
    from src.tools.common import write_to_file
    write_to_file(json.dumps(info), json_file)


def tsv_writer(values, tsv_file_name, sep='\t'):
    ensure_directory(os.path.dirname(tsv_file_name))
    tsv_lineidx_file = os.path.splitext(tsv_file_name)[0] + '.lineidx'
    tsv_8b_file = tsv_lineidx_file + '.8b'
    idx = 0
    tsv_file_name_tmp = tsv_file_name + '.tmp'
    tsv_lineidx_file_tmp = tsv_lineidx_file + '.tmp'
    tsv_8b_file_tmp = tsv_8b_file + '.tmp'
    import sys
    is_py2 = sys.version_info.major == 2
    if not is_py2:
        sep = sep.encode()
    with open(tsv_file_name_tmp, 'wb') as fp, open(tsv_lineidx_file_tmp, 'w') as fpidx, open(tsv_8b_file_tmp, 'wb') as fp8b:
        assert values is not None
        for value in values:
            assert value is not None
            if is_py2:
                v = sep.join(map(lambda v: str(v) if not isinstance(v, six.string_types) else v, value)) + '\n'
                if type(v) is unicode:
                    v = v.encode('utf-8')
            else:
                value = map(lambda v: v if type(v) == bytes else str(v).encode(),
                        value)
                v = sep.join(value) + b'\n'
            fp.write(v)
            fpidx.write(str(idx) + '\n')
            # although we can use sys.byteorder to retrieve the system-default
            # byte order, let's use little always to make it consistent and
            # simple
            fp8b.write(idx.to_bytes(8, 'little'))
            idx = idx + len(v)
    # the following might crash if there are two processes which are writing at
    # the same time. One process finishes the renaming first and the second one
    # will crash. In this case, we know there must be some errors when you run
    # the code, and it should be a bug to fix rather than to use try-catch to
    # protect it here.
    os.rename(tsv_file_name_tmp, tsv_file_name)
    os.rename(tsv_lineidx_file_tmp, tsv_lineidx_file)
    os.rename(tsv_8b_file_tmp, tsv_8b_file)


def tsv_reader(tsv_file_name, sep='\t'):
    with open(tsv_file_name, 'r') as fp:
        for i, line in enumerate(fp):
            yield [x.strip() for x in line.split(sep)]


def delete_tsv_files(tsvs):
    for t in tsvs:
        if op.isfile(t):
            os.remove(t)
        line = op.splitext(t)[0] + '.lineidx'
        if op.isfile(line):
            os.remove(line)

def concat_files(ins, out):
    ensure_directory(op.dirname(out))
    out_tmp = out + '.tmp'
    with open(out_tmp, 'wb') as fp_out:
        for i, f in enumerate(ins):
            logging.info('concating {}/{} - {}'.format(i, len(ins), f))
            with open(f, 'rb') as fp_in:
                shutil.copyfileobj(fp_in, fp_out, 1024*1024*10)
    os.rename(out_tmp, out)

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


def concat_tsv_files(tsvs, out_tsv):
    if len(tsvs) == 1 and tsvs[0] == out_tsv:
        return
    concat_files(tsvs, out_tsv)
    sizes = [os.stat(t).st_size for t in tsvs]
    import numpy as np
    sizes = np.cumsum(sizes)
    all_idx = []
    for i, t in enumerate(tsvs):
        for idx in load_list_file(op.splitext(t)[0] + '.lineidx'):
            if i == 0:
                all_idx.append(idx)
            else:
                all_idx.append(str(int(idx) + sizes[i - 1]))
    write_to_file('\n'.join(all_idx), op.splitext(out_tsv)[0] + '.lineidx')


def csv_reader(tsv_file_name):
    return tsv_reader(tsv_file_name, ',')


def get_meta_file(tsv_file):
    return op.splitext(tsv_file)[0] + '.meta.yaml'


def extract_label(full_tsv, label_tsv):
    if op.isfile(label_tsv):
        logging.info('label file exists and will skip to generate: {}'.format(
            label_tsv))
        return
    if not op.isfile(full_tsv):
        logging.info('the file of {} does not exist'.format(full_tsv))
        return
    rows = tsv_reader(full_tsv)

    def gen_rows():
        for i, row in enumerate(rows):
            if (i % 1000) == 0:
                logging.info('extract_label: {}-{}'.format(full_tsv, i))
            del row[2]
            assert len(row) == 2
            assert type(row[0]) == str
            assert type(row[1]) == str
            yield row
    tsv_writer(gen_rows(), label_tsv)


def create_inverted_tsv(rows, inverted_label_file, label_map):
    '''
    deprecated, use create_inverted_list
    save the results based on the label_map in label_map_file. The benefit is
    to seek the row given a label
    '''
    inverted = {}
    for i, row in enumerate(rows):
        labels = json.loads(row[1])
        if type(labels) is list:
            # detection dataset
            curr_unique_labels = set([l['class'] for l in labels])
        else:
            assert type(labels) is int
            curr_unique_labels = [label_map[labels]]
        for l in curr_unique_labels:
            assert type(l) == str or type(l) == unicode
            if l not in inverted:
                inverted[l] = [i]
            else:
                inverted[l].append(i)
    def gen_rows():
        for label in inverted:
            assert label in label_map
        for label in label_map:
            i = inverted[label] if label in inverted else []
            yield label, ' '.join(map(str, i))
    tsv_writer(gen_rows(), inverted_label_file)


def is_verified_rect(rect):

    if 'uhrs' in rect:
        judge_result = rect['uhrs']
        assert judge_result.get('1', 0) >= judge_result.get('2', 0)
        return True

    if 'class' not in rect or 'rect' not in rect:
        return False

    if 'uhrs_confirm' in rect:
        assert rect['uhrs_confirm'] > 0
        return True

    if 'conf' in rect and rect['conf'] < 1:
        return False

    if 'merge_from' in rect:
        return all(is_verified_rect(r) for r in rect['merge_from'])

    return True


def create_inverted_list(rows):
    inverted = {}
    inverted_with_bb = {}
    inverted_no_bb = {}
    inverted_with_bb_verified = {}
    inverted_with_bb_noverified = {}
    logging.info('creating inverted')
    for i, row in tqdm(enumerate(rows), mininterval=2):
        labels = json.loads(row[1])
        if type(labels) is list:
            # detection dataset
            curr_unique_labels = set([l['class'] for l in labels])
            curr_unique_with_bb_labels = set([l['class'] for l in labels
                if 'rect' in l and any(x != 0 for x in l['rect'])])
            curr_unique_no_bb_labels = set([l['class'] for l in labels
                if 'rect' not in l or all(x == 0 for x in l['rect'])])
            curr_unique_with_bb_verified_labels = set([l['class'] for l in labels
                if 'rect' in l and any(x != 0 for x in l['rect']) and is_verified_rect(l)])
            curr_unique_with_bb_noverified_labels = set([l['class'] for l in labels
                if 'rect' in l and any(x != 0 for x in l['rect']) and not is_verified_rect(l)])
        else:
            assert type(labels) is int
            curr_unique_labels = [str(labels)]
            curr_unique_with_bb_labels = []
            curr_unique_no_bb_labels = curr_unique_labels
            curr_unique_with_bb_verified_labels = set()
            curr_unique_with_bb_noverified_labels = set()
        def update(unique_labels, inv):
            for l in unique_labels:
                assert type(l) == str
                if l not in inv:
                    inv[l] = [i]
                else:
                    inv[l].append(i)
        update(curr_unique_labels, inverted)
        update(curr_unique_with_bb_labels, inverted_with_bb)
        update(curr_unique_no_bb_labels, inverted_no_bb)
        update(curr_unique_with_bb_verified_labels, inverted_with_bb_verified)
        update(curr_unique_with_bb_noverified_labels, inverted_with_bb_noverified)
    return {'inverted.label': inverted,
            'inverted.label.with_bb': inverted_with_bb,
            'inverted.label.no_bb': inverted_no_bb,
            'inverted.label.with_bb.verified': inverted_with_bb_verified,
            'inverted.label.with_bb.noverified': inverted_with_bb_noverified}


def tsv_shuffle_reader(tsv_file):
    logging.warn('deprecated: using TSVFile to randomly seek')
    lineidx_file = op.splitext(tsv_file)[0] + '.lineidx'
    lineidx = load_list_file(lineidx_file)
    random.shuffle(lineidx)
    with open(tsv_file, 'r') as fp:
        for l in lineidx:
            fp.seek(int(float(l)))
            yield [x.strip() for x in fp.readline().split('\t')]


def load_labelmap(data):
    dataset = TSVDataset(data)
    return dataset.load_labelmap()


def get_caption_data_info(name):
    dataset = TSVDataset(name)
    splits = get_default_splits()
    from collections import defaultdict
    split_to_versions = defaultdict(list)
    for split in splits:
        v = 0
        while True:
            if not dataset.has(split, 'caption', v):
                break
            split_to_versions[split].append(v)
            v = v + 1
    return split_to_versions


def get_all_data_info():
    names = os.listdir('./data')
    name_splits_labels = []
    names.sort(key=lambda n: n.lower())
    for name in names:
        dataset = TSVDataset(name)
        if not op.isfile(dataset.get_labelmap_file()):
            continue
        labels = dataset.load_labelmap()
        valid_splits = []
        if len(dataset.get_train_tsvs()) > 0:
            valid_splits.append('train')
        for split in ['trainval', 'test']:
            if not op.isfile(dataset.get_data(split)):
                continue
            valid_splits.append(split)
        name_splits_labels.append((name, valid_splits, labels))
    return name_splits_labels


def load_labels(file_name):
    rows = tsv_reader(file_name)
    key_to_rects = {}
    key_to_idx = {}
    for i, row in enumerate(rows):
        key = row[0]
        rects = json.loads(row[1])
        #assert key not in labels, '{}-{}'.format(file_name, key)
        key_to_rects[key] = rects
        key_to_idx[key] = i
    return key_to_rects, key_to_idx


def load_list_file(fname):
    # prefer this than qd.qd_common.load_list_file
    with exclusive_open_to_read(fname) as fp:
        lines = fp.readlines()
    result = [line.strip() for line in lines]
    if len(result) > 0 and result[-1] == '':
        result = result[:-1]
    return result


def convert_data_to_yaml(
    data, split, yaml,
    is_train=True,
    label=None,
    feature=None,
    qd_format=False,
    label_version=None,
    feature_version=None):
    # used for captioning-related scripts
    if qd_format:
        info = {
            'feature': feature if feature is not None else {
                'data': data,
                'split': split,
                't': 'feature',
                'version': feature_version,
            },
            'hw': {'data': data, 'split': split, 't': 'hw'},
            'img': {'data': data, 'split': split},
            'label': label if label is not None else {
                'data': data,
                'split': split,
                't': 'label',
                'version': label_version,
            },
            'caption': {'data': data, 'split': split, 't': 'hw'},
            'composite': False,
            'qd_format': True,
        }
    else:
        assert label is None and feature is None
        # will be deprecated
        from src.qd.tsv_io import TSVDataset
        yaml_folder = op.dirname(yaml)
        dataset = TSVDataset(data)
        if not op.isfile(dataset.get_data(split + 'X')):
            # we prefer to use the composite
            info = {
                'feature': op.relpath(dataset.get_data('train', 'feature', version=feature_version), yaml_folder),
                'label': op.relpath(dataset.get_data(split, 'label', version=label_version), yaml_folder),
                'hw': op.relpath(dataset.get_data(split, 'hw'), yaml_folder),
                'img': op.relpath(dataset.get_data(split), yaml_folder),
                'caption': op.relpath(dataset.get_data(split, 'caption'), yaml_folder),
                'composite': False,
            }
        else:
            def get_rel_path(p):
                return op.relpath(op.realpath(p), op.realpath(yaml_folder))
            splitX = split + 'X'
            from src.tools.common import load_list_file
            info = {
                'feature': list(map(get_rel_path, load_list_file(dataset.get_data(splitX, 'feature', version=feature_version)))),
                'label': list(map(get_rel_path, load_list_file(dataset.get_data( splitX, 'label', version=label_version)))),
                'hw': list(map(get_rel_path, load_list_file(dataset.get_data(splitX, 'hw')))),
                'img': list(map(get_rel_path, load_list_file(dataset.get_data(splitX)))),
                'caption': list(map(get_rel_path, load_list_file(dataset.get_data(splitX, 'caption')))),
                'composite': True,
            }
            if is_train:
                caption_linelist = dataset.get_data(split, 'caption_linelist')
                assert op.isfile(caption_linelist)
                info['caption_linelist'] = caption_linelist
            else:
                caption_linelist = dataset.get_data(split, 'caption_linelist_test')
                if not op.isfile(caption_linelist):
                    from src.qd.tsv_io import tsv_reader
                    tsv_writer(((a, b, 0) for a, b in
                                tsv_reader(dataset.get_shuffle_file(split))),
                               caption_linelist)
                info['caption_linelist'] = caption_linelist
    from src.tools.common import write_to_yaml_file
    write_to_yaml_file(info, yaml)


