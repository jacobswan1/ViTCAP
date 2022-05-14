import json
import numpy as np
from .dataset import TSVSplitProperty
from torchvision.transforms import transforms
from src.tools.common import dict_update_path_value
from src.tools.common import img_from_base64, pilimg_from_base64
from src.tools.common import dict_has_path, dict_get_path_value, dict_remove_path


class Keys:
    # json.loads(label_row_second_part)
    # can be loaded by LoadLabel
    label = 'label'
    # ConvertToDomainClassIndex
    class_idx = 'label_idx'
    pred = 'pred'


class ImageTransform2Dict(object):
    def __init__(self, image_transform):
        self.image_transform = image_transform

    def __call__(self, dict_data):
        out = dict(dict_data.items())
        out['image'] = self.image_transform(dict_data['image'])
        return out

    def __repr__(self):
        return 'ImageTransform2Dict(image_transform={})'.format(
            self.image_transform,
        )

def get_default_mean():
    return [0.485, 0.456, 0.406]


def get_default_std():
    return [0.229, 0.224, 0.225]


def get_data_normalize():
    normalize = transforms.Normalize(mean=get_default_mean(),
                                     std=get_default_std())
    return normalize


class BGR2RGB(object):
    def __call__(self, im):
        return im[:, :, [2, 1, 0]]


def get_inception_train_transform(bgr2rgb=False,
                                  crop_size=224,
                                  small_scale=None,
                                  normalize=None,
                                  backend='cv'
                                  ):
    normalize = normalize or get_data_normalize()
    totensor = transforms.ToTensor()
    color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
    all_trans = []
    if backend == 'cv':
        if bgr2rgb:
            all_trans.append(BGR2RGB())
        all_trans.append(transforms.ToPILImage())
    if small_scale is None:
        small_scale = 0.08
    scale = (small_scale, 1.)
    all_trans.append(transforms.RandomResizedCrop(crop_size, scale=scale))
    if backend == 'pil':
        if bgr2rgb:
            all_trans.append(
                lambda image: image.convert("RGB"),
            )
    all_trans.extend([
        color_jitter,
        transforms.RandomHorizontalFlip(),
        totensor,
        normalize,])
    data_augmentation = transforms.Compose(all_trans)
    return data_augmentation


class LoadLabel(object):
    def __init__(self, data, split, version, out_key=None):
        from .dataset import TSVSplitProperty
        self.label_tsv = TSVSplitProperty(
            data, split, 'label', version=version)
        self.out_key = out_key if out_key else Keys.label

    def __repr__(self):
        return 'LoadLabel(data={}, split={}, version={})'.format(
            self.label_tsv.data, self.label_tsv.split, self.label_tsv.version,
        )

    def __call__(self, data):
        idx_img = data['idx_img']
        key, str_label = self.label_tsv[idx_img]
        rects = json.loads(str_label)

        assert self.out_key not in data
        data[self.out_key] = rects
        return data


class LoadImage(object):
    def __init__(self, data, split,
                 add_key=False, backend='cv', hold_buffer=0, save_original=False):
        self.tsv = TSVSplitProperty(data, split, hold_buffer=hold_buffer)
        from src.tools.common import print_frame_info
        print_frame_info()
        self.add_key = add_key
        self.bk = backend
        assert backend in ['cv', 'pil']
        self.save_original = save_original

    def __call__(self, data):
        r = self.tsv[data['idx_img']]
        # for the image, we do not check the key as the key could be different,
        # which is by design, for some kind of composite dataset.
        # key = r[0]
        str_im = r[-1]
        if self.bk == 'cv':
            img = img_from_base64(str_im)
        else:
            img = pilimg_from_base64(str_im)
        assert 'image' not in data
        data['image'] = img

        if self.save_original:
            data['ori_image'] = np.array(pilimg_from_base64(str_im).resize((384, 384)))
        if self.add_key:
            data['key'] = r[0]
        if 'future_idx_img' in data:
            self.tsv.prepare(data['future_idx_img'])
        return data


class LoadHW(object):
    # idx_img -> height/width
    def __init__(self, data, split, cache_policy=None):
        self.tsv = TSVSplitProperty(data,
                                    split,
                                    'hw',
                                    cache_policy=cache_policy)

    def __call__(self, data):
        idx_img = data['idx_img']
        key, str_hw = self.tsv[idx_img]
        if 'key' not in data:
            data['key'] = key

        # FIXME: this line check failed sometime, fix later.
        assert key == data['key']
        try:
            hw_info = json.loads(str_hw)
            if isinstance(hw_info, list):
                assert len(hw_info) == 1
                hw_info = hw_info[0]
            data.update(hw_info)
        except:
            h, w = map(int, str_hw.split(' '))
            data['height'] = h
            data['width'] = w
        return data


class LoadCaption(object):
    def __init__(self, data, split, version, cache_policy=None):
        super().__init__()
        self.tsv = TSVSplitProperty(data,
                                    split,
                                    'caption',
                                    version=version,
                                    cache_policy=cache_policy)

    def __repr__(self):
        return 'LoadCaption(tsv={})'.format(self.tsv)

    def __call__(self, data):
        idx_img = data['idx_img']
        key, str_cap = self.tsv[idx_img]
        # assert key == data['key']
        caps = json.loads(str_cap)
        idx_cap = data['idx_cap']
        cap = caps[idx_cap]
        data['caption'] = cap
        return data

    def get_captions_by_key(self, img_idx):
        # get a list of captions for image (by key)
        # img_idx = self.key2index[key]
        cap_info = json.loads(self.tsv[img_idx][1])
        return [c['caption'] for c in cap_info]


class IdentifyTextAB(object):
    # if it is captioning dataset, captioning description is text a; optionally
    # label str is text b; if it is qa, we have several options. by default,
    # question is text a and answer is text b
    def __init__(self, add_od_labels, od_label_conf, label_sort_by_conf,
                 unique_labels_on, qa2caption=None, sep_token=None):
        super().__init__()
        self.add_od_labels = add_od_labels
        self.od_label_conf = od_label_conf
        self.sort_by_conf = label_sort_by_conf
        self.unique_labels_on = unique_labels_on
        self.qa2caption = qa2caption
        self.sep_token = sep_token

    def __call__(self, data):
        # currently, this function is used to load information for current
        # instance and the negative instance. If we'd like to have different
        # behaviors for the negative instance, we can add options to this
        # function
        if self.add_od_labels:
            label_info = data['label']
            for lab in label_info:
                if 'conf' not in lab:
                    lab['conf'] = 1.0
            if self.od_label_conf is None:
                self.od_label_conf = 0.2
            if len(label_info) > 0 and self.od_label_conf > 0 and 'conf' in label_info[0]:
                # select labels based on confidence
                label_info = [l for l in label_info if l['conf'] >= self.od_label_conf]
            if self.sort_by_conf:
                label_info = sorted(label_info, key = lambda x : -x['conf'])
            if self.unique_labels_on:
                # keep the order so it is deterministic
                label_list = []
                for lab in label_info:
                    if lab['class'].lower() not in label_list:
                        label_list.append(lab['class'].lower())
                od_labels = ' '.join(label_list)
            else:
                od_labels = ' '.join([l['class'].lower() for l in label_info])
        else:
            od_labels = ''
        caption_dict = data.get('caption')
        if caption_dict is None:
            # in captioning downstream task/test phase, there is no need to
            # load caption and caption is also not used. Thus, we will set
            # caption = ''
            caption = ''
            data['text_ab_type'] = 'empty_label'
        elif 'caption' in caption_dict:
            caption = caption_dict['caption']
            data['text_ab_type'] = 'cap_label'
        else:
            raise NotImplementedError
        data['text_a'] = caption
        data['text_b'] = od_labels
        return data


class RemoveUselessKeys(object):
    def __init__(self, keys=None):
        if keys is None:
            self.keys = ['io_dataset']
        else:
            self.keys = keys

    def __call__(self, sample):
        for k in self.keys:
            if dict_has_path(sample, k):
                dict_remove_path(sample, k)
        return sample


class RenameKey(object):
    def __init__(self, ft=None):
        # from to
        self.ft = ft

    def __call__(self, data):
        if self.ft is None:
            return data
        for k, k1 in self.ft.items():
            # we should not fall to the situation where some data has some key
            # and some data has not some key. We should either have a key or
            # not for all data consistently. Thus, for re-naming, we should not
            # to check whether it has or not. it should always have that key.
            # otherwise, we should not specify it.
            if dict_has_path(data, k):
                v = dict_get_path_value(data, k)
                dict_update_path_value(data, k1, v)
                dict_remove_path(data, k)
        return data
