from torchvision.transforms import transforms
from .dataset import (
    ImageIdxTSVDataset,
    CaptionIdxTSVDataset,
    DatasetPlusTransform,
)
from .transform import (
    LoadLabel,
    LoadHW,
    LoadImage,
    LoadCaption,
    IdentifyTextAB,
)

from torch.utils.data.dataloader import default_collate
import torch


def collate_fn(batch):
    # this function is designed to support any customized type and to be compatible
    # with the default collate function
    ele = batch[0]

    # to handle the rect returning, check the type
    if isinstance(ele, dict):
        re = {}
        for key in ele:
            if type(ele[key]) == torch.Tensor or type(ele[key]) == int:
                re[key] = collate_fn([d[key] for d in batch])
            else:
                re[key] = [d[key] if key in d else [] for d in batch]
        del ele, batch
        return re

    else:
        if all(isinstance(b, torch.Tensor) for b in batch) and len(batch) > 0:
            if not all(b.shape == batch[0].shape for b in batch[1:]):
                assert all(len(b.shape) == len(batch[0].shape) for b in batch[1:])
                shape = torch.tensor([b.shape for b in batch])
                max_shape = tuple(shape.max(dim=0)[0].tolist())
                batch2 = []
                for b in batch:
                    if any(c < m for c, m in zip(b.shape, max_shape)):
                        b2 = torch.zeros(max_shape, dtype=b.dtype, device=b.device)
                        if b.dim() == 1:
                            b2[:b.shape[0]] = b
                        elif b.dim() == 2:
                            b2[:b.shape[0], :b.shape[1]] = b
                        else:
                            raise NotImplementedError
                        b = b2
                    batch2.append(b)
                batch = batch2
        return default_collate(batch)


# def build_caption_dataset(
#     data, split, is_train, on_memory,
#     feature_version,
#     img_feature_dim,
#     max_img_seq_len,
#     feat_sort_by_conf,
#     add_od_labels,
#     label_version,
#     od_label_conf,
#     label_sort_by_conf,
#     unique_labels_on,
#     qa2caption,
#     sep_token,
#     tensorizer,
#     # the following parameters are only used in the case of e2e where feature
#     # is not pre-calculated
#     input_crop_size=224,
#     test_resize_size=None,
#     input_small_scale=None,
#     pad_to_max=True,
# ):
#     if is_train:
#
#         dataset = CaptionIdxTSVDataset(
#             data=data,
#             split=split,
#             caption_version=None,
#         )
#     else:
#         dataset = ImageIdxTSVDataset(
#             data=data,
#             split=split,
#         )
#     all_trans = []
#     cache_policy = 'memory' if on_memory else None
#     hw_loader = LoadHW(
#         data=data, split=split,
#         cache_policy=cache_policy,
#     )
#     all_trans.append(hw_loader)
#
#     load_feature = max_img_seq_len > 0
#
#     # by default, we don't load any object detector features
#     assert not load_feature
#
#     # load image and we will extract the features online. This is mainly
#     # used for end-to-end training or inference.
#     image_loader = LoadImage(data, split)
#     if is_train:
#         from src.qd.data_layer.transform import get_inception_train_transform
#         image_transform = get_inception_train_transform(
#             bgr2rgb=True,
#             crop_size=input_crop_size,
#             small_scale=input_small_scale,
#         )
#     else:
#         if test_resize_size is None:
#             resize_size = 256 * input_crop_size // 224
#         from src.qd.data_layer.transform import get_inception_test_transform
#         image_transform = get_inception_test_transform(
#             bgr2rgb=True,
#             resize_size=resize_size,
#             crop_size=input_crop_size,
#         )
#     from src.qd.data_layer.transform import ImageTransform2Dict
#     image_transform = ImageTransform2Dict(image_transform)
#     feature_loader = transforms.Compose([
#         image_loader,
#         image_transform,
#     ])
#
#     all_trans.append(feature_loader)
#
#     if is_train:
#         caption_loader = LoadCaption(
#             data=data, split=split, version=None,
#             cache_policy=cache_policy,
#         )
#         all_trans.append(caption_loader)
#
#     if add_od_labels:
#         label_loader = LoadLabel(
#             data=data, split=split,
#             version=label_version)
#         all_trans.append(label_loader)
#
#     text_ab = IdentifyTextAB(
#         add_od_labels, od_label_conf, label_sort_by_conf,
#         unique_labels_on, qa2caption, sep_token,
#     )
#     all_trans.append(text_ab)
#
#     trans_tensorizer = CaptionTensorizer(
#         tensorizer,
#         with_img_feats=load_feature,
#         pad_to_max=pad_to_max,
#     )
#     all_trans.append(trans_tensorizer)
#
#     useless_keys = [
#             'idx',
#             'idx_img',
#             'idx_cap',
#             'dataset',
#             'label',
#             'caption',
#             'text_ab_type',
#             'text_a',
#             'text_b',
#             'width',
#             'height',
#             'text_changed',
#             'text_a_or_b_changed',
#             'img_feat',
#             'max_seq_a_len',
#             'feats_conf',
#             'feats_class',
#             'vocab_size',
#             'feats_class_token_ids',
#             'feats_class_tokens',
#     ]
#     all_trans.extend([
#         RemoveUselessKeys(useless_keys),
#         RenameKey({'segment_ids': 'token_type_ids'}),
#     ])
#     all_trans = transforms.Compose(all_trans)
#     dataset = DatasetPlusTransform(dataset, all_trans)
#     return dataset
