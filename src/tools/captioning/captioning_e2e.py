import os
import os.path as op
import requests
import base64
import time
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
import math

from src.qd.mask.data.transforms.build import build_transforms_mmask as build_transforms
from src.qd.mask.data.datasets.utils.load_files import load_labelmap_file
from src.qd.mask.data.datasets.caption_tensorizer import CaptionTensorizer
from src.tools.layers.bert import BertTokenizer, BertConfig, BertForImageCaptioning


def download_image(url):
    """Download raw image from url
    """
    r = requests.get(url, stream=True, timeout=0.5)
    assert r.status_code == 200, "Invalid URL"
    return r.content

class ObjectDetector(nn.Module):
    # Wrap object detector and return object information like box and features
    def __init__(self, cfg):
        super(ObjectDetector, self).__init__()
        self.cfg = cfg
        from mmask.modeling.detector import build_detection_model
        self.model = build_detection_model(cfg)
        from mmask.utils.checkpoint import DetectronCheckpointer
        checkpointer = DetectronCheckpointer(cfg, self.model, is_test=True)
        checkpointer.load(cfg.MODEL.WEIGHT)
        self.transforms = build_transforms(cfg, is_train=False)
        self.labelmap = load_labelmap_file(op.join(cfg.DATA_DIR, cfg.DATASETS.LABELMAP_FILE))
        self.labelmap_invert = {self.labelmap[key] + 1 : key for key in self.labelmap}

    def pre_processing(self, img_file):
        if isinstance(img_file, dict):
            raw = img_file.get('b64')
            if raw:
                raw = base64.b64decode(raw.encode('utf-8'))
            else:
                raw = download_image(img_file['url'])
            img_file = raw
        if isinstance(img_file, bytes):
            # Raw bytes
            nparr = np.frombuffer(img_file, np.uint8)
            cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            del nparr
        else:
            # Path to a file
            assert op.isfile(img_file)
            cv2_img = cv2.imread(img_file)
            del img_file
        assert cv2_img is not None and len(cv2_img) > 1, "Invalid Image"
        if self.cfg.INPUT.CV2_OUTPUT:
            img = cv2_img.astype(np.float32, copy=True)
            img_size = img.shape[:2][::-1]
        else:
            img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
            img_size = img.size
        new_img = self.transforms(img)[0]
        new_img_size = new_img.shape[1:]
        scale = math.sqrt(float(new_img_size[0] * new_img_size[1]) / (img_size[0] * img_size[1]))
        return new_img, scale

    def forward(self, image, scale):
        image = (image,)
        scale = [(scale, )]
        with torch.no_grad():
            output = self.model(image, scales=scale)
        return output


class BertCaptioning(nn.Module):
    # Wrap Bert for image captioning and return a caption.
    def __init__(self, args):
        super(BertCaptioning, self).__init__()
        self.inference_mode = args.mode
        if self.inference_mode in ['prod']:
            assert args.num_beams == 1
            args.output_hidden_states = True
        elif self.inference_mode in ['prod_no_hidden']:
            assert args.num_beams == 1
            args.output_hidden_states = False
        else:
            assert self.inference_mode=="default", 'unknown inference mode: {}'.format(
                    self.inference_mode)

        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        self.config = BertConfig.from_pretrained(checkpoint)
        self.config.output_hidden_states = args.output_hidden_states
        self.tokenizer = BertTokenizer.from_pretrained(checkpoint)
        self.tensorizer = CaptionTensorizer(self.tokenizer, args.max_img_seq_length, 
                args.max_seq_length, args.max_seq_a_length, is_train=False)
        self.model = BertForImageCaptioning.from_pretrained(checkpoint, config=self.config)
        self.args = args
        self.convert_special_tokens_to_ids()
        self.input_params = {
            'bos_token_id': self.cls_token_id, 
            'eos_token_ids': [self.sep_token_id, self.pad_token_id], 
            'mask_token_id': self.mask_token_id,
            'add_od_labels': args.add_od_labels, 
            'od_labels_start_posid': args.max_seq_a_length,
            'max_length': args.max_gen_length, 
        }
        self.extra_params = {
            'pad_token_id': self.pad_token_id,
            'is_decode': True,
            'do_sample': False, 
            'num_beams': args.num_beams,
            "temperature": args.temperature, 
            "top_k": self.args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "length_penalty": args.length_penalty,
            "num_return_sequences": args.num_return_sequences,
            "num_keep_best": args.num_keep_best,
        }

    def convert_special_tokens_to_ids(self):
        self.cls_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        self.period_token_id = self.tokenizer.convert_tokens_to_ids('.')

    def pre_processing(self, img_features, labels=None):
        caption = ''
        if self.args.add_od_labels:
            if type(labels) == list:
                labels = ' '.join(labels)
        if self.inference_mode in ['prod', 'prod_no_hidden']:
            return self.tensorizer.prod_tensorize_example(caption, img_features, labels)
        return self.tensorizer.tensorize_example(caption, img_features, labels)

    def forward(self, *args, **kwargs):
        if self.inference_mode in ['prod', 'prod_no_hidden']:
            return self.prod_forward(*args, **kwargs)
        return self._forward(*args, **kwargs)

    def _forward(self, input_ids, attention_mask, token_type_ids, img_feats, masked_pos):
        # for back compatibility
        inputs = {
            'input_ids': input_ids, 
            'attention_mask': attention_mask,
            'token_type_ids':token_type_ids,
            'img_feats': img_feats,
            'masked_pos': masked_pos,
        }
        inputs.update(self.extra_params)
        inputs.update(self.input_params)
        with torch.no_grad():
            outputs = self.model(**inputs)
        caps = outputs[0][0]  # batch_size * num_keep_best * max_len
        confs = torch.exp(outputs[1][0])
        if torch._C._get_tracing_state():
            return caps, confs
        result = []
        for cap, conf in zip(caps, confs):
            cap = self.tokenizer.decode(cap.tolist(), skip_special_tokens=True)
            result.append({'caption': cap, 'conf': conf.item()})
        return result

    def prod_forward(self, od_label_ids, img_feats):
        inputs = {
            'inference_mode': self.inference_mode,
            'od_label_ids': od_label_ids,
            'img_feats': img_feats,
        }
        inputs.update(self.input_params)

        with torch.no_grad():
            outputs = self.model(**inputs)
        caps = outputs[0]   # batch_size max_len
        confs = torch.exp(outputs[1])
        if torch._C._get_tracing_state():
            return caps, confs
        result = []
        for cap, conf in zip(caps, confs):
            cap = self.tokenizer.decode(cap.tolist(), skip_special_tokens=True)
            result.append({'caption': cap, 'conf': conf.item()})
        return result


def create_image_features(bbox, region_features, image):
    # to keep it consistent with VLP pre-training
    if image.ndim == 3:
        image_height, image_width = image.shape[1:]
    else:
        assert image.ndim == 4
        image_height, image_width  = image.shape[2:]
    box_width = bbox[:, 2] - bbox[:, 0]
    box_height = bbox[:, 3] - bbox[:, 1]
    scaled_width = box_width / image_width
    scaled_height = box_height / image_height
    scaled_x = bbox[:, 0] / image_width
    scaled_y = bbox[:, 1] / image_height
    spatial_features = torch.stack((scaled_x, scaled_y, scaled_x + scaled_width, 
            scaled_y + scaled_height, scaled_width, scaled_height), dim=1)
    features = torch.cat((region_features, spatial_features), 1)
    return features

def convert_prediction_results_base(bbox, region_features, labels, image, labelmap):
    features = create_image_features(bbox, region_features, image) 
    labels = [labelmap[l] for l in labels]
    return features, labels


def convert_prediction_results(predictions, image, labelmap):
    return convert_prediction_results_base(
        predictions.bbox, 
        predictions.get_field('box_features'), 
        predictions.get_field('labels').tolist(),
        image, labelmap
    )

def predict(detector, captioner, image_file, bert_device=torch.device('cpu'), cpu_device=torch.device('cpu')):
    tic = time.time()
    image, scale = detector.pre_processing(image_file)
    print("pre_processing time: {}".format(time.time() - tic))
    tic = time.time()
    image.to(cpu_device)
    predictions = detector(image, scale)
    print("OD inference time: {}".format(time.time() - tic))

    predictions = predictions[0].to(cpu_device)
    tic = time.time()
    img_feats, od_labels = convert_prediction_results(predictions, image, detector.labelmap_invert)
    batch = captioner.pre_processing(img_feats, od_labels)
    batch = tuple([torch.unsqueeze(t, 0).to(bert_device) for t in batch])
    result = captioner(*batch)
    print("BERT inference time: {}".format(time.time() - tic))
    return result
