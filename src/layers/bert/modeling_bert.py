# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import math
import json
import torch
import logging
from io import open
from torch import nn
from .activations import ACT2FN
import torch.nn.functional as F
from .modeling_utils import (PretrainedConfig, PreTrainedModel, prune_linear_layer,
                             add_start_docstrings)

logger = logging.getLogger()

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
}

BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-config.json",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-config.json",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-config.json",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-config.json",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-config.json",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-config.json",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-config.json",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-config.json",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-config.json",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-config.json",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-config.json",
}


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


class BertConfig(PretrainedConfig):
    r"""
        :class:`~pytorch_transformers.BertConfig` is the configuration class to store the configuration of a
        `BertModel`.


        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
    """
    pretrained_config_archive_map = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size_or_config_json_file=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 **kwargs):
        super(BertConfig, self).__init__(**kwargs)
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")


LayerNormClass = torch.nn.LayerNorm


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNormClass(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertImgEmbeddings(nn.Module):
    """ BERT Language - Image Embedding
    Construct the embeddings from word & Images, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertImgEmbeddings, self).__init__()
        self.img_dim = 565

        self.img_embeddings = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNormClass(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        img_embeddings = self.img_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = img_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        if torch._C._get_tracing_state():
            # exporter is not smart enough to detect dynamic size for some paths
            x = x.view(x.shape[0], -1, self.num_attention_heads, self.attention_head_size)
        else:
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None,
            history_state=None):
        if history_state is not None:
            x_states = torch.cat([history_state, hidden_states], dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNormClass(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        for head in heads:
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads

    def forward(self, input_tensor, attention_mask, head_mask=None,
            history_state=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask,
                history_state)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNormClass(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None,
                history_state=None):
        attention_outputs = self.attention(hidden_states, attention_mask,
                head_mask, history_state)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class TIMMVitSplitEncoder(nn.Module):
    # avoid to use TimmVitEncoder
    def __init__(self, config):
        super().__init__()
        # logging.info(config)
        logging.info('TIMM Split image encoder load from pre-trained: {}'.format(config.pretrained))

        self.config = config
        from src.pytorch_image_models import timm
        model = timm.create_model(config.net, pretrained=True,)

        self.blocks = model.blocks

        # By default just use last 4 blocks for tag classification, but seems it's a hyper-param that different layers
        #  lead better results. Didn't explore more here for simplicity.
        self.split_blocks = getattr(config, 'split_blocks', 4)
        self.tag_blocks = timm.create_model(config.net, pretrained=True,).blocks[-self.split_blocks:]

    def forward(self, hidden_states, attention_mask, head_mask=None, encoder_history_states=None):
        assert all(m is None for m in head_mask), 'not supported'
        assert encoder_history_states is None, 'not supported'

        all_hidden_states = ()
        tag_hidden_states = None
        for layer_idx, blk in enumerate(self.blocks):

            if layer_idx == len(self.blocks) - self.split_blocks:
                tag_hidden_states = hidden_states

            all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = blk(hidden_states, attention_mask)

        # Tagger Split
        for layer_idx, blk in enumerate(self.tag_blocks):
            tag_hidden_states = blk(tag_hidden_states, attention_mask)

        outputs = (hidden_states, tag_hidden_states)

        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        # self.output_attentions = config.output_attn
        self.output_attentions = config.output_attentions

        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None,
                encoder_history_states=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            history_state = None if encoder_history_states is None else encoder_history_states[i]
            layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i],
                    history_state)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # outputs, (hidden states), (attentions)


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = LayerNormClass(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def __init__(self, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__(*inputs, **kwargs)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNormClass):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


BERT_START_DOCSTRING = r"""    The BERT model was proposed in
    `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_
    by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. It's a bidirectional transformer
    pre-trained using a combination of masked language modeling objective and next sentence prediction
    on a large corpus comprising the Toronto Book Corpus and Wikipedia.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`:
        https://arxiv.org/abs/1810.04805

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~pytorch_transformers.BertConfig`): Model configuration class with all the parameters of the model.
"""

BERT_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, BERT input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]``

                ``token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``

                ``token_type_ids:   0   0   0   0  0     0   0``

            Indices can be obtained using :class:`pytorch_transformers.BertTokenizer`.
            See :func:`pytorch_transformers.PreTrainedTokenizer.encode` and
            :func:`pytorch_transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1[``.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            (see `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_ for more details).
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""


class BertCaptioningHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertCaptioningLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.label_smoothing = getattr(config, 'label_smoothing', 0)
        self.drop_worst_ratio = getattr(config, 'drop_worst_ratio', 0)
        self.drop_worst_after = getattr(config, 'drop_worst_after', 0)
        self.log_soft = nn.LogSoftmax(dim=1)
        self.kl = nn.KLDivLoss(reduction='none')
        self.iter = 0

    def forward(self, logits, target):
        self.iter += 1
        if logits.numel() == 0:
            # this happens when we ignore masked tokens for unmatched
            # image-text pairs, in which we may ignore all tokens
            return torch.tensor(0., requires_grad=True, device=logits.device)
        eps = self.label_smoothing
        n_class = logits.size(1)
        one_hot = torch.zeros_like(logits).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = self.log_soft(logits)
        loss = self.kl(log_prb, one_hot).sum(1)

        if self.drop_worst_ratio > 0 and self.iter > self.drop_worst_after:
            loss, _ = torch.topk(loss,
                    k=int(loss.shape[0] * (1-self.drop_worst_ratio)),
                    largest=False)
        loss = loss.mean()

        return loss


@add_start_docstrings("""Bert Model transformer for image captioning""",
    BERT_START_DOCSTRING)
class ViTCAP(BertPreTrainedModel):
    r"""
    Bert for Image Captioning.
    """
    def __init__(self, config):
        super(ViTCAP, self).__init__(config)
        self.config = config

        self.bert = ViTSplitCLSEmbModel(config)

        # self.tag_tokenizer = BertEmbeddings(config)
        self.cls = BertCaptioningHeads(config)

        # hinge the weight of tag classifier and caption classifier -> this seems not producing better results
        # self._tie_or_clone_weights(self.tag_tokenizer.word_embeddings, self.cls.predictions.decoder)

        self.loss = BertCaptioningLoss(config)

        if getattr(config, 'loss', None) == 'focal':
            from src.layers.loss import FocalLossWithLogitsNegLoss
            self.tag_loss = FocalLossWithLogitsNegLoss(alpha=0.5, gamma=1)
        else:
            self.tag_loss = torch.nn.BCEWithLogitsLoss()

        # self.bert.apply(self.init_weights)
        self.cls.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        if getattr(self.config, 'tie_tag_weights', False) and self.config.tie_weights:
            self._tie_or_clone_weights(self.bert.tag_logit.predictions.decoder,
                                       self.bert.embeddings.word_embeddings)

        if hasattr(self.config, 'tie_weights') and self.config.tie_weights:
            self._tie_or_clone_weights(self.cls.predictions.decoder,
                                       self.bert.embeddings.word_embeddings)
        freeze = False
        if hasattr(self.config, 'freeze_embedding'):
            freeze = self.config.freeze_embedding
        self.bert.embeddings.word_embeddings.weight.requires_grad = not freeze

    def forward(self, *args, **kwargs):
        is_decode = kwargs.get('is_decode', False)
        inference_mode = kwargs.get('inference_mode', '')
        if inference_mode:
            kwargs.pop('inference_mode')
            if inference_mode == 'prod':
                return self.prod_generate(*args, **kwargs)
            if inference_mode == 'prod_no_hidden':
                return self.prod_no_hidden_generate(*args, **kwargs)
            assert False, 'unknown inference_mode: {}'.format(inference_mode)
        if is_decode:
            return self.generate(*args, **kwargs)
        else:
            return self.encode_forward(*args, **kwargs)

    def encode_forward(self, input_ids, img_feats, attention_mask, label=None,
                       masked_pos=None, masked_ids=None, token_type_ids=None,
                       position_ids=None, head_mask=None, is_training=True,
                       encoder_history_states=None, return_dict=False, matched=None,
                       return_hidden=False, gen_tag_ratio=None,):

        # Forward to compute the tag-gradient
        outputs, tag_logit = self.bert(input_ids,
                                       img_feats=img_feats,
                                       label=label,
                                       attention_mask=attention_mask,
                                       position_ids=position_ids,
                                       token_type_ids=token_type_ids,
                                       head_mask=head_mask,
                                       encoder_history_states=encoder_history_states,
                                       cls_emb=self.cls.predictions.decoder,
                                       gen_tag_ratio=gen_tag_ratio,
                                       )

        all_sequence_output = outputs[0]
        if is_training and not torch._C._get_tracing_state():
            sequence_output = outputs[0][:, :masked_pos.shape[-1], :]
            # num_masks_in_batch * hidden_size
            if matched is not None:
                # if it is not matched, we make all the masked pos = 0 to ignore the
                # loss do it as in place as we will use it to calculate the acc somewhere
                masked_pos.requires_grad = False
                masked_pos[matched.logical_not()] = 0
                # make it as padded id and we will remove
                masked_ids.requires_grad = False
                masked_ids[matched.logical_not()] = 0

            sequence_output_masked = sequence_output[masked_pos==1, :]
            class_logits = self.cls(sequence_output_masked)
            masked_ids = masked_ids[masked_ids != 0]   # remove padding masks
            masked_loss = self.loss(class_logits.float(), masked_ids)

            # Tagger loss
            tag_loss = self.tag_loss(tag_logit, label)
            if getattr(self.config, 'loss', None) == 'focal':
                tag_loss = tag_loss.sum()

            if not return_dict:
                result = (masked_loss, class_logits, tag_loss) + outputs[2:]
            else:
                result = {
                    'masked_loss': masked_loss,
                    'class_logits': class_logits,
                    # 'pooled_output': outputs[1],
                    'masked_ids': masked_ids,
                    'tag_loss': tag_loss,
                    'tag_logits': tag_logit,
                }
                if len(outputs) > 2:
                    # intermediate layers' infomation
                    result['inter_info'] = outputs[2:]
                    result['last_hidden'] = all_sequence_output
        else:
            sequence_output = outputs[0][:, :input_ids.shape[-1], :]
            class_logits = self.cls(sequence_output)
            if not return_dict:
                result = (class_logits,) + outputs[2:]
            else:
                if not return_hidden:
                    result = {
                        'class_logits': class_logits,
                    }
                else:
                    result = {
                        'class_logits': class_logits,
                        'hidden_states': sequence_output[2:]
                    }
        return result

    def prepare_inputs_for_generation(self, curr_ids, past=None):
        # NOTE: if attention is on, it should be the token used to mask words in training
        mask_token_id = self.mask_token_id
        batch_size = curr_ids.shape[0]
        mask_ids = torch.full(
            (batch_size, 1), mask_token_id, dtype=torch.long, device=curr_ids.device
        )

        def _slice(t, start, end):
            if t is None:
                return t
            assert t.shape == (batch_size, self.max_seq_len + self.od_labels_len)
            return t[:, start: end]

        def _remove_elements(t, start, end):
            if t is None:
                return t
            assert t.shape == (batch_size, self.max_seq_len + self.od_labels_len)
            return torch.cat([t[:, :start], t[:, end:]], dim=1)

        if past is None:
            input_ids = torch.cat([curr_ids, mask_ids], dim=1)

            curr_len = input_ids.shape[1]
            full_len = self.max_seq_len + self.od_labels_len + self.img_seq_len
            assert self.full_attention_mask.shape == (batch_size,
                    full_len, full_len)

            def _remove_rows_cols(t, row_start, row_end, col_start, col_end):
                t00 = t[:, :row_start, :col_start]
                t01 = t[:, :row_start, col_end:]
                t10 = t[:, row_end:, :col_start]
                t11 = t[:, row_end:, col_end:]
                res = torch.cat([torch.cat([t00, t01], dim=2), torch.cat([t10, t11],
                            dim=2)], dim=1)
                assert res.shape == (t.shape[0], t.shape[1]-row_end+row_start,
                        t.shape[2]-col_end+col_start)
                return res

            seq_start = curr_len
            seq_end = self.max_seq_len
            attention_mask = _remove_rows_cols(self.full_attention_mask, seq_start,
                    seq_end, seq_start, seq_end)

            masked_pos = _remove_elements(self.full_masked_pos, seq_start, seq_end)
            token_type_ids = _remove_elements(self.full_token_type_ids, seq_start, seq_end)
            position_ids = _remove_elements(self.full_position_ids, seq_start, seq_end)
            img_feats = self.img_feats

            if self.add_od_labels:
                assert self.od_label_ids.shape[1] == self.od_labels_len
                input_ids = torch.cat([input_ids, self.od_label_ids], dim=1)
        else:
            last_token = curr_ids[:, -1:]
            # The representation of last token should be re-computed, because
            # it depends on both self-attention context and input tensor
            input_ids = torch.cat([last_token, mask_ids], dim=1)
            start_pos = curr_ids.shape[1] - 1
            end_pos = start_pos + input_ids.shape[1]
            masked_pos = _slice(self.full_masked_pos, start_pos, end_pos)
            token_type_ids = _slice(self.full_token_type_ids, start_pos, end_pos)
            position_ids = _slice(self.full_position_ids, start_pos, end_pos)

            img_feats = None
            assert past[0].shape[0] == batch_size
            if self.prev_encoded_layers is None:
                assert start_pos == 1  # the first token after BOS
                assert past[0].shape[1] == 2 + self.od_labels_len + self.img_seq_len
                # reorder to [od_labels, img_feats, sentence]
                self.prev_encoded_layers = [
                        torch.cat([x[:, 2:, :], x[:, :start_pos,:]], dim=1)
                        for x in past]
                s2s = self.full_attention_mask[:, :self.max_seq_len,
                        :self.max_seq_len]
                s2i = self.full_attention_mask[:, :self.max_seq_len,
                        self.max_seq_len:]
                i2s = self.full_attention_mask[:, self.max_seq_len:,
                        :self.max_seq_len]
                i2i = self.full_attention_mask[:, self.max_seq_len:,
                        self.max_seq_len:]
                self.full_attention_mask = torch.cat(
                        [torch.cat([i2i, i2s], dim=2),
                        torch.cat([s2i, s2s], dim=2)],
                        dim=1)
            else:
                assert start_pos > 1
                assert past[0].shape[1] == 2
                self.prev_encoded_layers = [torch.cat([x, p[:, :-1, :]], dim=1)
                        for x, p in zip(self.prev_encoded_layers, past)]

            attention_mask = self.full_attention_mask[:,
                self.od_labels_len+self.img_seq_len+start_pos: self.od_labels_len+self.img_seq_len+end_pos,
                :self.od_labels_len+self.img_seq_len+end_pos]

        return {'input_ids': input_ids,   'img_feats': img_feats,
                'masked_pos': masked_pos, 'attention_mask': attention_mask,
                'token_type_ids': token_type_ids, 'position_ids': position_ids,
                'is_training': False,
                'encoder_history_states': self.prev_encoded_layers}

    def get_output_embeddings(self):
        return self.decoder

    def generate(self, img_feats, label=None, attention_mask=None, masked_pos=None, token_type_ids=None,
            position_ids=None, head_mask=None, input_ids=None, max_length=None, do_sample=None, num_beams=None,
            temperature=None, top_k=None, top_p=None, repetition_penalty=None, bos_token_id=None, pad_token_id=None,
            eos_token_ids=None, mask_token_id=None, length_penalty=None, num_return_sequences=None, num_keep_best=1,
            is_decode=None, add_od_labels=False, od_labels_start_posid=None, use_cbs=False, fsm=None, num_constraints=None,
            min_constraints_to_satisfy=None, use_hypo=False, decoding_constraint_flag=None, bad_ending_ids=None, gen_tag_ratio=None,):
        """ Generates captions given image features
        """
        assert is_decode
        batch_size = img_feats.shape[0]
        self.img_seq_len = img_feats.shape[1]
        self.max_seq_len = max_length
        self.mask_token_id = mask_token_id
        self.prev_encoded_layers = None
        # NOTE: num_keep_best is not equavilant to num_return_sequences
        # num_keep_best is the number of hypotheses to keep in beam search
        # num_return_sequences is the repeating times of input, coupled with
        # do_sample=True can generate more than one samples per image
        self.num_keep_best = num_keep_best

        vocab_size = self.config.vocab_size
        if not use_cbs:
            num_fsm_states = 1
        else:
            b, num_fsm_states, f1, v = fsm.shape
            assert b==batch_size and v==vocab_size and f1==num_fsm_states

        # set it to True for inference convenience but d make sure there's no detector tags attached in the end.
        self.add_od_labels = add_od_labels

        # avoid position_ids collision of caption and od labels
        self.od_labels_start_posid = max(od_labels_start_posid, self.max_seq_len)

        # get padded od labels part from input_ids
        assert input_ids.shape[0] == batch_size
        od_label_ids = input_ids[:, self.max_seq_len:]
        self.od_labels_len = input_ids.shape[1] - self.max_seq_len
        input_ids = None

        if input_ids is None:
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."
            assert input_ids.shape[0] == batch_size, "Input batch size must match image features"

        cur_len = input_ids.shape[1]
        if num_return_sequences != 1:
            # Expand input to num return sequences
            input_ids = self._expand_for_beams(input_ids, num_return_sequences)
            effective_batch_size = batch_size * num_return_sequences
        else:
            effective_batch_size = batch_size

        if position_ids is None:
            position_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=input_ids.device)
            posids_len = self.max_seq_len
            if self.add_od_labels:
                od_labels_posids = torch.arange(
                        self.od_labels_start_posid,
                        self.od_labels_start_posid + self.od_labels_len, dtype=torch.long, device=input_ids.device)
                position_ids = torch.cat([position_ids, od_labels_posids])
                posids_len += self.od_labels_len
            position_ids = position_ids.unsqueeze(0).expand([batch_size, posids_len])

        num_expand = num_beams * num_fsm_states * num_return_sequences
        self.od_label_ids = self._expand_for_beams(od_label_ids, num_expand)
        self.img_feats = self._expand_for_beams(img_feats, num_expand)
        self.full_attention_mask = self._expand_for_beams(attention_mask, num_expand)
        self.full_masked_pos = self._expand_for_beams(masked_pos, num_expand)
        self.full_token_type_ids = self._expand_for_beams(token_type_ids, num_expand)
        self.full_position_ids = self._expand_for_beams(position_ids, num_expand)
        self.full_head_mask = self._expand_for_beams(head_mask, num_expand)

        if not use_cbs:
            if num_beams > 1:
                output = self._generate_beam_search(
                    input_ids,
                    cur_len,
                    max_length,
                    do_sample,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    pad_token_id,
                    eos_token_ids,
                    effective_batch_size,
                    length_penalty,
                    num_beams,
                    vocab_size,
                )
            else:
                output = self._generate_no_beam_search(
                    input_ids,
                    cur_len,
                    max_length,
                    do_sample,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    pad_token_id,
                    eos_token_ids,
                    effective_batch_size,
                )
        else:
            from src.tools.captioning.utils_cbs import (ConstrainedBeamSearch,
                    select_best_beam_with_constraints)
            assert self.num_keep_best == 1, 'not supported n_best > 1 for CBS'
            searcher = ConstrainedBeamSearch(eos_token_ids, max_length,
                    num_beams, use_hypo=use_hypo,
                    decoding_constraint_flag=decoding_constraint_flag,
                    bad_ending_ids=bad_ending_ids)
            curr_ids, sum_logprobs = searcher.search(
                    input_ids,
                    None,
                    self._decode_step,
                    fsm,
            )
            curr_ids, logprobs = select_best_beam_with_constraints(
                curr_ids,
                sum_logprobs,
                num_constraints,
                min_constraints_to_satisfy,
                eos_token_ids,
            )
            # (batch_size, n_best, max_len), (batch_size, n_best)
            output = (curr_ids.unsqueeze(1), logprobs.unsqueeze(1))

        return output

    def _expand_for_beams(self, x, num_expand):
        if x is None or num_expand == 1:
            return x

        input_shape = list(x.shape)
        expanded_shape = input_shape[:1] + [num_expand] + input_shape[1:]
        x = x.unsqueeze(1).expand(expanded_shape)
        # (batch_size * num_expand, ...)
        x = x.contiguous().view([input_shape[0] * num_expand] + input_shape[1:])
        return x

    def _do_output_past(self, outputs):
        return len(outputs) > 1

    def prod_generate(self, img_feats, od_label_ids, max_length,
            bos_token_id, eos_token_ids, mask_token_id, od_labels_start_posid,
            add_od_labels=True, cls_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1,
            ):
        """ Generates captions for PROD, batch size=1, num_beams=1.
            Use faster generation where output_hidden_states = True
        """
        batch_size = img_feats.shape[0]
        assert batch_size == 1
        device = img_feats.device
        assert od_label_ids.shape[0] == batch_size
        od_labels_len = od_label_ids.shape[1]
        img_seq_len = img_feats.shape[1]

        mask_ids = torch.full(
            (1, 1), mask_token_id, dtype=torch.long, device=device
        )

        # prepare inputs
        cur_ids = torch.full((1, 1), bos_token_id,
                dtype=torch.long, device=device)

        input_ids = torch.cat([cur_ids, mask_ids, od_label_ids], dim=1)
        token_type_ids = torch.cat([
                torch.tensor([[cls_token_segment_id, sequence_a_segment_id]],
                    dtype=torch.long, device=device),
                torch.full((1, od_labels_len), sequence_b_segment_id,
                    dtype=torch.long, device=device)
                ], dim=1)

        position_ids = torch.arange(2, dtype=torch.long, device=device)
        od_labels_start_posid = max(od_labels_start_posid, max_length)
        if add_od_labels:
            od_labels_posids = torch.arange(
                    od_labels_start_posid, od_labels_start_posid + od_labels_len,
                    dtype=torch.long, device=device)
            position_ids = torch.cat([position_ids, od_labels_posids])
        posids_len = 2 + od_labels_len
        position_ids = position_ids.unsqueeze(0).expand([1, posids_len])

        attention_mask = torch.ones(
                (1, 2+od_labels_len+img_seq_len, 2+od_labels_len+img_seq_len),
                dtype=torch.long, device=device)
        attention_mask[:, 0, 1] = 0   # words in sentence can not see words after itself
        attention_mask[:, 2:, :2] = 0 # od_label, img_feat can not see sentence

        # make empty history states for the first step
        encoder_history_states = tuple(
            torch.empty([1, 0, self.config.hidden_size], device=device)
            for _ in range(self.config.num_hidden_layers)
        )

        # prepare inputs for >1 steps
        token_type_ids_after_first = torch.full([1, 2], sequence_a_segment_id,
                dtype=torch.long, device=device)
        img_feats_after_first = torch.empty([1, 0, self.config.img_feature_dim],
                device=device)  # place holder to avoid None

        # initial model inputs for the first step
        model_inputs = {'input_ids': input_ids, 'img_feats': img_feats,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids, 'position_ids': position_ids,
            'is_training': False,
            'encoder_history_states': encoder_history_states}
        cur_len = cur_ids.shape[1]
        sum_logprob = 0
        while True:
            outputs = self(**model_inputs)

            assert self._do_output_past(outputs)
            if cur_len == 1:
                assert outputs[0].shape[1] == 2 + od_labels_len
            else:
                assert cur_len > 1
                assert outputs[0].shape[1] == 2

            # greedy decoding
            next_token_idx = 1
            next_token_logits = outputs[0][:, next_token_idx, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            # Compute scores
            _scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)
            sum_logprob += _scores[:, next_token].item()

            if next_token in eos_token_ids:
                break
            cur_ids = torch.cat([cur_ids, next_token.unsqueeze(-1)], dim=-1)
            cur_len = cur_ids.shape[1]
            if cur_len == max_length:
                break

            # prepare model inputs for the next step
            past = outputs[1]
            last_token = cur_ids[:, -1:]
            input_ids = torch.cat([last_token, mask_ids], dim=1)
            position_ids = torch.arange(cur_len - 1, cur_len + 1,
                    dtype=torch.long, device=device)
            attention_mask = torch.ones([1, 2, od_labels_len+img_seq_len+cur_len+1],
                    dtype=torch.long, device=device)
            attention_mask[:, 0, -1] = 0
            assert past[0].shape[0] == batch_size
            # special handle for the first step
            if cur_len == 2:
                assert past[0].shape[1] == 2 + od_labels_len + img_seq_len
                # remove the first token after BOS
                # reorder to [od_labels, img_feats, sentence]
                encoder_history_states = [
                        torch.cat([x[:, 2:, :], x[:, :1,:]], dim=1)
                        for x in past]
            else:
                assert cur_len > 2
                assert past[0].shape[1] == 2
                encoder_history_states = [torch.cat([x, p[:, :-1, :]], dim=1)
                        for x, p in zip(encoder_history_states, past)]

            model_inputs = {'input_ids': input_ids,
                'img_feats': img_feats_after_first,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids_after_first,
                'position_ids': position_ids,
                'is_training': False,
                'encoder_history_states': encoder_history_states}

        logprob = sum_logprob / cur_ids.shape[1]

        # (batch_size, max_len), (batch_size)
        return cur_ids, torch.full((1,), logprob, device=device)

    def prod_no_hidden_generate(self, img_feats, od_label_ids, max_length,
            bos_token_id, eos_token_ids, mask_token_id, od_labels_start_posid,
            add_od_labels=True, cls_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1,
            ):
        """ Generates captions for PROD, batch size=1, num_beams=1.
            Use output_hidden_states = False
        """
        batch_size = img_feats.shape[0]
        assert batch_size == 1
        device = img_feats.device
        assert od_label_ids.shape[0] == batch_size
        od_labels_len = od_label_ids.shape[1]
        img_seq_len = img_feats.shape[1]

        mask_ids = torch.full(
            (1, 1), mask_token_id, dtype=torch.long, device=device
        )

        # prepare inputs
        cur_ids = torch.full((1, 1), bos_token_id,
                dtype=torch.long, device=device)
        od_labels_start_posid = max(od_labels_start_posid, max_length)
        triangle_mask = torch.tril(torch.ones([max_length, max_length],
                dtype=torch.long, device=device))

        def _prepare_inputs(cur_ids):
            cur_len = cur_ids.shape[1]
            input_ids = torch.cat([cur_ids, mask_ids, od_label_ids], dim=1)
            token_type_ids = torch.cat([
                    torch.tensor([[cls_token_segment_id]],
                        dtype=torch.long, device=device),
                    torch.full((1, cur_len), sequence_a_segment_id,
                        dtype=torch.long, device=device),
                    torch.full((1, od_labels_len), sequence_b_segment_id,
                        dtype=torch.long, device=device)
                    ], dim=1)

            token_len = cur_len + 1
            position_ids = torch.arange(token_len, dtype=torch.long, device=device)
            if add_od_labels:
                od_labels_posids = torch.arange(
                        od_labels_start_posid, od_labels_start_posid + od_labels_len,
                        dtype=torch.long, device=device)
                position_ids = torch.cat([position_ids, od_labels_posids])
            posids_len = token_len + od_labels_len
            position_ids = position_ids.unsqueeze(0).expand([1, posids_len])

            attention_mask = torch.ones(
                    (1, token_len+od_labels_len+img_seq_len, token_len+od_labels_len+img_seq_len),
                    dtype=torch.long, device=device)
            attention_mask[:, :token_len,
                    :token_len].copy_(triangle_mask[:token_len, :token_len])
            attention_mask[:, token_len:, :token_len] = 0 # od_label, img_feat can not see sentence
            return input_ids, token_type_ids, position_ids, attention_mask

        # initial model inputs for the first step
        input_ids, token_type_ids, position_ids, attention_mask = \
                _prepare_inputs(cur_ids)
        model_inputs = {'input_ids': input_ids, 'img_feats': img_feats,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids, 'position_ids': position_ids,
            'is_training': False,
            }
        cur_len = cur_ids.shape[1]
        sum_logprob = 0
        while True:
            outputs = self(**model_inputs)

            assert not self._do_output_past(outputs)

            # greedy decoding
            next_token_idx = cur_len
            next_token_logits = outputs[0][:, next_token_idx, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            # Compute scores
            _scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)
            sum_logprob += _scores[:, next_token].item()

            if next_token in eos_token_ids:
                break
            cur_ids = torch.cat([cur_ids, next_token.unsqueeze(-1)], dim=-1)
            cur_len = cur_ids.shape[1]
            if cur_len == max_length:
                break

            # prepare model inputs for the next step
            input_ids, token_type_ids, position_ids, attention_mask = \
                    _prepare_inputs(cur_ids)
            model_inputs = {'input_ids': input_ids,
                'img_feats': img_feats,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'position_ids': position_ids,
                'is_training': False,
                }

        logprob = sum_logprob / cur_ids.shape[1]

        # (batch_size, max_len), (batch_size)
        return cur_ids, torch.full((1,), logprob, device=device)


class ViTSplitCLSEmbModel(BertPreTrainedModel):

    def __init__(self, config):
        super(ViTSplitCLSEmbModel, self).__init__(config)

        self.config = config
        self.embeddings = BertEmbeddings(config)

        self.extra_embeddings = BertEmbeddings(config)

        self.encoder = TIMMVitSplitEncoder(config)

        # let's simply re-use bert-pooler
        self.caption_pooler = BertPooler(config)

        # For Tag Prediction
        self.pooler = BertPooler(config)
        self.tag_vocab = config.vocab
        self.tokenizer = config.tokenizer

        if config.category == 'vinvl':
            caption_vocab_size = config.vocab_size
            config.vocab_size = len(self.tag_vocab['label_to_idx'])
            self.tag_logit = BertCaptioningHeads(config)
            config.vocab_size = caption_vocab_size
        else:
            self.tag_logit = BertCaptioningHeads(config)

        self.category = config.category

        self.img_dim = config.img_feature_dim  # 2054 #565
        logger.info('BertImgModel Image Dimension: {}'.format(self.img_dim))

        self.config = config

        if getattr(config, 'decoder_layer', None):
            config.num_hidden_layers = config.decoder_layer
        else:
            config.num_hidden_layers = 4
        self.decoder = BertEncoder(config)

        self.img_dim = config.img_feature_dim  # 2054 #565
        logger.info('BertImgModel Image Dimension: {}'.format(self.img_dim))

        self.img_feature_type = config.img_feature_type

        try:
            self.use_img_layernorm = config.use_img_layernorm
        except:
            self.use_img_layernorm = None

        self.config = config

        self.img_embedding = nn.Identity()
        self.dropout = nn.Identity()

        self.decoder.apply(self.init_weights)
        self.embeddings.apply(self.init_weights)
        self.caption_pooler.apply(self.init_weights)

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def encode_tag_to_embedding(self, pred_topk, cls_emb=None, caption_len=20):
        seq_length = pred_topk.size(1)

        # TAG classifier embedding
        if cls_emb is not None:
            tag_embedding = F.embedding(pred_topk,
                                        weight=cls_emb.weight,
                                        padding_idx=0,
                                        max_norm=None,
                                        norm_type=2,
                                        scale_grad_by_freq=False,
                                        sparse=False)
        # BERT Tokenizer embedding
        else:
            tag_embedding = self.embeddings.word_embeddings(pred_topk)

        tag_position_ids = torch.arange(seq_length, dtype=torch.long, device=pred_topk.device) + caption_len
        tag_position_ids = tag_position_ids.unsqueeze(0).expand_as(pred_topk)
        tag_position_embedding = self.embeddings.position_embeddings(tag_position_ids)
        tag_token_type_ids = torch.zeros_like(pred_topk)
        tag_token_type_embedding = self.embeddings.token_type_embeddings(tag_token_type_ids)

        tag_embedding = tag_embedding + tag_position_embedding + tag_token_type_embedding
        tag_embedding = self.embeddings.LayerNorm(tag_embedding)
        tag_embedding = self.embeddings.dropout(tag_embedding)
        return tag_embedding

    def forward(self, input_ids, img_feats=None, label=None, token_type_ids=None, attention_mask=None,
                position_ids=None, encoder_history_states=None, head_mask=None, cls_emb=None, gen_tag_ratio=None):

        # =========================================== ViT Encoder ==================================================
        head_mask = [None] * self.config.num_hidden_layers

        # encoder visual attention
        visual_attention = torch.zeros(img_feats.shape[0], 1, img_feats.shape[-2], img_feats.shape[-2]).cuda()

        # feed the image encodings into the encoder
        encoder_outputs, tag_encoder_outputs = self.encoder(img_feats,
                                                            visual_attention,
                                                            head_mask=head_mask,
                                                            encoder_history_states=encoder_history_states)

        # use the CLS token for Multi-Tags classification. Decoding here as the tags.
        pooled_output = self.pooler(tag_encoder_outputs)
        logit = self.tag_logit(pooled_output)

        # non-differentiable tokenization
        with torch.no_grad():
            offline_logit = torch.nn.functional.sigmoid(logit.detach())
            topk = self.config.topk
            prob, pred_topk = offline_logit.topk(topk, dim=1, largest=True)
            topk_len = (prob >= 0.2).sum(dim=1)

        # Training mode. Attach it after 20th token (default caption length)
        if topk_len[0] + 20 <= input_ids.shape[1]:

            if gen_tag_ratio is not None:
                # fuse the generated tags with GT tags at specific portion X%
                for batch_idx, lab in enumerate(label):
                    batch_tag = torch.nonzero(lab, as_tuple=False).squeeze(1)
                    batch_len = int((1 - gen_tag_ratio) * len(batch_tag))
                    indices = torch.randperm(batch_len)
                    batch_tag = batch_tag[indices]
                    pred_topk[batch_idx, :batch_len] = batch_tag

            # manually add the end token
            pred_topk[:, -1] = 102

            # textual encoding
            embedding_output = self.embeddings(input_ids, position_ids=position_ids,
                                               token_type_ids=token_type_ids)

            # using Tag classifier embedding: Use classifier's embedding but BERT tokenizer's position id
            if getattr(self.config, 'tagemb', None) == 'cls':
                # tag_embedding = self.encode_tag_to_embedding(pred_topk, cls_emb=cls_emb,)
                tag_embedding = F.embedding(pred_topk,
                                            weight=cls_emb.weight,
                                            padding_idx=0,
                                            max_norm=None,
                                            norm_type=2,
                                            scale_grad_by_freq=False,
                                            sparse=False)

            # using BERT tokenizer embedding
            else:
                tag_embedding = self.encode_tag_to_embedding(pred_topk, cls_emb=None,)
                # tag_embedding = self.extra_embeddings(pred_topk)

            # apply the tag embedding for concatenation
            embedding_output[:, -pred_topk.shape[1]:] = tag_embedding

        # Inference mode. Attach it right after the SEP token.
        else:
            start_id = input_ids.shape[1] - topk_len
            # input_ids[:, start_id:] = encoded_tags
            input_ids[:, start_id] = 102
            pred_topk[:, -1] = 102

            # tag encoding
            if getattr(self.config, 'tagemb', None) == 'cls':
                tag_embedding = self.encode_tag_to_embedding(pred_topk, cls_emb=cls_emb, )
            else:
                # tag_embedding = self.encode_tag_to_embedding(pred_topk, cls_emb=None,)
                tag_embedding = self.extra_embeddings(pred_topk,
                                                      position_ids=position_ids[:, -pred_topk.shape[1]:])

            embedding_output = self.embeddings(input_ids, position_ids=position_ids,
                                               token_type_ids=token_type_ids)
            embedding_output[:, -pred_topk.shape[1]:] = tag_embedding

        # =========================================== Bert Encoder ==================================================
        # FIXME: for Tagger CLS token to Visual Tokens
        encoder_outputs = torch.cat([tag_encoder_outputs[:, 0, :].unsqueeze(1), encoder_outputs], 1)
        attention_mask = torch.cat([attention_mask, attention_mask[:, -1].unsqueeze(1)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones(attention_mask.shape[0],
                                                               attention_mask.shape[1], 1).cuda()], dim=2)

        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = torch.cat((embedding_output, encoder_outputs), 1)

        decoder_outputs = self.decoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask,
                                       encoder_history_states=encoder_history_states)
        decoder_outputs = decoder_outputs[0]

        # =========================================== CLS Predict ==================================================
        sequence_output = decoder_outputs
        pooled_output = self.caption_pooler(sequence_output)

        outputs = (sequence_output, pooled_output,)
        return outputs, logit  # sequence_output, pooled_output,
