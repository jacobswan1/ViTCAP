# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""SCAN model"""
import math
import os
import torch
import torch.nn as nn
import torch.nn.init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import numpy as np
from collections import OrderedDict
from PIL import Image, ImageDraw, ImageFont


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.norm(X, dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def EncoderFeature(feat_dim, embed_size,
                 precomp_enc_type='basic', no_featnorm=False):
    """A wrapper for feature encoders. Chooses between an different encoders
    that uses precomputed image/text features.
    """
    if precomp_enc_type == 'basic':
        feat_enc = EncoderFeaturePrecomp(
            feat_dim, embed_size, no_featnorm)
    elif precomp_enc_type == 'weight_norm':
        feat_enc = EncoderFeatureWeightNormPrecomp(
            feat_dim, embed_size, no_featnorm)
    else:
        raise ValueError(
            "Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return feat_enc


class EncoderFeaturePrecomp(nn.Module):

    def __init__(self, feat_dim, embed_size, no_featnorm=False):
        super(EncoderFeaturePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_featnorm = no_featnorm
        self.fc = nn.Linear(feat_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, pre_features, feat_lens):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(pre_features)

        # normalize in the joint embedding space
        if not self.no_featnorm:
            features = l2norm(features, dim=-1)

        # ensuring that padded features always equal zero
        mask = torch.zeros((features.shape[0], features.shape[1]),
                           dtype=torch.bool)
        for i in range(mask.shape[0]):
            mask[i, feat_lens[i]:] = True
        mask.unsqueeze_(-1)

        if torch.cuda.is_available():
            mask = mask.cuda()
        features.masked_fill_(mask, 0)

        return features

    def load_state_dict(self, state_dict, strict=True):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderFeaturePrecomp, self).load_state_dict(new_state, strict)


class EncoderFeatureWeightNormPrecomp(nn.Module):

    def __init__(self, feat_dim, embed_size, no_featnorm=False):
        super(EncoderFeatureWeightNormPrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_featnorm = no_featnorm
        self.fc = weight_norm(nn.Linear(feat_dim, embed_size), dim=None)

    def forward(self, pre_features, feat_lens):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(pre_features)

        # normalize in the joint embedding space
        if not self.no_featnorm:
            features = l2norm(features, dim=-1)

        # ensuring that padded features always equal zero
        mask = torch.zeros((features.shape[0], features.shape[1]),
                           dtype=torch.bool)
        for i in range(mask.shape[0]):
            mask[i, feat_lens[i]:] = True
        mask.unsqueeze_(-1)

        if torch.cuda.is_available():
            mask = mask.cuda()
        features.masked_fill_(mask, 0)

        return features

    def load_state_dict(self, state_dict, strict=True):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderFeatureWeightNormPrecomp, self).load_state_dict(new_state,
                                                                   strict)


# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False, features_as_input=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm
        self.features_as_input = features_as_input

        # word embedding
        if features_as_input:
            # reuse the EncodeImage for the text feature embedding. We will do the txtnorm in the last layer
            self.embed = EncoderFeature(
                vocab_size, word_dim,
                precomp_enc_type='basic',
                no_featnorm=True,
            )
        else:
            self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        if num_layers == 0:
            assert word_dim == embed_size, \
                "for GRU with 0 lWayer, the input embedding size should be the output embedding size!"
            self.rnn = None
        else:
            self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True,
                              bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        if not self.features_as_input:
            self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        if self.features_as_input:
            x = self.embed(x, lengths)
        else:
            x = self.embed(x)

        if self.rnn:
            # we have rnn with layers greater than 1
            packed = pack_padded_sequence(x, lengths, batch_first=True)

            # Forward propagate RNN
            out, _ = self.rnn(packed)

            # Reshape *final* output to (batch_size, hidden_size)
            padded = pad_packed_sequence(out, batch_first=True)
            cap_emb, cap_len = padded
            if self.use_bi_gru:
                cap_emb = (cap_emb[:, :, :cap_emb.size(2) // 2] +
                           cap_emb[:, :, cap_emb.size(2) // 2:]) / 2
        else:
            # layer == 0
            cap_emb = x
            cap_len = lengths
            if not self.features_as_input:
                # if token as input, ensuring that padded features always equal zero
                mask = torch.zeros((cap_emb.shape[0], cap_emb.shape[1]),
                                   dtype=torch.bool)
                for i in range(mask.shape[0]):
                    mask[i, cap_len[i]:] = True
                mask.unsqueeze_(-1)

                if torch.cuda.is_available():
                    mask = mask.cuda()
                cap_emb.masked_fill_(mask, 0)

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)
        return cap_emb, cap_len


def func_attention(query, context, opt, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if opt.RAW_FEATURE_NORM == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif opt.RAW_FEATURE_NORM == "l2norm":
        attn = l2norm(attn, 2)
    elif opt.RAW_FEATURE_NORM == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif opt.RAW_FEATURE_NORM == "l1norm":
        attn = l1norm(attn, 2)
    elif opt.RAW_FEATURE_NORM == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm(attn, 2)
    elif opt.RAW_FEATURE_NORM == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif opt.RAW_FEATURE_NORM == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opt.RAW_FEATURE_NORM)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax(dim=-1)(attn * smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)

    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def xattn_score_t2i(images, captions, cap_lens, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    attn_maps = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext, attn = func_attention(cap_i_expand, images, opt,
                                          smooth=opt.LAMBDA_SOFTMAX)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        attn_maps.append(torch.transpose(attn.data.cpu(), 1, 2).numpy())

        if opt.AGG_FUNC == 'LogSumExp':
            row_sim.mul_(opt.LAMBDA_LSE).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / opt.LAMBDA_LSE
        elif opt.AGG_FUNC == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.AGG_FUNC == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.AGG_FUNC == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.AGG_FUNC))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)

    return similarities, attn_maps


def xattn_score_i2t(images, img_lens, captions, cap_lens, opt):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    attn_maps = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext, attn = func_attention(images, cap_i_expand, opt,
                                          smooth=opt.LAMBDA_SOFTMAX)
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        attn_maps.append(attn.data.cpu().numpy())

        if opt.AGG_FUNC == 'LogSumExp':
            # need to filter out the effect of padding
            pad_size = img_lens.max() - img_lens
            pad_size.unsqueeze_(-1)

            row_sim.mul_(opt.LAMBDA_LSE).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True) - pad_size.float()
            row_sim = torch.log(row_sim) / opt.LAMBDA_LSE
        elif opt.AGG_FUNC == 'Max':
            # TODO: filter out the effect of padding here
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.AGG_FUNC == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.AGG_FUNC == 'Mean':
            row_sim = row_sim.sum(dim=1, keepdim=True) / img_lens.unsqueeze(-1)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.AGG_FUNC))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities, attn_maps


def visualize_attn_map(I, attMaps, boxes, words):
    columns = 2
    rows = int(math.ceil((len(attMaps) + 2) / float(columns)))
    h_unit_off, v_unit_off, txt_unit_size = 10, 5, 19
    total_width = I.shape[1] * columns + h_unit_off * (columns + 1)
    total_height = I.shape[0] * rows + v_unit_off * (rows + 1)

    new_im = Image.new('RGBA', (total_width, total_height))
    white_img = Image.fromarray(
        np.uint8(np.ones((total_height, total_width, 3)) * 255))
    new_im.paste(white_img, (0, 0, total_width, total_height))
    fnt = ImageFont.truetype(
        os.path.join(os.path.dirname(__file__), 'freemono.ttf'), size=30,
    )
    PIL_im = Image.fromarray(np.uint8(I))
    new_im.paste(PIL_im, (0, 0, int(I.shape[1]), int(I.shape[0])))

    pos_y, pos_x = 0, 1
    h_off = h_unit_off * (pos_x + 1) + I.shape[1] * pos_x
    v_off = v_unit_off * (pos_y + 1) + I.shape[0] * pos_y
    for bbox in boxes:
        d = ImageDraw.Draw(PIL_im)
        d.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=(255, 0, 0))
    new_im.paste(PIL_im, (h_off, v_off, h_off + I.shape[1], v_off + I.shape[0]))

    for i, word in enumerate(words):
        attMap = attMaps[i]  # Inputs to the network
        shape = (I.shape[0], I.shape[1], 1)
        A = np.zeros(shape)
        for k in range(len(attMap)):
            bbox = boxes[k].astype(int)
            A[bbox[1]:bbox[3], bbox[0]:bbox[2]] += attMap[k]
        A /= np.max(A)
        A = A * I + (1.0 - A) * 255
        A = Image.fromarray(A.astype('uint8'))
        d = ImageDraw.Draw(A)
        bbox = boxes[np.argmax(attMap)]
        d.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=(255, 0, 0))
        txt_len = len(word) * txt_unit_size
        d.rectangle([0, 0, txt_len, 40], fill=(0, 80, 200))
        d.text((0, 0), word, font=fnt, fill=(255, 255, 255))

        pos_y, pos_x = (i + 2) // columns, (i + 2) % columns
        h_off = h_unit_off * (pos_x + 1) + I.shape[1] * pos_x
        v_off = v_unit_off * (pos_y + 1) + I.shape[0] * pos_y
        new_im.paste(A, (h_off, v_off, h_off + I.shape[1], v_off + I.shape[0]))
    return new_im


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, im_l, s, s_l):
        # compute image-sentence score matrix
        if self.opt.CROSS_ATTN == 't2i':
            scores, _ = xattn_score_t2i(im, s, s_l, self.opt)
        elif self.opt.CROSS_ATTN == 'i2t':
            scores, _ = xattn_score_i2t(im, im_l, s, s_l, self.opt)
        else:
            raise ValueError("unknown first norm type:",
                             self.opt.RAW_FEATURE_NORM)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5

        if torch.cuda.is_available():
            mask = mask.cuda()
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()
