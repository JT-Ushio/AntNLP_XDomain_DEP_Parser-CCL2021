import sys
import argparse

import torch
import torch.nn as nn
import numpy as np
from antu.io import Vocabulary
from antu.io import glove_reader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from module.bilstm import BiLSTM
from module.biaffine import Biaffine
from module.dropout import IndependentDropout, SharedDropout
from module.mlp import MLP
from module.transformer import XformerEncoder, LearnedPositionEncoding


class Parser(nn.Module):

    def __init__(
        self,
        vocabulary: Vocabulary,
        cfg: argparse.Namespace):
        super(Parser, self).__init__()

        # Build word lookup from pre-trained embedding file
        _, v_glove = glove_reader(cfg.GLOVE)
        d_glove, n_glove = len(v_glove[1]), vocabulary.get_vocab_size('glove')
        v_glove = [[0.0]*d_glove, [0.0]*d_glove] + v_glove
        v_glove = np.array(v_glove, dtype=np.float32) #/np.std(v_glove)
        PAD = vocabulary.get_padding_index('glove')
        self.glookup = nn.Embedding(n_glove, d_glove, padding_idx=PAD)
        self.glookup.weight.data.copy_(torch.from_numpy(v_glove))
        self.glookup.weight.requires_grad = not cfg.IS_FIX_GLOVE

        # Build word lookup embedding
        n_word = vocabulary.get_vocab_size('word')
        PAD = vocabulary.get_padding_index('word')
        self.wlookup = nn.Embedding(n_word, d_glove, padding_idx=PAD)
        self.wlookup.weight.data.fill_(0)

        # Build POS tag lookup
        n_tag = vocabulary.get_vocab_size('tag')
        PAD = vocabulary.get_padding_index('tag')
        self.tlookup = nn.Embedding(n_tag, cfg.D_TAG, padding_idx=PAD)
        # self.tlookup_rel = nn.Embedding(n_tag, 50, padding_idx=PAD)
        # self.tlookup_arc = nn.Embedding(n_tag, 50, padding_idx=PAD)
        # Emb. Dropout
        self.emb_drop = IndependentDropout(cfg.EMB_DROP)
        # self.tag_drop = nn.Dropout(p=cfg.MLP_DROP)

        # Encoder Layer
        ## BiLSTM
        if cfg.MODEL_TYPE == 'RNN':
            D_RNN_IN = d_glove+cfg.D_TAG
            self.bilstm = BiLSTM(
                D_RNN_IN, cfg.D_RNN_HID, cfg.N_RNN_LAYER, cfg.RNN_DROP)
            self.bilstm_drop = SharedDropout(cfg.RNN_DROP)
            D_MLP_IN = cfg.D_RNN_HID*2
        ## Xformer
        elif 'Xformer' in cfg.MODEL_TYPE:
            self.xformer = XformerEncoder(cfg)
            self.xformer_drop = SharedDropout(cfg.MLP_DROP)
            D_MLP_IN = cfg.D_MODEL

        # MLP Layer
        self.mlp_d = MLP(D_MLP_IN, cfg.D_ARC+cfg.D_REL, cfg.MLP_DROP)
        self.mlp_h = MLP(D_MLP_IN, cfg.D_ARC+cfg.D_REL, cfg.MLP_DROP)
        self.d_arc = cfg.D_ARC
        # Bi-affine Layer
        self.arc_attn = Biaffine(cfg.D_ARC, 1, True, False)
        n_rel = vocabulary.get_vocab_size('rel')
        self.rel_attn = Biaffine(cfg.D_REL, n_rel, True, True)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        # Define CE_loss
        self.CELoss = nn.CrossEntropyLoss()


    def forward(self, x):
        max_len, lens = x['w_lookup'].size()[1], x['mask'].sum(dim=1)

        # Embedding Layer
        v_w = self.wlookup(x['w_lookup']) + self.glookup(x['g_lookup'])
        v_t = self.tlookup(x['t_lookup'])
        # v_t_arc, v_t_rel = self.tlookup_arc(x['t_lookup']), self.tlookup_rel(x['t_lookup'])
        # v_t_arc, v_t_rel = self.tag_drop(v_t_arc), self.tag_drop(v_t_rel)
        v_w, v_t = self.emb_drop(v_w, v_t)
        v = torch.cat((v_w, v_t), dim=-1)

        # BiLSTM Layer
        if hasattr(self, 'bilstm'):
            v = pack_padded_sequence(v, lens.cpu(), True, False)
            v, _ = self.bilstm(v)
            v, _ = pad_packed_sequence(v, True, total_length=max_len)
            v = self.bilstm_drop(v)
        # Xformer Layer
        elif hasattr(self, 'xformer'):
            v = v.permute(1, 0, 2)
            v = self.xformer(v, ~x['mask'])
            v = v.permute(1, 0, 2)
            v = self.xformer_drop(v)

        # MLP Layer
        h, d = self.mlp_h(v), self.mlp_d(v)
        h_arc, d_arc = h[..., :self.d_arc], d[..., :self.d_arc]
        h_rel, d_rel = h[..., self.d_arc:], d[..., self.d_arc:]

        # h_arc = torch.cat((h_arc, v_t_arc), dim=-1)
        # d_arc = torch.cat((d_arc, v_t_arc), dim=-1)
        # h_rel = torch.cat((h_rel, v_t_rel), dim=-1)
        # d_rel = torch.cat((d_rel, v_t_rel), dim=-1)

        # Arc Bi-affine Layer
        s_arc = self.arc_attn(d_arc, h_arc)
        s_arc.masked_fill_(~x['mask'].unsqueeze(1), float('-inf'))

        # mask the ROOT token
        x['mask'][:, 0] = 0
        pred_arc = s_arc[x['mask']]

        # Rel Bi-affine Layer
        s_rel = self.rel_attn(d_rel, h_rel).permute(0, 2, 3, 1)
        pred_rel = s_rel[x['mask']]

        # Calc CE_Loss
        gold_arc = x['head'][x['mask'].view(-1)]
        gold_rel = x['rel'][x['mask'].view(-1)]
        pred_rel_, pred_arc_ = pred_rel, pred_arc
        n_token = torch.arange(len(gold_arc))
        pred_rel = pred_rel[n_token, gold_arc]

        gold_arc_loss = gold_arc.masked_select(x['prob'])
        gold_rel_loss = gold_rel.masked_select(x['prob'])
        pred_rel_loss = pred_rel[x['prob'].nonzero(), :].squeeze()
        pred_arc_loss = pred_arc[x['prob'].nonzero(), :].squeeze()
        arc_loss = self.CELoss(pred_arc_loss, gold_arc_loss)
        rel_loss = self.CELoss(pred_rel_loss, gold_rel_loss)
        if self.training: return arc_loss, rel_loss
        pred_arc = pred_arc_.argmax(-1)
        pred_rel = pred_rel_[n_token, pred_arc].argmax(-1)
        return arc_loss, rel_loss, pred_arc.tolist(), pred_rel.tolist()




