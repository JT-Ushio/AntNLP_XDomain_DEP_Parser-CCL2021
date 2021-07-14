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
from utils.mst_decoder import MST_inference
from torch.optim import AdamW
from module.exp_scheduler import ExponentialLRwithWarmUp
from module.gat import GraphAttentionLayer


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
        v_glove = np.array(v_glove, dtype=np.float32) / np.std(v_glove)
        PAD = vocabulary.get_padding_index('glove')
        self.glookup = nn.Embedding(n_glove, d_glove, padding_idx=PAD)
        self.glookup.weight.data.copy_(torch.from_numpy(v_glove))
        self.glookup.weight.requires_grad = not cfg.IS_FIX_GLOVE

        # Build word lookup embedding
        n_word = vocabulary.get_vocab_size('word')
        PAD = vocabulary.get_padding_index('word')
        self.wlookup = nn.Embedding(n_word, d_glove, padding_idx=PAD)
        self.wlookup.weight.data.fill_(0)

        # Build char lookup embedding
        if cfg.D_CHAR:
            n_char = vocabulary.get_vocab_size('char')
            PAD = vocabulary.get_padding_index('char')
            self.clookup = nn.Embedding(n_char, cfg.D_CHAR, padding_idx=PAD)
            self.charlstm = nn.LSTM(cfg.D_CHAR, cfg.D_CHARNN_HID, cfg.N_CHARNN_LAYER, dropout=cfg.RNN_DROP, bidirectional=True, batch_first=True)
            self.charlstm_drop = nn.Dropout(p=cfg.RNN_DROP)
            self.char_emb_drop = nn.Dropout(p=cfg.EMB_DROP)

        # Build POS tag lookup
        if cfg.D_TAG:
            n_tag = vocabulary.get_vocab_size('tag')
            PAD = vocabulary.get_padding_index('tag')
            self.tlookup = nn.Embedding(n_tag, cfg.D_TAG, padding_idx=PAD)

        # Emb. Dropout
        self.emb_drop = IndependentDropout(cfg.EMB_DROP)
        self.N_GNN = cfg.N_GNN_LAYER
        self.H_GAT = GraphAttentionLayer(cfg.D_ARC, cfg.D_ARC, cfg.MLP_DROP, 0.1)
        self.D_GAT = GraphAttentionLayer(cfg.D_ARC, cfg.D_ARC, cfg.MLP_DROP, 0.1)
        self.norm_h = nn.ModuleList([nn.LayerNorm(cfg.D_ARC) for _ in range(self.N_GNN)])
        self.norm_d = nn.ModuleList([nn.LayerNorm(cfg.D_ARC) for _ in range(self.N_GNN)])
        self.attn_drop = nn.Dropout(p=cfg.GNN_ATTN_DROP)

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
        self.LAMBDA = cfg.LAMBDA
        self.arc_attn = nn.ModuleList([Biaffine(cfg.D_ARC, 1, True, False) for _ in range(self.N_GNN)])
        self.arc_attn_last = Biaffine(cfg.D_ARC, 1, True, False)
        n_rel = vocabulary.get_vocab_size('rel')
        self.rel_attn = Biaffine(cfg.D_REL, n_rel, True, True)
        # Define CE_loss
        self.CELoss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)
        self.MIN_PROB = cfg.MIN_PROB

    def set_optimizer(self, cfg):
        self.optim = AdamW(self.parameters(), cfg.LR, cfg.BETAS, cfg.EPS)
        self.sched = ExponentialLRwithWarmUp(
            self.optim, cfg.LR_DECAY, cfg.LR_ANNEAL, cfg.LR_DOUBLE, cfg.LR_WARM)
        self.CLIP = cfg.CLIP

    def update(self):
        nn.utils.clip_grad_norm_(self.parameters(), self.CLIP)
        self.optim.step()
        self.optim.zero_grad()
        self.sched.step()

    def calu_arc_loss(self, s_arc, x, prob, loss_weight, gold_arc_loss):
        pred_arc = s_arc[x['mask_root']]
        pred_arc_loss = pred_arc[prob.nonzero(), :].squeeze()
        arc_loss = self.CELoss(pred_arc_loss, gold_arc_loss)
        arc_loss = torch.mean(arc_loss * loss_weight)
        return arc_loss

    def calu_rel_loss(self, s_rel, x, n_token, prob, loss_weight, gold_arc):
        pred_rel = s_rel[x['mask_root']]
        gold_rel = x['rel'][x['mask_root'].view(-1)]
        pred_rel_ = pred_rel[n_token, gold_arc]
        gold_rel_loss = gold_rel.masked_select(prob)
        pred_rel_loss = pred_rel_[prob.nonzero(), :].squeeze()
        rel_loss = self.CELoss(pred_rel_loss, gold_rel_loss)
        rel_loss = torch.mean(rel_loss * loss_weight)
        return pred_rel, rel_loss

    def forward(self, x):
        max_len, lens = x['w_lookup'].size(1), x['mask'].sum(dim=1)

        # Embedding Layer
        v_w = self.wlookup(x['w_lookup']) + self.glookup(x['g_lookup'])
        if hasattr(self, 'tlookup'):
            v_t = self.tlookup(x['t_lookup'])
        if hasattr(self, 'charlstm'):
            char_lens = (x['c_lookup']!=1).sum(dim=1)
            v_c = self.clookup(x['c_lookup'])
            v_c = self.char_emb_drop(v_c)
            v_c = pack_padded_sequence(v_c, char_lens.cpu(), True, False)
            _, (v_c, _) = self.charlstm(v_c)
            v_c = torch.cat((v_c[0, ...], v_c[1, ...]), dim=-1)
            v_c = self.charlstm_drop(v_c)
            v_c = torch.cat((torch.zeros_like(v_c[:1, ...]), v_c), dim=0)
            v_c = v_c[x['w2c'], :]
            v_t = torch.cat((v_t, v_c), dim=-1) if hasattr(self, 'tlookup') else v_c
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

        prob = x['prob']>self.MIN_PROB
        loss_weight = x['prob'][prob]
        n_token = torch.arange(x['mask_root'].sum())
        gold_arc = x['head'][x['mask_root'].view(-1)]
        gold_arc_loss = gold_arc.masked_select(prob)

        gnn_loss = 0
        for i in range(self.N_GNN):
            # Arc Bi-affine Layer
            s_arc = self.arc_attn[i](d_arc, h_arc)
            s_arc.masked_fill_(~x['mask'].unsqueeze(1), float('-inf'))
            arc_loss = self.calu_arc_loss(s_arc, x, prob, loss_weight, gold_arc_loss)
            gnn_loss += arc_loss
            p_arc = self.attn_drop(self.softmax(s_arc))
            h_arc_new, d_arc_new = self.H_GAT(h_arc, d_arc, p_arc), self.D_GAT(d_arc, h_arc, p_arc)
            h_arc, d_arc = self.norm_h[i](h_arc_new + h_arc), self.norm_d[i](d_arc_new + d_arc)

        s_arc = self.arc_attn_last(d_arc, h_arc)
        s_arc.masked_fill_(~x['mask'].unsqueeze(1), float('-inf'))
        arc_loss = self.calu_arc_loss(s_arc, x, prob, loss_weight, gold_arc_loss)
        arc_loss = self.LAMBDA[0]*gnn_loss + self.LAMBDA[1]*arc_loss
        # Rel Bi-affine Layer
        s_rel = self.rel_attn(d_rel, h_rel).permute(0, 2, 3, 1)
        s_rel.masked_fill_(~x['mask'].unsqueeze(1).unsqueeze(3), float('-inf'))

        # Calc CE_Loss
        pred_rel, rel_loss = self.calu_rel_loss(s_rel, x, n_token, prob, loss_weight, gold_arc)
        if self.training: return arc_loss, rel_loss

        # pred_arc = pred_arc_.argmax(-1)
        pred_arc = [MST_inference(p, l, m)[m][1:] for p, l, m in zip(
            self.softmax(s_arc).cpu().numpy(),
            lens.unsqueeze(1).cpu().numpy(),
            x['mask'].cpu().numpy())]
        pred_arc = torch.tensor(np.concatenate(pred_arc), dtype=torch.long, device=torch.device("cuda"))
        pred_rel = pred_rel[n_token, pred_arc].argmax(-1)
        return arc_loss, rel_loss, pred_arc.tolist(), pred_rel.tolist()
