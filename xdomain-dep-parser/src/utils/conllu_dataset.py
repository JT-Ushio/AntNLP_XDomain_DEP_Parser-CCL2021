from typing import Set, List
from itertools import cycle
from collections import Counter

import torch
from torch.utils.data import Dataset, get_worker_info
import numpy as np
from antu.io import Vocabulary, DatasetReader


class CoNLLUDataset(Dataset):

    def __init__(
        self,
        file_path: str,
        data_reader: DatasetReader,
        vocabulary: Vocabulary,
        counters: Counter = None,
        min_count: int = 0,
        no_pad_namespace: Set[str] = None,
        no_unk_namespace: Set[str] = None):

        self.data = data_reader.read(file_path)
        if counters:
            for ins in self.data:
                ins.count_vocab_items(counters)
            vocabulary.extend_from_counter(
                counters, min_count, no_pad_namespace, no_unk_namespace)
            CoNLLUDataset.PAD = vocabulary.get_padding_index('word')
            CoNLLUDataset.UNK = vocabulary.get_unknow_index('word')
        self.vocabulary = vocabulary

    def __getitem__(self, idx: int):
        return self.data[idx].index_fields(self.vocabulary)

    def __len__(self):
        return len(self.data)


def conllu_fn(batch):
    max_len = max([len(ins['head']) for ins in batch])
    PAD, UNK = CoNLLUDataset.PAD, CoNLLUDataset.UNK
    truth = {'head': [], 'rel': []}
    inputs = {'word': [], 'glove': [], 'tag': [], 'prob': []}
    tmp = []
    for ins in batch:
        pad_len = max_len-len(ins['head'])
        pad_seq = [PAD] * pad_len
        # PAD word
        inputs['word'].append(ins['word']['word']+pad_seq)
        tmp.extend(ins['word']['word'])
        # PAD glove
        glove_idxs = [x for x in ins['word']['glove']]
        inputs['glove'].append(glove_idxs+pad_seq)
        # PAD tag
        inputs['tag'].append(ins['tag']['tag']+pad_seq)
        # PAD head
        pad_seq = [0] * pad_len
        truth['head'].extend(ins['head']+pad_seq)
        # PAD prob
        inputs['prob'].extend(ins['prob'][1:])
        # PAD rel
        truth['rel'].extend(ins['rel']['rel']+pad_seq)

    device = torch.device("cuda" if not get_worker_info() and torch.cuda.is_available() else "cpu")
    res = {}
    res['w_lookup'] = torch.tensor(inputs['word'], dtype=torch.long, device=device)
    res['g_lookup'] = torch.tensor(inputs['glove'], dtype=torch.long, device=device)
    res['t_lookup'] = torch.tensor(inputs['tag'], dtype=torch.long, device=device)
    res['prob'] = torch.tensor(inputs['prob'], dtype=torch.bool, device=device)
    res['head'] = torch.tensor(truth['head'], dtype=torch.long, device=device)
    res['rel'] = torch.tensor(truth['rel'], dtype=torch.long, device=device)
    res['mask'] = res['w_lookup'].ne(PAD)
    return res

