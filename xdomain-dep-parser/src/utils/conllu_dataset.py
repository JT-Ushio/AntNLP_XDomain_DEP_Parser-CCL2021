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
    max_char_len = max([len(char) for ins in batch for char in ins['word']['char']])
    PAD, UNK = CoNLLUDataset.PAD, CoNLLUDataset.UNK
    pad_tok_seq, zero_pad_seq = [PAD]*max_len, [0]*max_len
    pad_char_seq = [PAD]*max_char_len
    truth = {'head': [], 'rel': []}
    inputs = {'word': [], 'glove': [], 'tag': [], 'prob': [], 'char': [], 'word2char': []}
    word_dict = {PAD: 0}
    tmp = []
    for ins in batch:
        pad_len = max_len-len(ins['head'])
        # PAD word
        inputs['word'].append(ins['word']['word']+pad_tok_seq[:pad_len])
        tmp.extend(ins['word']['word'])
        # PAD glove
        glove_idxs = [x for x in ins['word']['glove']]
        inputs['glove'].append(glove_idxs+pad_tok_seq[:pad_len])
        # PAD tag
        inputs['tag'].append(ins['tag']['tag']+pad_tok_seq[:pad_len])
        # PAD head
        truth['head'].extend(ins['head']+zero_pad_seq[:pad_len])
        # PAD prob
        inputs['prob'].extend(ins['prob'][1:])
        # PAD rel
        truth['rel'].extend(ins['rel']['rel']+zero_pad_seq[:pad_len])
        # PAD char
        char_ids = []
        for word, char in zip(ins['word']['word'], ins['word']['char']):
            if word not in word_dict:
                word_dict[word] = len(word_dict)
                inputs['char'].append(char+pad_char_seq[:max_char_len-len(char)])
            char_ids.append(word_dict[word])
        inputs['word2char'].append(char_ids+zero_pad_seq[:pad_len])


    device = torch.device("cuda" if not get_worker_info() and torch.cuda.is_available() else "cpu")
    res = {}
    res['w_lookup'] = torch.tensor(inputs['word'], dtype=torch.long, device=device)
    res['g_lookup'] = torch.tensor(inputs['glove'], dtype=torch.long, device=device)
    res['t_lookup'] = torch.tensor(inputs['tag'], dtype=torch.long, device=device)
    res['c_lookup'] = torch.tensor(inputs['char'], dtype=torch.long, device=device)
    res['w2c'] = torch.tensor(inputs['word2char'], dtype=torch.long, device=device)
    # print(res['c_lookup'].size(), res['t_lookup'].size(), res['w2c'].size())
    res['prob'] = torch.tensor(inputs['prob'], dtype=torch.float, device=device)
    res['head'] = torch.tensor(truth['head'], dtype=torch.long, device=device)
    res['rel'] = torch.tensor(truth['rel'], dtype=torch.long, device=device)
    res['mask'] = res['w_lookup'].ne(PAD)
    return res

