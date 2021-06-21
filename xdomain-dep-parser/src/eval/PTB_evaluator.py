import os
import logging
from typing import List, Tuple

import torch
from antu.io import Vocabulary

logger = logging.getLogger('__main__')


def ptb_evaluation(
    vocab: Vocabulary,
    pred: List[Tuple[torch.Tensor, torch.Tensor]],
    pred_path:str=None,
    gold_path:str=None) -> Tuple[str, str]:

    n_rel = vocab.get_vocab_size('rel')
    i2r = [vocab.get_token_from_index(i, 'rel') for i in range(n_rel)]
    arcs, rels = pred['arcs'], pred['rels']

    with open(pred_path, 'w') as fout, open(gold_path, 'r') as fin:
        i = 0
        for line in fin:
            if line.strip() == '':
                fout.write(line)
            else:
                ins = line.strip().split('\t')
                ins[6], ins[7] = str(arcs[i]), i2r[rels[i]]
                pred_line = '\t'.join(ins) + '\n'
                fout.write(pred_line)
                i += 1
    res = os.popen(f'python eval/evaluate.py {gold_path} {pred_path}')
    res.readline()
    result = res.readline()
    UAS = float(result.split()[4][:-1])
    LAS = float(result.split()[-1][1:])
    # res = os.popen(f'perl eval/eval.pl -q -g {gold_path} -s {pred_path}')
    # LAS = float(res.readline().split()[-2])
    # UAS = float(res.readline().split()[-2])
    return UAS, LAS
