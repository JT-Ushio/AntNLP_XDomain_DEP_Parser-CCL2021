from typing import Callable, List, Dict
from overrides import overrides
import re, sys
from collections import Counter
from antu.io import Instance, DatasetReader
from antu.io.fields import Field, TextField, IndexField, FloatField
from antu.io.token_indexers import TokenIndexer, SingleIdTokenIndexer, CharTokenIndexer


def char_transform(x: str):
    if x == '**root**':  # indicate root token
        return ['<B>', '**root**', '<E>']
    res = ['<B>'] + list(x) + ['<E>']
    if len(res) > 32:  # MAX_CHAR_LENGTH = 30
        res = res[:16] + res[-16:]
    return res


class PTBReader(DatasetReader):

    def __init__(self, field_list: List[str], root: str, spacer: str, min_prob: float):
        self.field_list, self.root, self.spacer, self.min_prob = field_list, root, spacer, min_prob

    def _read(self, file_path: str) -> Instance:
        with open(file_path, 'rt') as fp:
            root_token = re.split(self.spacer, self.root)
            tokens = [[item,] for item in root_token]
            for line in fp:
                token = re.split(self.spacer, line.strip())
                if line.strip() == '':
                    if len(tokens[0]) > 1: yield tokens
                    tokens = [[item,] for item in root_token]
                else:
                    if token[6] == '-1' and token[7] == 'none':
                        token[6], token[7], token[9] = '0', '**rrel**', '0.0'
                    try:
                        if float(token[9]) > 1.5: token[9] = '1.0'
                    except ValueError:
                        token[9] = '1.0'
                    for idx, item in enumerate(token):
                        tokens[idx].append(item)
            if len(tokens[0]) > 1: yield tokens

    @overrides
    def read(self, file_path: str) -> List[Instance]:
        # Build indexers
        indexers = dict()
        word_indexer = SingleIdTokenIndexer(['word', 'glove'])
        char_indexer = CharTokenIndexer(['char'], char_transform)
        indexers['word'] = [word_indexer, char_indexer]
        tag_indexer = SingleIdTokenIndexer(['tag'])
        indexers['tag'] = [tag_indexer,]
        rel_indexer = SingleIdTokenIndexer(['rel'])
        indexers['rel'] = [rel_indexer,]

        # Build instance list
        res = []
        for sentence in self._read(file_path):
            res.append(self.input_to_instance(sentence, indexers))
        return res

    @overrides
    def input_to_instance(
        self,
        inputs: List[List[str]],
        indexers: Dict[str, List[TokenIndexer]]) -> Instance:
        fields = []
        if 'word' in self.field_list:
            fields.append(TextField('word', inputs[1], indexers['word']))
        if 'tag' in self.field_list:
            fields.append(TextField('tag', inputs[3], indexers['tag']))
        if 'head' in self.field_list:
            fields.append(IndexField('head', inputs[6]))
        if 'rel' in self.field_list:
            fields.append(TextField('rel', inputs[7], indexers['rel']))
        if 'prob' in self.field_list:
            fields.append(FloatField('prob', inputs[9]))
        return Instance(fields)