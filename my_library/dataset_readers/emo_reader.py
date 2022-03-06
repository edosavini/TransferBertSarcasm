from typing import Dict
import logging
import csv

from overrides import overrides

from allennlp.data.tokenizers.token import Token
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from transformers import BertTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("emo_reader")
class EmotionDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 seq_len: int = 512,
                 bert_model_name: str = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        if bert_model_name:
            self._tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        else:
            self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.seq_len = seq_len
        self.labels = list()

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as tsvfile:
            logger.info("Reading instances from lines in file at: %s", file_path)
            reader = csv.reader(tsvfile, delimiter='\t')

            for line in reader:
                response = line[0]
                response = response.rsplit('#', 1)[0]
                label = line[1]
                if label not in self.labels:
                    self.labels.append(label)
                yield self.text_to_instance(response, label)

    @overrides
    def text_to_instance(self, response: str, label: str = None) -> Instance:
        response = self._tokenizer.tokenize(response)
        tokenized_response = []
        for w in response:
            tokenized_response.append(Token(w))
        rf = TextField(tokenized_response, self._token_indexers)

        fields = {'quote_response': rf}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)
