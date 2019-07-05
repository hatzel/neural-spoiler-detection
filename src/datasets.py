import csv
import torch
import torch.nn.utils.rnn as rnn
import torch.utils.data
from functools import lru_cache
from dataclasses import dataclass
from typing import List
from pytorch_pretrained_bert import BertTokenizer


class Task:
    def __init__(self):
        pass

    def training_set():
        pass

    def dev_set():
        pass


@dataclass
class TvTropesFeature:
    token_ids: torch.Tensor
    sentence_ids: torch.Tensor

    def __len__(self):
        return len(self.token_ids)


class BinarySpoilerDataset(torch.utils.data.Dataset):
    def __init__(self, file_name: str, tokenizer: BertTokenizer):
        self.file_name: str = file_name
        self.tokenizer = tokenizer
        self.labels: List[bool] = []
        self.texts: List[str] = []
        with open(file_name, "r") as file:
            reader = csv.reader(file)
            for line in reader:
                self.texts.append(line[0])
                self.labels.append(True if line[1] == "True" else False)
        super(BinarySpoilerDataset, self).__init__()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.to_feature(self.texts[index]),
                torch.tensor(self.labels[index], dtype=torch.long))

    @lru_cache(2 ** 14)
    def to_feature(self, text) -> TvTropesFeature:
        tokens = ["[CLS]"]
        tokens.extend(self.tokenizer.tokenize(text))
        tokens.append("[SEP]")
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        sentence_ids = torch.tensor([0 for _ in token_ids])
        return TvTropesFeature(
            token_ids=token_ids,
            sentence_ids=sentence_ids,
        )


class PaddedBatch:
    def __init__(self, data):
        transposed = list(zip(*data))
        self.token_ids = rnn.pad_sequence(
            [torch.tensor(feature.token_ids, dtype=torch.long)
             for feature in transposed[0]],
            batch_first=True)
        self.sequence_ids = rnn.pad_sequence(
            [feature.sentence_ids for feature in transposed[0]],
            batch_first=True,
        )

        # Initialize input mask with ones
        self.input_mask = torch.ones(
            len(transposed[0]), max(len(el) for el in transposed[0]))
        # zero out those values that are just padding
        for i, el in enumerate(transposed[0]):
            self.input_mask[i][len(el):] = 0

        self.labels = torch.stack(transposed[1], 0)

    def __repr__(self):
        return f"<PaddedBatch labels={self.labels}>"
