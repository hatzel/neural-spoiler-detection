import csv
import json
from enum import Enum
import torch
import torch.nn.utils.rnn as rnn
import torch.utils.data
from functools import lru_cache
from dataclasses import dataclass
from tqdm import tqdm
from typing import List
from pytorch_pretrained_bert import BertTokenizer


class FileType(Enum):
    JSON = 1
    CSV = 2


class Task:
    def __init__(self):
        pass

    def training_set():
        pass

    def dev_set():
        pass


def guess_format(file_name, limit=10):
    f = open(file_name, "r")
    if all(
        line.startswith("{") and "}" in line
        for _, line in zip(range(limit), f)
    ):
        return FileType.JSON
    else:
        return FileType.CSV


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
        format = guess_format(file_name)
        self.saved_data = {}
        with open(file_name, "r") as file:
            if format == FileType.CSV:
                reader = csv.reader(file)
                for n, line in enumerate(
                    tqdm(reader, desc="Loading json dataset")
                ):
                    self.saved_data[str(n)] = self.to_feature(line[0]),
                    self.labels.append(True if line[1] == "True" else False)
            else:
                for n, line in enumerate(
                    tqdm(file, desc="Loading json dataset")
                ):
                    data = json.loads(line)
                    self.saved_data[str(n)] = self.to_feature(data["text"]),
                    self.labels.append(data["spoiler"])
        super(BinarySpoilerDataset, self).__init__()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.saved_data[str(index)][0],
            torch.tensor(self.labels[index], dtype=torch.long)
        )

    def to_feature(self, text) -> TvTropesFeature:
        tokens = ["[CLS]"]
        tokens.extend(self.tokenizer.tokenize(text)[:498])
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
