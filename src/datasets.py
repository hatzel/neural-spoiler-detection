import csv
import json
from enum import Enum
import torch
import torch.nn.utils.rnn as rnn
import torch.utils.data
from dataclasses import dataclass
from tqdm import tqdm
from typing import List
from transformers import BertTokenizer
import xml.etree.ElementTree as ElementTree


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


def iter_text_is_spoiler(root):
    for el in root.iter():
        if el.tag == "spoiler":
            yield (el.text, True)
        else:
            yield (el.text, False)
        if el.tail:
            yield (el.tail, False)


@dataclass
class TvTropesFeature:
    token_ids: torch.Tensor
    sentence_ids: torch.Tensor

    def __len__(self):
        return len(self.token_ids)


@dataclass
class TokenFeature:
    token_ids: torch.Tensor
    sentence_ids: torch.Tensor
    labels: torch.Tensor

    def __len__(self):
        return len(self.token_ids)


class BinarySpoilerDataset(torch.utils.data.Dataset):
    def __init__(self, file_name: str, tokenizer: BertTokenizer, limit=None):
        self.file_name: str = file_name
        self.tokenizer = tokenizer
        self.labels: List[bool] = []
        self.clipped_count = 0
        format = guess_format(file_name)
        self.saved_data = {}
        with open(file_name, "r") as file:
            if format == FileType.CSV:
                reader = csv.reader(file)
                for n, line in enumerate(
                    tqdm(reader, desc="Loading json dataset")
                ):
                    if n == limit:
                        break
                    self.saved_data[str(n)] = self.to_feature(line[0])
                    self.labels.append(True if line[1] == "True" else False)
            else:
                for n, line in enumerate(
                    tqdm(file, desc="Loading json dataset")
                ):
                    if n == limit:
                        break
                    data = json.loads(line)
                    self.saved_data[str(n)] = self.to_feature(data["text"])
                    self.labels.append(data["spoiler"])
        print(f"Clipped {self.clipped_count} posts.")
        super(BinarySpoilerDataset, self).__init__()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.saved_data[str(index)],
            torch.tensor(self.labels[index], dtype=torch.bool)
        )

    def to_feature(self, text) -> TvTropesFeature:
        tokens = ["[CLS]"]
        tokenized_text = self.tokenizer.tokenize(text)
        if len(tokenized_text) > 498:
            self.clipped_count += 1
        tokens.extend(tokenized_text[:498])
        tokens.append("[SEP]")
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        sentence_ids = torch.tensor([0 for _ in token_ids])
        return TvTropesFeature(
            token_ids=token_ids,
            sentence_ids=sentence_ids,
        )


class TokenSpoilerDataset(BinarySpoilerDataset):
    def  __init__(self, file_name: str, tokenizer: BertTokenizer, limit=None):
        self.file_name: str = file_name
        self.tokenizer = tokenizer
        self.labels: List[torch.Tensor] = []
        self.clipped_count = 0
        self.saved_data = {}
        if file_name:
            with open(file_name, "r") as file:
                for n, line in enumerate(
                    tqdm(file, desc="Loading json dataset")
                ):
                    if n == limit:
                        break
                    data = json.loads(line)
                    self.saved_data[str(n)] = self.to_feature(data["text"])
                    self.labels.append(self.saved_data[str(n)].labels)
            print(f"Clipped {self.clipped_count} posts.")
        super(BinarySpoilerDataset, self).__init__()

    def to_feature(self, text) -> TokenFeature:
        tokens = ["[CLS]"]
        # Normalize xml parser chokes on this
        text = text.replace('\x10', '')
        try:
            root = ElementTree.fromstring(f"<data>{text}</data>")
        except ElementTree.ParseError:
            print(f"Error parsing comment: {text}")
        spoiler_bools = []
        tokenized_text = []
        for el_text, is_spoiler in iter_text_is_spoiler(root):
            if el_text:
                tokenized_el = self.tokenizer.tokenize(el_text)
                for word in tokenized_el:
                    tokenized_text.append(word)
                    spoiler_bools.append(is_spoiler)
        if len(tokenized_text) > 498:
            self.clipped_count += 1
        spoiler_bools = [False] + spoiler_bools[:498] + [False]
        tokens.extend(tokenized_text[:498])
        tokens.append("[SEP]")
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        sentence_ids = torch.tensor([0 for _ in token_ids])
        return TokenFeature(
            token_ids=token_ids,
            sentence_ids=sentence_ids,
            labels=torch.tensor(spoiler_bools, dtype=torch.bool),
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

        if len(transposed[1][0].shape) == 0:
            self.labels = torch.stack(transposed[1], 0)
        else:
            size = max(len(el) for el in transposed[1])
            self.labels = torch.zeros(len(transposed[1]), size, dtype=torch.bool)
            for i, el in enumerate(transposed[1]):
                self.labels[i][:len(el)] = el

    def mask(self, tensor):
        result = []
        for to_mask, mask in zip(tensor, self.input_mask):
            try:
                index = (mask == 0).nonzero().reshape(-1)[0]
                result.append(to_mask[:index])
            except IndexError:
                result.append(to_mask)
        return result

    def __repr__(self):
        return f"<PaddedBatch labels={self.labels}>"
