import csv
import json
import conllu
import gzip
from enum import Enum
import torch
import torch.nn.utils.rnn as rnn
import torch.utils.data
from dataclasses import dataclass
from tqdm import tqdm
from typing import List
from transformers import BertTokenizer
import xml.etree.ElementTree as ElementTree

MAX_SIZE_CONTENT_TOKENS = 510


class FileType(Enum):
    JSON = 1
    CSV = 2
    CONLLU = 3


class Task:
    def __init__(self):
        pass

    def training_set():
        pass

    def dev_set():
        pass


def open_maybe_gzip(file_name):
    try:
        f = gzip.open(file_name, "rt")
        f.readlines(10)
        f = gzip.open(file_name, "rt")
    except OSError:
        f = open(file_name, "r")
    return f


def guess_format(file_name, limit=10):
    f = open_maybe_gzip(file_name)
    first_n_lines = []
    for _, line in zip(range(limit), f):
        first_n_lines.append(line)
    if all(
        line.startswith("{") and "}" for line in first_n_lines
    ):
        return FileType.JSON
    elif first_n_lines[0].startswith("# global.columns"):
        return FileType.CONLLU
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
    full_token_ids: torch.Tensor
    sentence_ids: torch.Tensor
    labels: torch.Tensor
    full_labels: torch.Tensor
    conll_ids: torch.Tensor # the id in the CoNLL dataset (just as long as full token_ids)
    conll_labels: torch.Tensor # the label in the CoNLL dataset

    def __len__(self):
        return len(self.token_ids)


class BinarySpoilerDataset(torch.utils.data.Dataset):
    def __init__(self, file_names: str, tokenizer: BertTokenizer, limit=None):
        self.file_names = file_names
        self.tokenizer = tokenizer
        self.labels: List[bool] = []
        self.clipped_count = 0
        self.saved_data = {}
        n = 0
        for file_name in file_names:
            self.format = guess_format(file_name)
            with open_maybe_gzip(file_name) as file:
                if self.format == FileType.CSV:
                    reader = csv.DictReader(file)
                    for line in tqdm(reader, desc=f"Loading csv dataset {file_name}"):
                        if n == limit:
                            break
                        self.saved_data[str(n)] = self.to_feature(
                            line["sentence"],
                            second_sequence=line.get("story_text")[1:-1],
                        )
                        self.labels.append(
                            True if line["spoiler"] == "True" else False
                        )
                        n += 1
                elif self.format == FileType.JSON:
                    for line in tqdm(file, desc=f"Loading json dataset {file_name}"):
                        if n == limit:
                            break
                        data = json.loads(line)
                        self.saved_data[str(n)] = self.to_feature(data["text"])
                        self.labels.append(data["spoiler"])
                        n += 1
        print(f"Clipped {self.clipped_count} posts.")
        super(BinarySpoilerDataset, self).__init__()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.saved_data[str(index)],
            torch.tensor(self.labels[index].clone().detach(), dtype=torch.bool)
            if type(self.labels[index]) != bool else
            torch.tensor(self.labels[index], dtype=torch.bool)
        )

    def to_feature(self, text, second_sequence=None) -> TvTropesFeature:
        if second_sequence:
            self.to_feature_trim_second(text, second_sequence=second_sequence)
        end_first_token_sample = 128
        start_second_token_sample = MAX_SIZE_CONTENT_TOKENS - end_first_token_sample
        max_tokens = MAX_SIZE_CONTENT_TOKENS
        if second_sequence is not None:
            end_first_token_sample = int(end_first_token_sample / 2)
            start_second_token_sample = int(start_second_token_sample / 2)
            max_tokens /= 2
        token_ids = []
        sentence_ids = []
        sentences = [text, second_sequence] \
            if second_sequence is not None \
            else [text]
        for i, sentence in enumerate(sentences):
            tokenized_text = self.tokenizer.tokenize(sentence)
            if i == 1 and len(tokenized_text) > max_tokens:
                self.clipped_count += 1
            new_tokens = []
            # Apply head + tail truncation as suggested in:
            # "How to fine tune Bert for text classification?"
            if i == 0:
                new_tokens.append("[CLS]")
            if max_tokens <= len(tokenized_text):
                new_tokens.extend(tokenized_text[:end_first_token_sample])
                new_tokens.extend(tokenized_text[-start_second_token_sample:])
            else:
                new_tokens.extend(tokenized_text)
            new_tokens.append("[SEP]")
            new_token_ids = self.tokenizer.convert_tokens_to_ids(new_tokens)
            token_ids += new_token_ids
            sentence_ids.append(torch.tensor([i for _ in new_token_ids]))
        return TvTropesFeature(
            token_ids=token_ids,
            sentence_ids=torch.cat(sentence_ids, -1),
        )

    def to_feature_trim_second(self, text, second_sequence=None) -> TvTropesFeature:
        # sentences = [text, second_sequence]
        tokens = ["[CLS]"]
        tokenized_text = self.tokenizer.tokenize(text)
        tokens.extend(tokenized_text)
        tokens.append("[SEP]")
        first_len = len(tokens)
        to_go = MAX_SIZE_CONTENT_TOKENS - first_len
        second_tokenized = self.tokenizer.tokenize(second_sequence)
        tokens.extend(second_tokenized[:to_go])
        sentence_ids = ([torch.unsqueeze(torch.tensor(0), -1)] * to_go) + \
                ([torch.unsqueeze(torch.tensor(1), -1)] * len(second_sequence[:to_go]))
        return TvTropesFeature(
            token_ids=self.tokenizer.convert_tokens_to_ids(tokens),
            sentence_ids=torch.cat(sentence_ids),
        )


class TokenSpoilerDataset(BinarySpoilerDataset):
    def  __init__(self, file_names: str, tokenizer: BertTokenizer, limit=None):
        self.file_names = file_names
        self.tokenizer = tokenizer
        self.labels: List[torch.Tensor] = []
        self.clipped_count = 0
        self.saved_data = {}
        n = 0
        for file_name in file_names:
            format = guess_format(file_name)
            with open_maybe_gzip(file_name) as file:
                parser = conllu.parse_incr(file)
                if format != FileType.CONLLU:
                    raise Exception(f"Unsupported filetype {format}")
                for sentence in tqdm(parser,
                                 desc=f"Loading conllu dataset {file_name}"):
                    if n == limit:
                        break
                    self.saved_data[str(n)] = self.to_feature(sentence)
                    self.labels.append(self.saved_data[str(n)].labels)
                    n += 1
            print(f"Clipped {self.clipped_count} posts.")
        super(BinarySpoilerDataset, self).__init__()

    def to_feature(self, sentence) -> TokenFeature:
        tokens = ["[CLS]"]
        full_spoiler_bools = []
        tokenized_text = []
        conll_ids = []
        conll_labels = []
        for whitespace_token in sentence:
            tokenized = self.tokenizer.tokenize(
                whitespace_token["form"]
            )
            label = bool(int(whitespace_token["reddit_spoilers:spoiler"]))
            conll_labels.append(
                label
            )
            for token in tokenized:
                tokenized_text.append(token)
                full_spoiler_bools.append(label)
                conll_ids.append(
                    int(whitespace_token["id"])
                )
        if len(tokenized_text) > MAX_SIZE_CONTENT_TOKENS:
            self.clipped_count += 1
        spoiler_bools = [False] + full_spoiler_bools[:MAX_SIZE_CONTENT_TOKENS] + [False]
        tokens.extend(tokenized_text[:MAX_SIZE_CONTENT_TOKENS])
        tokens.append("[SEP]")
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        full_token_ids = self.tokenizer.convert_tokens_to_ids(
            tokenized_text
        )
        sentence_ids = torch.tensor([0 for _ in token_ids])
        return TokenFeature(
            token_ids=token_ids,
            full_token_ids=full_token_ids,
            sentence_ids=sentence_ids,
            labels=torch.tensor(spoiler_bools, dtype=torch.bool),
            full_labels=torch.tensor(full_spoiler_bools),
            conll_ids=torch.tensor(conll_ids),
            conll_labels=torch.tensor(conll_labels),
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
        if hasattr(transposed[0][0], "full_labels"):
            self.full_labels = [feature.full_labels for feature in transposed[0]]

        if hasattr(transposed[0][0], "conll_labels"):
            self.conll_labels = [feature.conll_labels for feature in transposed[0]]
            self.conll_ids = [feature.conll_ids for feature in transposed[0]]

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

    def to_full_prediction(self, tensor, fill):
        result = []
        for to_mask, mask, full_labels in zip(tensor, self.input_mask, self.full_labels):
            goal_len = len(full_labels)
            try:
                index = (mask == 0).nonzero().reshape(-1)[0]
            except IndexError:
                index = len(mask)
            filling = fill.repeat(goal_len - int(index - 2)).reshape(-1, 1).cuda()
            result.append(torch.cat([to_mask[1:index - 1], filling]))

        return result

    def to_full_prediction_merged(self, tensor, fill):
        """
        Convert to prediction, combining entries that belong to multiple conll ids
        """
        result = self.to_full_prediction(tensor, fill)
        merged_result = []
        for sentence, ids, labels \
                in zip(result, self.conll_ids, self.conll_labels):
            try:
                max_id = len(labels)
            except IndexError:
                merged_result.append(torch.tensor([]))
                continue
            pooling = torch.zeros(max_id).float()
            counts = torch.zeros(max_id).float()
            for score, id in zip(sentence, ids):
                counts[id - 1] += 1
                pooling[id - 1] += score.squeeze()
            avg_pooled = torch.div(pooling, counts)
            avg_pooled[torch.isnan(avg_pooled)] = 0
            merged_result.append(avg_pooled)
        return merged_result


    def merged_tokens(self, i, tokenizer):
        """
        Return list of strings with tokens combined according to the conll_ids.
        """
        out = []
        counts = []
        last_id = None
        tokens = tokenizer.convert_ids_to_tokens(self.token_ids[i].tolist())
        for token, current_id in zip(tokens[1:-1], self.conll_ids[i]):
            if last_id == current_id.item():
                if token.startswith("##"):
                    out[-1] += (token[2:])
                else:
                    out[-1] += token
                counts[-1] += 1
            else:
                out.append(token)
                counts.append(1)
            last_id = current_id
        return out, counts


    def __repr__(self):
        return f"<PaddedBatch labels={self.labels}>"
