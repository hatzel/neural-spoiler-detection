import sys
from datasets import TokenSpoilerDataset, PaddedBatch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from metrics import sequence_lengths
import pandas
import torch
from collections import Counter

file_names = sys.argv[1:]

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


print(file_names)
for name in file_names:
    train_dataset = TokenSpoilerDataset(
        [name],
        tokenizer,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=32,
        collate_fn=PaddedBatch,
        shuffle=True
    )

    lengths_spoilers = []
    lengths_non_spoilers = []

    for batch in loader:
        lengths_spoilers.extend(sequence_lengths(batch.full_labels, only_for_classes=[torch.tensor(1)]))
        lengths_non_spoilers.extend(sequence_lengths(batch.full_labels, only_for_classes=[torch.tensor(0)]))

for lengths, name in [(lengths_spoilers, "spoilers"), (lengths_non_spoilers, "non-spoilers")]:
    df = pandas.DataFrame(lengths, columns=["Sequence Length"])
    print(df.describe())
    counts = Counter(lengths)

    f = open(f"sequence_length_counts_{name}.csv", "w")
    for k, v in sorted(counts.items(), key=lambda kv: kv[0]):
        f.write(f"{k},{v}\n")
