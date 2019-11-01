import sys
from datasets import TokenSpoilerDataset, PaddedBatch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from metrics import sequence_lengths
import pandas
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

    lengths = []

    for batch in loader:
        lengths.extend(sequence_lengths(batch.full_labels))

df = pandas.DataFrame(lengths, columns=["Sequence Length"])
print(df.describe())
counts = Counter(lengths)

f = open("sequence_length_counts.csv", "w")
for k, v in sorted(counts.items(), key=lambda kv: kv[0]):
    f.write(f"{k},{v}\n")
