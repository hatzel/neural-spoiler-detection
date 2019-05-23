from datasets import PaddedBatch
from datasets import TvTropesDataset
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader


def main():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = TvTropesDataset(
        "../master-thesis/data/boyd2013spoiler/train.balanced.csv",
        tokenizer,
    )
    loader = DataLoader(dataset, batch_size=8, collate_fn=PaddedBatch)
    classifier = BertForSequenceClassification\
        .from_pretrained("bert-base-uncased", num_labels=2)
    for batch in loader:
        with torch.no_grad():
            print(classifier(batch.token_ids, batch.sequence_ids, batch.input_mask))


def find_vocab(dataset, tokenizer):
    for sentence in dataset.texts:
        tokenizer.tokenize(sentence)
    tokenizer.save_vocabulary("vocab")


if __name__ == "__main__":
    main()
