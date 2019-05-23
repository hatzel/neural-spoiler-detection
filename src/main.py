from datasets import PaddedBatch
from datasets import TvTropesDataset
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import metrics
from torch.utils.data import DataLoader


def main():
    tokenizer, classifier = train()
    test(tokenizer, classifier)


def train():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = TvTropesDataset(
        "../master-thesis/data/boyd2013spoiler/train.balanced.csv",
        tokenizer,
    )
    classifier = BertForSequenceClassification\
        .from_pretrained("bert-base-uncased", num_labels=2).cuda()
    for epoch in range(3):
        loader = DataLoader(dataset, batch_size=8, collate_fn=PaddedBatch, shuffle=True)
        optimizer = torch.optim.Adam(classifier.parameters(recurse=True), lr=1 * 10**-5)
        for batch in tqdm(loader):
            optimizer.zero_grad()
            output = classifier(
                batch.token_ids.cuda(),
                batch.sequence_ids.cuda(),
                batch.input_mask.cuda()
            )
            loss = F.cross_entropy(output, batch.labels.cuda())
            loss.backward()
            optimizer.step()
    return tokenizer, classifier


def test(tokenizer, classifier):
    dataset = TvTropesDataset(
        "../master-thesis/data/boyd2013spoiler/dev1.balanced.csv",
        tokenizer,
    )
    loader = DataLoader(dataset, batch_size=8, collate_fn=PaddedBatch)
    labels = []
    predicted = []
    for batch in tqdm(loader):
        with torch.no_grad():
            output = classifier(
                batch.token_ids.cuda(),
                batch.sequence_ids.cuda(),
                batch.input_mask.cuda()
            )
            labels.extend(batch.labels)
            predicted.extend(list(output.argmax(1)))
    labels, predicted = ([t.item() for t in labels], [t.item() for t in predicted])
    # accuracy = metrics.accuracy_score(labels, predicted)
    # f1 = metrics.f1_score(labels, predicted)
    # confusion_matrix = metrics.confusion_matrix(labels, predicted)
    report = metrics.classification_report(labels, predicted)
    print(report)


if __name__ == "__main__":
    main()
