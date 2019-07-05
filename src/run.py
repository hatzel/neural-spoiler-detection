import torch
from tqdm import tqdm

from pytorch_pretrained_bert import (
    BertTokenizer,
    BertForSequenceClassification
)
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn import metrics

from result import Result
from datasets import BinarySpoilerDataset, PaddedBatch
import util


class BertRun():
    def __init__(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.classifier = BertForSequenceClassification\
            .from_pretrained("bert-base-uncased", num_labels=2).cuda()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.training_parameters = []

    def train(self, batch_size=8, lr=1 * 10 ** -5, num_epochs=3, seed=None):
        if seed is not None:
            util.seed(seed)
        self.training_parameters.append({
            "batch_size": batch_size,
            "lr": lr,
            "num_epochs": num_epochs,
            "seed": seed,
        })
        for epoch in range(num_epochs):
            loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                collate_fn=PaddedBatch,
                shuffle=True
            )
            optimizer = torch.optim.Adam(
                self.classifier.parameters(recurse=True),
                lr=lr
            )
            for batch in tqdm(loader):
                optimizer.zero_grad()
                output = self.classifier(
                    batch.token_ids.cuda(),
                    batch.sequence_ids.cuda(),
                    batch.input_mask.cuda()
                )
                loss = F.cross_entropy(output, batch.labels.cuda())
                loss.backward()
                optimizer.step()

    def test(self):
        loader = DataLoader(self.test_dataset, batch_size=8,
                            collate_fn=PaddedBatch)
        labels = []
        predicted = []
        for batch in tqdm(loader):
            with torch.no_grad():
                output = self.classifier(
                    batch.token_ids.cuda(),
                    batch.sequence_ids.cuda(),
                    batch.input_mask.cuda()
                )
                labels.extend(batch.labels)
                predicted.extend(list(output.argmax(1)))
        labels, predicted = (
            [t.item() for t in labels],
            [t.item() for t in predicted]
        )
        # accuracy = metrics.accuracy_score(labels, predicted)
        # f1 = metrics.f1_score(labels, predicted)
        # confusion_matrix = metrics.confusion_matrix(labels, predicted)
        print(metrics.classification_report(labels, predicted))
        report = metrics.classification_report(
            labels, predicted, output_dict=True)
        return Result(
            training_parameters=self.training_parameters,
            train_dataset_path=self.train_dataset.file_name,
            test_dataset_path=self.test_dataset.file_name,
            model=self.classifier,
            report=report,
        )

    @staticmethod
    def from_file(model_path, train_path, test_path):
        run = BertRun.for_dataset(train_path, test_path)
        data = torch.load(model_path)
        run.classifier.load_state_dict(data)
        return run

    @staticmethod
    def for_dataset(train_path, test_path):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        train_dataset = BinarySpoilerDataset(
            train_path,
            tokenizer,
        )
        test_dataset = BinarySpoilerDataset(
            test_path,
            tokenizer,
        )
        return BertRun(train_dataset, test_dataset)
