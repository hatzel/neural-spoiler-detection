import torch
from tqdm import tqdm

from pytorch_pretrained_bert import (
    BertTokenizer,
    BertForSequenceClassification,
    BertForTokenClassification,
)
from pytorch_pretrained_bert.optimization import BertAdam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn import metrics
from apex import amp

from result import Result
from datasets import BinarySpoilerDataset, TokenSpoilerDataset, PaddedBatch
import util


class BertRun():
    def __init__(self, train_dataset, test_dataset, base_model, token_based=False):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.token_based = token_based
        bert_model = (
            BertForTokenClassification
            if token_based
            else BertForSequenceClassification
        )
        self.classifier = bert_model\
            .from_pretrained(base_model, num_labels=2).cuda()
        self.tokenizer = BertTokenizer.from_pretrained(base_model)
        self.training_parameters = []
        self.num_batches = 0
        self.base_model = base_model

    def train(self, writer=None, batch_size=8, lr=1 * 10 ** -5, num_epochs=3, seed=None):
        if seed is not None:
            util.seed(seed)
        self.training_parameters.append({
            "batch_size": batch_size,
            "lr": lr,
            "num_epochs": num_epochs,
            "seed": seed,
            "base_model": self.base_model,
        })
        self.optimizer = BertAdam(
            self.classifier.parameters(recurse=True),
            lr=lr
        )
        self.classifier, self.optimizer = amp.initialize(self.classifier, self.optimizer, opt_level="O1")
        for epoch in range(num_epochs):
            loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                collate_fn=PaddedBatch,
                shuffle=True
            )
            for batch in tqdm(loader):
                self.optimizer.zero_grad()
                output = self.classifier(
                    batch.token_ids.cuda(),
                    batch.sequence_ids.cuda(),
                    batch.input_mask.cuda(),
                    labels=batch.labels.cuda() if self.token_based else None,
                )
                if self.token_based:
                    loss = output.sum()
                else:
                    loss = F.cross_entropy(output, batch.labels.cuda())
                self.num_batches += 1
                if writer:
                    writer.add_scalar(
                        "cross entropy loss per batch",
                        loss,
                        self.num_batches
                    )
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                self.optimizer.step()

    def test(self, writer=None):
        loader = DataLoader(self.test_dataset, batch_size=8,
                            collate_fn=PaddedBatch)
        labels = []
        predicted = []
        spoiler_probability = []
        for batch in tqdm(loader):
            with torch.no_grad():
                output = self.classifier(
                    batch.token_ids.cuda(),
                    batch.sequence_ids.cuda(),
                    batch.input_mask.cuda()
                )
                if self.token_based:
                    # Essentially we merge all examples together here
                    # That means the accuracy is to be understood globally across all tokens
                    labels.extend(batch.labels.reshape(-1))
                    predicted.extend(output.argmax(2).reshape(-1))
                    spoiler_probability.extend(
                        torch.softmax(output.reshape(-1, 2), 1)[:, 1]
                    )
                else:
                    labels.extend(batch.labels)
                    predicted.extend(list(output.argmax(1)))
                    spoiler_probability.extend(
                        list(torch.softmax(output, 1)[:, 1])
                    )
        if writer:
            writer.add_pr_curve(
                "Precision Recall",
                torch.tensor(labels),
                torch.tensor(spoiler_probability)
            )
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
    def from_file(model_path, train_path, test_path, limit=None):
        run = BertRun.for_dataset(train_path, test_path, limit=limit)
        data = torch.load(model_path)
        run.classifier.load_state_dict(data)
        return run

    @staticmethod
    def for_dataset(train_path, test_path, base_model, limit=None, token_based=False):
        tokenizer = BertTokenizer.from_pretrained(base_model)
        if token_based:
            train_dataset = TokenSpoilerDataset(
                train_path,
                tokenizer,
                limit=limit,
            )
            test_dataset = TokenSpoilerDataset(
                test_path,
                tokenizer,
                limit=limit,
            )
        else:
            train_dataset = BinarySpoilerDataset(
                train_path,
                tokenizer,
                limit=limit,
            )
            test_dataset = BinarySpoilerDataset(
                test_path,
                tokenizer,
                limit=limit,
            )
        return BertRun(train_dataset, test_dataset, base_model, token_based=token_based)
