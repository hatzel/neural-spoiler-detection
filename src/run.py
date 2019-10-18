import torch
from tqdm import tqdm

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BertForTokenClassification,
)
from transformers.optimization import AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sklearn
from apex import amp

from result import Result
from datasets import BinarySpoilerDataset, TokenSpoilerDataset, PaddedBatch
from models import BertForBinarySequenceClassification, BertForBinaryTokenClassification
import metrics
import util


class BertRun():
    def __init__(self, train_dataset, test_dataset, base_model, token_based=False):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.token_based = token_based
        bert_model = (
            BertForBinaryTokenClassification
            if token_based
            else BertForBinarySequenceClassification
        )
        self.classifier = bert_model\
            .from_pretrained(base_model, num_labels=1).cuda()
        self.tokenizer = BertTokenizer.from_pretrained(base_model)
        self.training_parameters = []
        self.num_batches = 0
        self.base_model = base_model
        self.decision_boundary = 0.5


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
        self.optimizer = AdamW(
            self.classifier.parameters(),
            lr=lr
        )
        self.classifier, self.optimizer = amp.initialize(self.classifier, self.optimizer, opt_level="O1")
        for epoch in range(num_epochs):
            self.classifier.train()
            loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                collate_fn=PaddedBatch,
                shuffle=True
            )
            for batch in tqdm(loader):
                self.optimizer.zero_grad()
                loss, logits = self.classifier(
                    batch.token_ids.cuda(),
                    token_type_ids=batch.sequence_ids.cuda(),
                    attention_mask=batch.input_mask.cuda(),
                    labels=batch.labels.type(torch.float).cuda(),
                )
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
            del loader

    def test(self, writer=None, results_file_name=None):
        loader = DataLoader(self.test_dataset, batch_size=8,
                            collate_fn=PaddedBatch)
        labels = []
        labels_per_sample = []
        predicted = []
        predicted_per_sample = []
        spoiler_probability = []
        if results_file_name:
            results_file = open(results_file_name, "w")
        for batch in tqdm(loader):
            self.classifier.eval()
            with torch.no_grad():
                output, = self.classifier(
                    batch.token_ids.cuda(),
                    token_type_ids=batch.sequence_ids.cuda(),
                    attention_mask=batch.input_mask.cuda()
                )
                if self.token_based:
                    # Essentially we merge all examples together here
                    # That means the accuracy is to be understood globally across all tokens
                    labels.extend(batch.labels.reshape(-1))
                    predicted.extend(output.reshape(-1) > self.decision_boundary)
                    spoiler_probability.extend(output.reshape(-1))
                    predicted_per_sample.extend(batch.mask(output))
                    labels_per_sample.extend(batch.mask(batch.labels))
                    if results_file_name:
                        self._write_examples(
                            results_file, batch.token_ids, batch.labels, output
                        )
                else:
                    labels.extend(batch.labels)
                    predicted.extend(output > self.decision_boundary)
                    spoiler_probability.extend(output)
        if writer:
            writer.add_pr_curve(
                "precision_recall",
                torch.tensor(labels),
                torch.tensor(spoiler_probability)
            )
            # Flush is not present in our pytorch version
            # TODO: after upgrading pytorch replace with writer.flush()
            for w in writer.all_writers.values():
                w.flush()
        labels, predicted = (
            [t.item() for t in labels],
            [t.item() for t in predicted]
        )

        report = self.test_report(
            labels,
            predicted,
            labels_per_sample,
            predicted_per_sample,
        )

        return Result(
            training_parameters=self.training_parameters,
            train_dataset_path=self.train_dataset.file_name if self.train_dataset else None,
            test_dataset_path=self.test_dataset.file_name,
            model=self.classifier,
            report=report,
        )

    def test_report(self, labels, predicted, labels_per_sample, predicted_per_sample):
        report = {}
        precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(labels, predicted, average="binary")
        accuracy = sklearn.metrics.accuracy_score(labels, predicted)
        print(
            f"Accuracy: {accuracy}",
            f"Precision: {precision}",
            f"Recall: {recall}",
            f"F1-Score: {f1}",
            sep="\n"
        )
        report["accuracy"] = accuracy
        report["precision"] = precision
        report["recall"] = recall
        report["f1"] = f1
        tn, fp, fn, tp = sklearn.metrics\
            .confusion_matrix(labels, predicted).ravel()

        report["confusion_matrix"] = {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
        util.print_confusion_matrix(report["confusion_matrix"])

        if self.token_based:
            labels_per_sample, predicted_labels_per_sample = self._trimmed_per_label(
                labels_per_sample,
                [(t > self.decision_boundary).squeeze().cpu() for t in predicted_per_sample],
            )

            # For now we need to cast to long here: https://github.com/pytorch/pytorch/issues/27691
            report["windowdiff"] = metrics.windowdiff(
                [t.long() for t in labels_per_sample], [t.long() for t in predicted_labels_per_sample], window_size=3
            )
            report["winpr"] = metrics.winpr(
                [t.long() for t in labels_per_sample], [t.long() for t in predicted_labels_per_sample], window_size=3
            )
            print(f"Windowdiff: {report['windowdiff']}")
            print(f"WinP: {report['winpr']['winP']}, WinR {report['winpr']['winR']}")
            return report
        else:
            return report

    def _trimmed_per_label(self, labels_per_sample, predicted_per_sample):
        return (
            [x[1:-1] for x in labels_per_sample],
            [x[1:-1] for x in predicted_per_sample]
        )

    def _write_examples(self, results_file, token_ids, labels, output):
        sentences = []
        gold_labels = []
        results = []
        for sentence, gold, result in zip(token_ids, labels, output):
            sentences.append([
                self.tokenizer.ids_to_tokens[token.item()]
                for token in sentence
                if self.tokenizer.ids_to_tokens[token.item()] != "[PAD]"
            ])
            gold_labels.append(gold[:len(sentences[-1])])
            results.append(
                torch.softmax(result.reshape(-1, 2), 1)[:, 1][:len(sentences[-1])]
            )
        for gold, words, predict in zip(gold_labels, sentences, results):
            results_file.write(
                "\t".join(str(i) for i in gold.tolist()) + "\n"
            )
            results_file.write("\t".join(words) + "\n")
            predictions_colored = []
            for pred, g in zip(predict, gold):
                if round(pred.item()) == g:
                    predictions_colored.append(
                        f"{pred.item():.2f}"
                    )
                else:
                    predictions_colored.append(
                        f"\u001b[31m{pred.item():.2f}\u001b[0m"
                    )
            results_file.write(
                "\t".join(predictions_colored) + "\n"
            )
            results_file.write("\n\n")

    @staticmethod
    def from_file(model_path, train_path, test_path, base_model, **kwargs):
        run = BertRun.for_dataset(train_path, test_path, base_model, **kwargs)
        data = torch.load(model_path)
        run.classifier.load_state_dict(data)
        return run

    @staticmethod
    def for_dataset(train_path, test_path, base_model,
                    limit_test=None, train_limit=None, token_based=False):
        tokenizer = BertTokenizer.from_pretrained(base_model)
        train_dataset = None
        if token_based:
            if train_path:
                train_dataset = TokenSpoilerDataset(
                    train_path,
                    tokenizer,
                    limit=train_limit,
                )
            test_dataset = TokenSpoilerDataset(
                test_path,
                tokenizer,
                limit=limit_test,
            )
        else:
            if train_path:
                train_dataset = BinarySpoilerDataset(
                    train_path,
                    tokenizer,
                    limit=train_limit,
                )
            test_dataset = BinarySpoilerDataset(
                test_path,
                tokenizer,
                limit=limit_test,
            )
        return BertRun(train_dataset, test_dataset, base_model, token_based=token_based)
