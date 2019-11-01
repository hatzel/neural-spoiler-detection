import math
import os
import torch
from tqdm import tqdm

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BertForTokenClassification,
    WarmupLinearSchedule,
)
from transformers.optimization import AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sklearn

from stlr import STLR
from result import Result
from datasets import BinarySpoilerDataset, TokenSpoilerDataset, PaddedBatch, FileType
from models import BertForBinarySequenceClassification, BertForBinaryTokenClassification
import early_stopping
import metrics
import util

EPOCH_MODEL_PATH = "epoch_models"


class BertRun():
    def __init__(self, train_dataset, test_dataset, base_model,
                 token_based=False, test_loss_report=True,
                 test_loss_early_stopping=False, scheduler_epochs=None):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.token_based = token_based
        self.test_loss_report = test_loss_report
        self.test_loss_early_stopping = test_loss_early_stopping
        bert_model = (
            BertForBinaryTokenClassification
            if token_based
            else BertForBinarySequenceClassification
        )
        if token_based:
            # Based on tokens in an early dataset
            # Only applies to the token dataset as it is not balanced
            spoiler_class_weight = (213569 / 22383)
            self.classifier = bert_model.from_pretrained(
                base_model,
                positive_class_weight=torch.tensor(spoiler_class_weight),
                num_labels=1
            ).cuda()
        else:
            # The tv-tropes dataset is not quite balanced
            if train_dataset.format == FileType.CSV:
                spoiler_class_weight = (7800 / 6988)
            else:
                spoiler_class_weight = None
            self.classifier = bert_model.from_pretrained(
                base_model, num_labels=1, positive_class_weight=spoiler_class_weight).cuda()
        self.tokenizer = BertTokenizer.from_pretrained(base_model)
        self.training_parameters = []
        self.num_batches = 0
        self.base_model = base_model
        self.decision_boundary = 0.5
        self.epoch_models = {}
        self.early_stopped_at = None
        self.scheduler_epochs = scheduler_epochs

    def train(self, writer=None, batch_size=8, lr=1 * 10 ** -5, num_epochs=3,
              seed=None, half_precision=False):
        max_grad_norm = 1.0
        test_losses = []
        should_stop = early_stopping.ConsecutiveNonImprovment(3)
        self.training_parameters.append({
            "batch_size": batch_size,
            "lr": lr,
            "num_epochs": num_epochs,
            "seed": seed,
            "base_model": self.base_model,
        })
        optimizer = AdamW(
            self.classifier.parameters(),
            lr=lr
        )
        scheduler = self.warumup_cooldown_scheduler(optimizer, num_epochs, batch_size)
        if half_precision:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("For half precision you need to have apex installed!")
            self.classifier, optimizer = amp.initialize(
                self.classifier,
                optimizer,
                opt_level="O1"
            )
        for epoch in range(num_epochs):
            # To not depend on if we run tests after each epoch we need to seed here
            if seed is not None:
                util.seed(seed + epoch)
            print(f"Starting training epoch {epoch + 1}/{num_epochs}")
            self.classifier.train()
            loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                collate_fn=PaddedBatch,
                shuffle=True
            )
            for batch in tqdm(loader):
                optimizer.zero_grad()
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
                if half_precision:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
            util.seed_for_testing()
            if self.test_loss_report:
                result = self.test()
                writer.add_scalar(
                    "average test loss",
                    result.average_loss,
                    self.num_batches,
                )
                writer.add_scalar(
                    "test f1",
                    result.report["f1"],
                    self.num_batches,
                )
                writer.add_scalar(
                    "test accuracy",
                    result.report["accuracy"],
                    self.num_batches,
                )
                self.save_epoch_model(result, epoch)
                test_losses.append(result.average_loss)
                # Early stopping when test loss is no longer improving
                if should_stop(test_losses):
                    print("Test loss no longer improving, stopping!")
                    print(f"(losses were {test_losses})")
                    best_epoch = sorted(
                        enumerate(test_losses),
                        key=lambda kv: kv[1]
                    )[0][0]
                    self.early_stopped_at = best_epoch
                    self.load_epoch_model(best_epoch)
                    return
            del loader

    def test(self, writer=None, results_file_name=None):
        loader = DataLoader(self.test_dataset, batch_size=8,
                            collate_fn=PaddedBatch)
        util.seed_for_testing()
        labels = []
        labels_per_sample = []
        predicted = []
        predicted_per_sample = []
        spoiler_probability = []
        total_loss = 0
        num_losses = 0
        if results_file_name:
            results_file = open(results_file_name, "w")
        for batch in tqdm(loader):
            self.classifier.eval()
            with torch.no_grad():
                loss, output = self.classifier(
                    batch.token_ids.cuda(),
                    token_type_ids=batch.sequence_ids.cuda(),
                    attention_mask=batch.input_mask.cuda(),
                    labels=batch.labels.type(torch.float).cuda(),
                )
                total_loss += loss
                num_losses += 1
                if self.token_based:
                    # Essentially we merge all examples together here
                    # That means the accuracy is to be understood globally across all tokens
                    labels.extend(batch.labels.reshape(-1))
                    predicted.extend(output.reshape(-1) > self.decision_boundary)
                    spoiler_probability.extend(output.reshape(-1))
                    predicted_per_sample.extend(
                        t.squeeze() for t in batch.to_full_prediction(output, torch.tensor(0.0))
                    )
                    labels_per_sample.extend(batch.full_labels)
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
                torch.tensor(spoiler_probability),
            )
            writer.flush()
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
            train_dataset_path=" ".join(self.train_dataset.file_names) if self.train_dataset else None,
            test_dataset_path=" ".join(self.test_dataset.file_names),
            model=self.classifier,
            report=report,
            average_loss=total_loss / num_losses,
            early_stopped_at=self.early_stopped_at,
            scheduler_epochs=self.scheduler_epochs,
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
            predicted_labels_per_sample = [(t > self.decision_boundary).cpu() for t in predicted_per_sample]

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

    def warumup_cooldown_scheduler(self, optimizer, num_epochs, batch_size):
        epochs = self.scheduler_epochs or num_epochs
        total_num_batches = math.ceil(len(self.train_dataset) / batch_size) * epochs
        peak_lr_after = int(total_num_batches / 2)
        total_steps = total_num_batches
        return STLR(
            optimizer=optimizer,
            warmup_fraction=0.1,
            min_lr_ratio=1/32,
            t_total=total_steps,
        )

    def save_epoch_model(self, result, epoch):
        model_id = result.save(f"epoch_{epoch}", path=EPOCH_MODEL_PATH)
        self.epoch_models[epoch] = model_id

    def load_epoch_model(self, epoch):
        model_id = self.epoch_models[epoch]
        data = torch.load(f"{EPOCH_MODEL_PATH}/{model_id}.model")
        self.classifier.load_state_dict(data)

    def clear_epoch_models(self):
        for model_id in self.epoch_models.values():
            os.remove(f"{EPOCH_MODEL_PATH}/{model_id}.model")
            os.remove(f"{EPOCH_MODEL_PATH}/{model_id}.json")

    def __del__(self):
        # Epoch models should only be cleaned up
        self.clear_epoch_models()

    @staticmethod
    def from_file(model_path, train_paths, test_path, base_model, **kwargs):
        run = BertRun.for_dataset(train_paths, test_path, base_model, **kwargs)
        data = torch.load(model_path)
        run.classifier.load_state_dict(data)
        return run

    @staticmethod
    def for_dataset(train_paths, test_path, base_model,
                    limit_test=None, train_limit=None, token_based=False, **kwargs):
        tokenizer = BertTokenizer.from_pretrained(base_model)
        train_dataset = None
        if token_based:
            if train_paths:
                train_dataset = TokenSpoilerDataset(
                    train_paths,
                    tokenizer,
                    limit=train_limit,
                )
            test_dataset = TokenSpoilerDataset(
                test_path,
                tokenizer,
                limit=limit_test,
            )
        else:
            if train_paths:
                train_dataset = BinarySpoilerDataset(
                    train_paths,
                    tokenizer,
                    limit=train_limit,
                )
            test_dataset = BinarySpoilerDataset(
                test_path,
                tokenizer,
                limit=limit_test,
            )
        return BertRun(train_dataset, test_dataset, base_model, token_based=token_based, **kwargs)
