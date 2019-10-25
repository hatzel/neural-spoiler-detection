import torch
import subprocess
import json
from datetime import datetime
import copy


def get_version():
    """Use git describe to specify the code we are running."""
    ret, version = subprocess.getstatusoutput(
        "git describe --always --dirty=+"
    )
    if ret != 0:
        print("Warning: git describe failed!")
        return "unknown"
    else:
        return version


class Result():
    def __init__(self, training_parameters, train_dataset_path,
                 test_dataset_path, model, report, average_loss,
                 early_stopped_at, scheduler_epochs):
        self.training_parameters = training_parameters
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.model = model
        self.report = report
        self.version = get_version()
        self.timestamp = datetime.now().isoformat()
        self.average_loss = average_loss
        self.early_stopped_at = early_stopped_at
        self.scheduler_epochs = scheduler_epochs

    def save(self, name=None, path="results", writer=None):
        """Save result, including its model an parameters to a file."""
        data = {
            "training_parameters": self.training_parameters,
            "train_dataset_path": self.train_dataset_path,
            "test_dataset_path": self.test_dataset_path,
            "report": self.report,
            "version": self.version,
            "timestamp": self.timestamp,
            "early_stopped_at": self.early_stopped_at,
            "scheduler_epochs": self.scheduler_epochs,
        }
        name_postfix = f"-{name}" if name else ""
        base_name = f"{self.timestamp}-{self.version}{name_postfix}"
        f = open(f"{path}/{base_name}.json", "w")
        json.dump(data, f)
        torch.save(self.model.state_dict(),
                   f"{path}/{base_name}.model")
        if writer:
            self.log_to_tensorboard(writer)
        return base_name

    def log_to_tensorboard(self, writer):
        flat_report = copy.deepcopy(self.report)
        flat_report.update(flat_report["confusion_matrix"])
        del flat_report["confusion_matrix"]
        writer.add_hparams(self.training_parameters[0], flat_report)
