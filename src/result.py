import torch
import subprocess
import json
from datetime import datetime


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
    def __init__(self, training_parameters, train_dataset_path, test_dataset_path, model, report, average_loss):
        self.training_parameters = training_parameters
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.model = model
        self.report = report
        self.version = get_version()
        self.timestamp = datetime.now().isoformat()
        self.average_loss = average_loss

    def save(self, name=None):
        """Save result, including its model an parameters to a file."""
        data = {
            "training_parameters": self.training_parameters,
            "train_dataset_path": self.train_dataset_path,
            "test_dataset_path": self.test_dataset_path,
            "report": self.report,
            "version": self.version,
            "timestamp": self.timestamp,
        }
        name_postfix = f"-{name}" if name else ""
        f = open(f"results/{self.timestamp}-{self.version}{name_postfix}.json", "w")
        json.dump(data, f)
        torch.save(self.model.state_dict(),
                   f"results/{self.timestamp}-{self.version}{name_postfix}.model")
