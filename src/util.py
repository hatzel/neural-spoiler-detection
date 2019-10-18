import random
import torch
import numpy as np


def seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def print_confusion_matrix(confusion):
    print(f"{'':<18}{'Predicted Spoiler':>20}{'Predicted Non-Spoiler':>28}")
    print(f"{'Actual Spoiler':<18}{confusion['tp']:>20}{confusion['fn']:>28}")
    print(f"{'Actual Non-Spoiler':<18}{confusion['fp']:>20}{confusion['tn']:>28}")
