"""
Calculates the winPR and WindowDiff metrics.

Implemented as described in the 2012 paper by Martin Scaiano and Diana Inkpen.

"""
import math
import torch


def winpr(reference_labels, computed_labels, window_size=3, average="micro"):
    if average != "micro":
        raise Exception("Only the 'micro' averaging method is supported.")
    if len(reference_labels) != len(computed_labels):
        raise Exception("Sequences need to be of equal length.")
    too_many_labels = False
    if isinstance(reference_labels[0], torch.Tensor):
        unique_labels = torch.unique(
            torch.cat([*reference_labels, *computed_labels]).reshape(-1)
        )
        if len(unique_labels) > 2:
            too_many_labels = True
    else:
        all_labels = set(*reference_labels) | set(*computed_labels)
        print(all_labels)
        if len(all_labels) > 2:
            too_many_labels = True
    if too_many_labels:
        raise Exception(
            "Segmentation data should only contain two labels"
        )

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for reference_example, computed_example in zip(reference_labels, computed_labels):
        if len(reference_example) != len(computed_example):
            raise Exception("Inner sequences need to be of equal length.")
        i = 1 - window_size
        while i < len(computed_example):
            reference_window = reference_example[max(0, i - 1):i + window_size + 1]
            computed_window = computed_example[max(0, i - 1):i + window_size + 1]
            reference_boundaries = n_boundaries(reference_window)
            computed_boundaries = n_boundaries(computed_window)
            tp += min(reference_boundaries, computed_boundaries)
            # The paper can be a bit confusing for this one
            # The window size is not equal to k but instead k = window_size + 1
            tn += len(reference_window) - 1 - max(reference_boundaries, computed_boundaries)
            fp += max(0, computed_boundaries - reference_boundaries)
            fn += max(0, reference_boundaries - computed_boundaries)
            i += 1
    results = {
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "winP": tp / (tp + fp) if tp + fp > 0 else math.nan,
        "winR": tp / (tp + fn) if tp + fn > 0 else math.nan,
    }
    return results


def n_boundaries(input):
    total = 0
    for i in range(len(input) - 1):
        if input[i] != input[i+1]:
            total += 1
    return total
