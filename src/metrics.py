"""
Calculates the winPR and WindowDiff metrics.

Implemented as described in the 2012 paper by Martin Scaiano and Diana Inkpen.

"""
import math


def winpr(reference_labels, computed_labels, window_size=3):
    if len(reference_labels) != len(computed_labels):
        raise Exception("Sequences need to be of equal length.")
    items = {}
    for item in reference_labels:
        items[item] = None
    for item in computed_labels:
        items[item] = None
    if len(items) > 2:
        raise Exception("Segmentation data should only contain two labels")

    i = 1 - window_size
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    while i < len(reference_labels):
        reference_window = reference_labels[max(0, i - 1):i + window_size + 1]
        computed_window = computed_labels[max(0, i - 1):i + window_size + 1]
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
        "true_negatives":  tn,
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
