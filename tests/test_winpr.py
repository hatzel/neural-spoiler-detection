from src.metrics import winpr
import math
import torch


def test_winpr_correct():
    reference_labels = [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    computed_labels = [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    result = winpr(reference_labels, computed_labels, window_size=3)
    assert result["true_positives"] == 4
    assert result["true_negatives"] == 40
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0
    assert result["winP"] == 1
    assert result["winR"] == 1


def test_winpr_missed_boundary():
    reference_labels = [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    computed_labels = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    result = winpr(reference_labels, computed_labels, window_size=3)
    assert result["true_positives"] == 0
    assert result["true_negatives"] == 40
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 4
    assert result["winP"] is math.nan
    assert result["winR"] == 0


def test_winpr_near_boundary():
    reference_labels = [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    computed_labels = [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]]
    result = winpr(reference_labels, computed_labels, window_size=3)
    assert result["true_positives"] == 3
    # TODO: is this correct?
    # 40 is the value from the original paper but 39 seems to make more sense to me.
    assert result["true_negatives"] == 39
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["winP"] == 0.75
    assert result["winR"] == 0.75


def test_winpr_extra_boundary():
    reference_labels = [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    computed_labels = [[1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    result = winpr(reference_labels, computed_labels, window_size=3)
    assert result["true_positives"] == 4
    assert result["true_negatives"] == 36
    assert result["false_positives"] == 4
    assert result["false_negatives"] == 0
    assert result["winP"] == 0.5
    assert result["winR"] == 1


def test_winpr_extra_boundaries():
    reference_labels = [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    computed_labels = [[0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1]]
    result = winpr(reference_labels, computed_labels, window_size=3)
    assert result["true_positives"] == 4
    assert result["true_negatives"] == 32
    assert result["false_positives"] == 8
    assert result["false_negatives"] == 0
    assert result["winP"] == 1/3
    assert result["winR"] == 1


def test_winpr_tensors():
    reference_labels = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])
    computed_labels = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])
    result = winpr(reference_labels, computed_labels, window_size=3)
    assert result["true_positives"] == 4
    assert result["true_negatives"] == 40
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0
    assert result["winP"] == 1
    assert result["winR"] == 1


def test_winpr_batch_processing_tensors():
    real = [
        torch.tensor([1, 1, 1]),
        torch.tensor([0, 0, 0]),
    ]
    computed = [
        torch.tensor([1, 1, 1]),
        torch.tensor([0, 0, 0]),
    ]
    result = winpr(real, computed)
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0
