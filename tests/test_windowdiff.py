from src.metrics import windowdiff
import sys
try:
    from nltk.metrics import windowdiff as nltk_windowdiff
except ImportError:
    pass
# The values here differ from those in "Getting more from segementation evaluation"
# they do however align with the nltk implementation.

# We do however only get the same values as the nltk function called with k - 1.
# NLTK operates on boundaries, the original paper however is pretty clear that
# the number of boundaries between label i and label i + k is considered.
# "where b(i, j) represents the number of boundaries between positions i and j in the text"
# (A Critique and Improvement of an Evaluation Metric for Text Segmentation,
# Lev Pevzner, Marti A. Hearst)


def to_boundaries(input):
    """
    Convert class list to nltk's boundary representation.
    """
    result = ""
    for i in range(len(input) - 1):
        if input[i] == input[i + 1]:
            result += "0"
        else:
            result += "1"
    return result


def test_windowdiff_correct():
    reference_labels = [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    computed_labels = [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    result = windowdiff(reference_labels, computed_labels, window_size=3)
    assert result == 0
    if 'nltk' in sys.modules:
        result = nltk_windowdiff(to_boundaries(reference_labels[0]), to_boundaries(computed_labels[0]), 2)
        assert result == 0


def test_windowdiff_missed_boundary():
    reference_labels = [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    computed_labels = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    result = windowdiff(reference_labels, computed_labels, window_size=3)
    assert result == 0.2
    if 'nltk' in sys.modules:
        result = nltk_windowdiff(to_boundaries(reference_labels[0]), to_boundaries(computed_labels[0]), 2)
        assert result == 0.2


def test_windowdiff_near_boundary():
    reference_labels = [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    computed_labels = [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]]
    result = windowdiff(reference_labels, computed_labels, window_size=3)
    assert result == 0.2
    if 'nltk' in sys.modules:
        result = nltk_windowdiff(to_boundaries(reference_labels[0]), to_boundaries(computed_labels[0]), 2)
        assert result == 0.2


def test_windowdiff_extra_boundary():
    reference_labels = [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    computed_labels = [[1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    result = windowdiff(reference_labels, computed_labels, window_size=3)
    assert result == 0.1
    if 'nltk' in sys.modules:
        result = nltk_windowdiff(to_boundaries(reference_labels[0]), to_boundaries(computed_labels[0]), 2)
        assert result == 0.1


def test_windowdiff_extra_boundaries():
    # reference_labels = [[0, 0, 0, 0, 0,   0, | 1,   1, 1, 1, 1, 1]]
    # computed_labels =  [[0, 0, 0, 0, 0, | 1, | 0, | 1, 1, 1, 1, 1]]
    reference_labels = [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    computed_labels = [[0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1]]
    result = windowdiff(reference_labels, computed_labels, window_size=3)
    assert result == 0.4
    if 'nltk' in sys.modules:
        result = nltk_windowdiff(to_boundaries(reference_labels[0]), to_boundaries(computed_labels[0]), 2)
        assert result == 0.4
