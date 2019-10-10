from src.metrics import windowdiff

# The values here differ from those in "Getting more from segementation evaluation"
# they do however align with the nltk implementation.
# In fact it appears that the entire table 1 in "Getting more from segementation evaluation"
# assumes that there are 10 windows when in fact there should only be 9.
# If you want to test this, keep in mind that nltk.metrics.windowdiff doesn't
# take a sequence of classes but of boundaries.
# The following function can be used to convert them:


def to_boundaries(input):
    """
    Convert class list to nltk's boundary representation.
    """
    result = []
    for i in range(len(input) - 1):
        if input[i] == input[i + 1]:
            result.append(0)
        else:
            result.append(1)
    return result


def test_windowdiff_correct():
    reference_labels = [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    computed_labels = [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    result = windowdiff(reference_labels, computed_labels, window_size=3)
    assert result == 0


def test_windowdiff_missed_boundary():
    reference_labels = [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    computed_labels = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    result = windowdiff(reference_labels, computed_labels, window_size=3)
    assert result == 3/9


def test_windowdiff_near_boundary():
    reference_labels = [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    computed_labels = [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]]
    result = windowdiff(reference_labels, computed_labels, window_size=3)
    assert result == 2/9


def test_windowdiff_extra_boundary():
    reference_labels = [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    computed_labels = [[1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    result = windowdiff(reference_labels, computed_labels, window_size=3)
    assert result == 1/9


def test_windowdiff_extra_boundaries():
    reference_labels = [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    computed_labels = [[0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1]]
    result = windowdiff(reference_labels, computed_labels, window_size=3)
    assert result == 5/9
