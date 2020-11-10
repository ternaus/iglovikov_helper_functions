import numpy as np
from sklearn.metrics import confusion_matrix

from iglovikov_helper_functions.utils.metrics import (
    calculate_confusion_matrix_from_arrays,
    calculate_confusion_matrix_from_arrays_fast,
)


def test_confusion_matrix():
    num_classes = 19
    y_true = np.random.randint(low=0, high=25, size=(5, 7)).flatten()
    y_pred = np.random.randint(low=0, high=num_classes - 1, size=(5, 7)).flatten()

    normal_cm = calculate_confusion_matrix_from_arrays(y_true, y_pred, num_classes=num_classes)
    fast_cm = calculate_confusion_matrix_from_arrays_fast(y_true, y_pred, num_classes=num_classes)
    sklearn_cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    assert normal_cm.shape == sklearn_cm.shape
    assert fast_cm.shape == sklearn_cm.shape
    assert np.array_equal(normal_cm, sklearn_cm)
    assert np.array_equal(fast_cm, sklearn_cm)
