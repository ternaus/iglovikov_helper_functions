import numpy as np


def calculate_confusion_matrix_from_arrays(ground_truth: np.array, prediction: np.array, num_classes: int) -> np.array:
    """Calculate confusion matrix for a given set of classes.
    if GT value is outside of the [0, num_classes) it is excluded.
    Args:
        ground_truth:
        prediction:
        num_classes:
    Returns:
    """
    # a long 2xn array with each column being a pixel pair
    replace_indices = np.vstack((ground_truth.flatten(), prediction.flatten()))

    valid_index = replace_indices[0, :] < num_classes
    replace_indices = replace_indices[:, valid_index].T

    # add up confusion matrix
    confusion_matrix, _ = np.histogramdd(
        replace_indices, bins=(num_classes, num_classes), range=[(0, num_classes), (0, num_classes)]
    )
    return confusion_matrix.astype(np.uint64)


def calculate_confusion_matrix_from_arrays_fast(
    ground_truth: np.array, prediction: np.array, num_classes: int
) -> np.array:
    """Calculate confusion matrix for a given set of classes.

    if GT value is outside of the [0, num_classes) it is excluded.

    10x faster than scikit learn implementation. But consumes a lot of memory.

    Implemented by Anton Nesterenko.



    Args:
        ground_truth:
        prediction:
        num_classes:

    Returns:

    """
    if not prediction.max() < num_classes:
        raise ValueError(f"Max predicted class number must be equal {num_classes - 1}")

    r = np.zeros(num_classes ** 2)
    valid_idx = np.where(ground_truth < num_classes)[0]
    idx, vals = np.unique(prediction[valid_idx] + ground_truth[valid_idx] * num_classes, return_counts=True)
    r[idx] = vals
    return r.reshape(num_classes, num_classes).astype(np.uint64)


def calculate_tp_fp_fn(confusion_matrix: np.array) -> np.array:
    true_positives = np.diag(confusion_matrix)
    false_positives = confusion_matrix.sum(axis=0) - true_positives
    false_negatives = confusion_matrix.sum(axis=1) - true_positives
    return {"true_positives": true_positives, "false_positives": false_positives, "false_negatives": false_negatives}


def calculate_jaccard(tp_fp_fn_dict: dict) -> np.array:
    """Calculate list of Jaccard indices.

    Args:
        tp_fp_fn_dict: {"true_positives": true_positives,
                        "false_positives": false_positives,
                        "false_negatives": false_negatives}

    Returns:

    """
    epsilon = 1e-7

    true_positives = tp_fp_fn_dict["true_positives"]
    false_positives = tp_fp_fn_dict["false_positives"]
    false_negatives = tp_fp_fn_dict["false_negatives"]

    jaccard = (true_positives + epsilon) / (true_positives + false_positives + false_negatives + epsilon)

    if not np.all(jaccard <= 1):
        raise ValueError("Jaccard index should be less than 1")

    if not np.all(jaccard >= 0):
        raise ValueError("Jaccard index should be more than 1")

    return jaccard


def calculate_dice(tp_fp_fn_dict: dict) -> np.array:
    """Calculate list of Dice coefficients.

    Args:
        tp_fp_fn_dict: {"true_positives": true_positives,
                        "false_positives": false_positives,
                        "false_negatives": false_negatives}

    Returns:

    """
    epsilon = 1e-7

    true_positives = tp_fp_fn_dict["true_positives"]
    false_positives = tp_fp_fn_dict["false_positives"]
    false_negatives = tp_fp_fn_dict["false_negatives"]

    dice = (2 * true_positives + epsilon) / (2 * true_positives + false_positives + false_negatives + epsilon)

    if not np.all(dice <= 1):
        raise ValueError("Jaccard index should be less than 1")

    if not np.all(dice >= 0):
        raise ValueError("Jaccard index should be more than 1")

    return dice
