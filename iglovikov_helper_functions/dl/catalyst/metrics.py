from typing import Dict

import numpy as np
import torch
from catalyst.dl.core import Callback, RunnerState, CallbackOrder


def calculate_confusion_matrix_from_arrays_fast(
    ground_truth: np.array, prediction: np.array, num_classes: int
) -> np.array:
    """Calculate confusion matrix for a given set of classes.

        if GT value is outside of the [0, num_classes) it is excluded.

        10x faster than scikit learn implementation.

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


def get_confusion_matrix(y_pred_logits: torch.Tensor, y_true: torch.Tensor):
    num_classes = y_pred_logits.shape[1]
    y_pred = torch.argmax(y_pred_logits, dim=1)
    ground_truth = y_true.cpu().numpy()
    prediction = y_pred.cpu().numpy()

    return calculate_confusion_matrix_from_arrays_fast(ground_truth, prediction, num_classes)


def calculate_tp_fp_fn(confusion_matrix):
    true_positives = {}
    false_positives = {}
    false_negatives = {}

    for index in range(confusion_matrix.shape[0]):
        true_positives[index] = confusion_matrix[index, index]
        false_positives[index] = confusion_matrix[:, index].sum() - true_positives[index]
        false_negatives[index] = confusion_matrix[index, :].sum() - true_positives[index]

    return {"true_positives": true_positives, "false_positives": false_positives, "false_negatives": false_negatives}


def calculate_dice(tp_fp_fn_dict):
    epsilon = 1e-7

    dice = {}

    for i in range(len(tp_fp_fn_dict["true_positives"])):
        tp = tp_fp_fn_dict["true_positives"][i]
        fp = tp_fp_fn_dict["false_positives"][i]
        fn = tp_fp_fn_dict["true_positives"][i]

        dice[i] = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)

        if not 0 <= dice[i] <= 1:
            raise ValueError()

    return dice


def calculate_jaccard(tp_fp_fn_dict):
    epsilon = 1e-7

    jaccard = {}
    for i in range(len(tp_fp_fn_dict["true_positives"])):
        tp = tp_fp_fn_dict["true_positives"][i]
        fp = tp_fp_fn_dict["false_positives"][i]
        fn = tp_fp_fn_dict["true_positives"][i]

        jaccard[i] = (tp + epsilon) / (tp + fp + fn + epsilon)

        if not 0 <= jaccard[i] <= 1:
            raise ValueError()

    return jaccard


class MulticlassDiceMetricCallback(Callback):
    def __init__(self, prefix: str = "dice", input_key: str = "targets", output_key: str = "logits", **metric_params):
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.input_key = input_key
        self.output_key = output_key
        self.metric_params = metric_params
        self.confusion_matrix = None
        self.class_names = metric_params["class_names"]  # dictionary {class_id: class_name}
        self.class_prefix = metric_params["class_prefix"]

    def _reset_stats(self):
        self.confusion_matrix = None

    def on_batch_end(self, state: RunnerState):
        outputs = state.output[self.output_key]
        targets = state.input[self.input_key]

        confusion_matrix = get_confusion_matrix(outputs, targets)

        if self.confusion_matrix is None:
            self.confusion_matrix = confusion_matrix
        else:
            self.confusion_matrix += confusion_matrix

    def on_loader_end(self, state: RunnerState):

        tp_fp_fn_dict = calculate_tp_fp_fn(self.confusion_matrix)

        batch_metrics: Dict = calculate_dice(tp_fp_fn_dict)

        for metric_id, dice_value in batch_metrics.items():
            if metric_id not in self.class_names:
                continue

            metric_name = self.class_names[metric_id]
            state.metrics.epoch_values[state.loader_name][f"{self.class_prefix}_{metric_name}"] = dice_value

        state.metrics.epoch_values[state.loader_name]["mean"] = np.mean([x for x in batch_metrics.values()])

        self._reset_stats()


class MulticlassJaccardMetricCallback(Callback):
    def __init__(
        self, prefix: str = "jaccard", input_key: str = "targets", output_key: str = "logits", **metric_params
    ):
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.input_key = input_key
        self.output_key = output_key
        self.metric_params = metric_params
        self.confusion_matrix = None
        self.class_names = metric_params["class_names"]  # dictionary {class_id: class_name}
        self.class_prefix = metric_params["class_prefix"]

    def _reset_stats(self):
        self.confusion_matrix = None

    def on_batch_end(self, state: RunnerState):
        outputs = state.output[self.output_key]
        targets = state.input[self.input_key]

        confusion_matrix = get_confusion_matrix(outputs, targets)

        if self.confusion_matrix is None:
            self.confusion_matrix = confusion_matrix
        else:
            self.confusion_matrix += confusion_matrix

        # tp_fp_fn_dict = calculate_tp_fp_fn(confusion_matrix)

        # batch_metrics: Dict = {self.class_names[key]: value for key, value in calculate_dice(tp_fp_fn_dict).items()}

        # state.metrics.add_batch_value(metrics_dict=batch_metrics)

    def on_loader_end(self, state: RunnerState):

        tp_fp_fn_dict = calculate_tp_fp_fn(self.confusion_matrix)

        batch_metrics: Dict = calculate_jaccard(tp_fp_fn_dict)

        for metric_id, jaccard_value in batch_metrics.items():
            if metric_id not in self.class_names:
                continue

            metric_name = self.class_names[metric_id]
            state.metrics.epoch_values[state.loader_name][f"{self.class_prefix}_{metric_name}"] = jaccard_value

        state.metrics.epoch_values[state.loader_name]["mean"] = np.mean([x for x in batch_metrics.values()])

        self._reset_stats()
