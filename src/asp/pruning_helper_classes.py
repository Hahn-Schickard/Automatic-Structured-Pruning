'''Copyright [2020] Hahn-Schickard-Gesellschaft fuer angewandte Forschung e.V.,
                    Daniel Konegen + Marcus Rueb
   Copyright [2021] Karlsruhe Institute of Technology, Daniel Konegen
   Copyright [2022] Hahn-Schickard-Gesellschaft fuer angewandte Forschung e.V.,
                    Daniel Konegen + Marcus Rueb
   SPDX-License-Identifier: Apache-2.0
============================================================================'''

import tensorflow as tf
from typing import NamedTuple


class NetStructure(NamedTuple):
    parents: list
    children: list


class ThresholdCallback(tf.keras.callbacks.Callback):
    """Custom callback for model training.

    This is a custom callback function. You can define an accuracy threshold
    value when the model training should be stopped.

    Attributes:
        threshold:  Accuracy value to stop training.
    """

    def __init__(self, threshold):
        """
        Initialization of the ThresholdCallback class.

        Args:
            threshold:  Accuracy value to stop training.
        """
        super(ThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        """
        If the validation accuracy is higher than the threshold at the end of
        an epoch, the training is stopped.

        Args:
            epoch:  Current epoch
            logs:   Logs of model training
        """
        val_acc = logs["val_accuracy"]
        if val_acc >= self.threshold:
            self.model.stop_training = True
