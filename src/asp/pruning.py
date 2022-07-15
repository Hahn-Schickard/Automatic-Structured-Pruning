'''Copyright [2020] Hahn-Schickard-Gesellschaft fuer angewandte Forschung e.V.,
                    Daniel Konegen + Marcus Rueb
   Copyright [2021] Karlsruhe Institute of Technology, Daniel Konegen
   Copyright [2022] Hahn-Schickard-Gesellschaft fuer angewandte Forschung e.V.,
                    Daniel Konegen + Marcus Rueb
   SPDX-License-Identifier: Apache-2.0
============================================================================'''

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split


def factor_pruning(keras_model, prun_factor_dense=10, prun_factor_conv=10,
                   metric='L1', comp=None, label_one_hot=False):
    """
    A given keras model get pruned. The factor for dense and conv says how
    many percent of the dense and conv layers should be deleted.

    Args:
        keras_model:        Model which should be pruned
        prun_factor_dense:  Integer which says how many percent of the neurons
                            should be deleted
        prun_factor_conv:   Integer which says how many percent of the filters
                            should be deleted
        metric:             Metric which should be used for model pruning
        comp:               Dictionary with compiler settings
        label_one_hot:      Boolean value if labels are one hot encoded or not

    Return:
        pruned_model:      New model after pruning
    """

    if callable(getattr(keras_model, "predict", None)):
        model = keras_model
    elif isinstance(keras_model, str) and ".h5" in keras_model:
        model = load_model(keras_model)
    else:
        print("No model given to prune")
    
    num_classes = model.layers[-1].output_shape[1]

    if num_classes <= 2 and isinstance(comp, type(None)):
        comp = {
            "optimizer": 'adam',
            "loss": tf.keras.losses.BinaryCrossentropy(),
            "metrics": 'accuracy'}
    elif num_classes > 3 and isinstance(comp, type(None)):
        if label_one_hot:
            comp = {
                "optimizer": 'adam',
                "loss": tf.keras.losses.CategoricalCrossentropy(),
                "metrics": 'accuracy'}
        else:
            comp = {
                "optimizer": 'adam',
                "loss": tf.keras.losses.SparseCategoricalCrossentropy(),
                "metrics": 'accuracy'}

    layer_types, layer_params, layer_output_shape, layer_bias, netstr = (
        load_model_param(model))
    num_new_neurons = np.zeros(shape=len(layer_params), dtype=np.int16)
    num_new_filters = np.zeros(shape=len(layer_params), dtype=np.int16)

    layer_params, num_new_neurons, num_new_filters, layer_output_shape = (
        model_pruning(layer_types, layer_params, layer_output_shape,
                      layer_bias, netstr, num_new_neurons, num_new_filters,
                      prun_factor_dense, prun_factor_conv, metric))

    print("Finish with pruning")

    pruned_model = build_pruned_model(model, layer_params, layer_types,
                                      num_new_neurons, num_new_filters, comp)

    print("Model built")

    return pruned_model


def accuracy_pruning(keras_model, comp, x_train, y_train=None, x_val=None,
                    y_val=None, pruning_acc=None, max_acc_loss=5,
                    label_one_hot=False):
    """
    A given keras model gets pruned. Either an accuracy value (in %) can be
    specified, which the minimized model has to still achieve. Or the maximum
    loss of accuracy (in %) that the minimized model may experience. The model
    is reduced step by step until the accuracy value is under run or the
    accuracy loss is exceeded.

    Args:
        keras_model:        Model which should be pruned
        comp:               Compiler settings
        x_train:            Training data to retrain the model after pruning
        y_train:            Labels of training data if data loader used leave
                            it "None"
        x_val:              Validation data
        y_val:              Labels of validation data if data loader used leave
                            it "None"
        pruning_acc:        Integer which says which accuracy value (in %)
                            should not be fall below.
        max_acc_loss:       Integer which says which accuracy loss (in %)
                            should not be exceed, default is -5%
        label_one_hot:      Boolean value if labels are one hot encoded or not

    Return:
        pruned_model:   New model after pruning
    """
    pruning_factor = 5
    last_pruning_step = None
    all_pruning_factors = [5]
    lowest_pruning_factor_not_working = 100
    original_model_acc = None
    req_acc = None

    if callable(getattr(keras_model, "predict", None)):
        original_model = keras_model
    elif isinstance(keras_model, str) and ".h5" in keras_model:
        original_model = load_model(keras_model)
    else:
        print("No model given to prune")

    original_model.compile(**comp)

    if not isinstance(pruning_acc, type(None)):
        req_acc = pruning_acc / 100
    else:
        if isinstance(x_train, np.ndarray) or isinstance(x_train, list):
            if not isinstance(x_val, type(None)) and not isinstance(y_val, type(None)):
                original_model_acc = original_model.evaluate(
                    x_val, y_val)[-1]
            else:
                x_train, x_val, y_train, y_val = train_test_split(
                    x_train, y_train, test_size=0.2)
                original_model_acc = original_model.evaluate(
                    x_val, y_val)[-1]
        else:
            original_model_acc = original_model.evaluate(
                x_val)[-1]
        print(f"Start model accuracy: {original_model_acc * 100:.2f} %")
        req_acc = original_model_acc - (max_acc_loss / 100)
        print(f"Minimum required model accuracy: {req_acc * 100:.2f} %")

    train_epochs = 10
    threshold = ThresholdCallback(req_acc)
    callbacks = [threshold]

    while pruning_factor <= 95:

        print("Next pruning factors: " + str(pruning_factor))

        model = factor_pruning(original_model, prun_factor_dense=pruning_factor,
                            prun_factor_conv=pruning_factor, metric='L1',
                            comp=comp, label_one_hot=label_one_hot)

        if isinstance(x_train, np.ndarray) or isinstance(x_train, list):
            history = model.fit(x=x_train, y=y_train, batch_size=64,
                                validation_data=(x_val, y_val),
                                epochs=train_epochs, callbacks=callbacks)
        else:
            history = model.fit_generator(
                x_train, steps_per_epoch=len(x_train),
                validation_data=x_val,
                validation_steps=len(x_val),
                epochs=train_epochs, callbacks=callbacks)

        if history.history['val_accuracy'][-1] < req_acc:
            # Required accuracy is not reached
            if lowest_pruning_factor_not_working > pruning_factor:
                lowest_pruning_factor_not_working = pruning_factor

            if pruning_factor == 5:
                print("No pruning possible")
                return original_model

            if last_pruning_step == 2:
                print("Pruningfactor dense and conv: " +
                      str(pruning_factor - last_pruning_step))
                return pruned_model
            elif last_pruning_step == 5:
                pruning_factor -= 3
                last_pruning_step = 2
            elif last_pruning_step == 10:
                pruning_factor -= 5
                last_pruning_step = 5
            elif last_pruning_step == 15:
                pruning_factor -= 5
                last_pruning_step = 10

        else:
            # Required accuracy is reached
            pruned_model = model
            # Set pruning factor for next pruning step
            if (len(history.history['val_accuracy']) <=
                    int(0.3 * train_epochs)):
                pruning_factor += 15
                last_pruning_step = 15
            elif (len(history.history['val_accuracy']) <=
                    int(0.5 * train_epochs)):
                pruning_factor += 10
                last_pruning_step = 10
            elif (len(history.history['val_accuracy']) <=
                    int(0.7 * train_epochs)):
                pruning_factor += 5
                last_pruning_step = 5
            elif (len(history.history['val_accuracy']) >
                    int(0.7 * train_epochs)):
                pruning_factor += 2
                last_pruning_step = 2

        if lowest_pruning_factor_not_working < pruning_factor:
            # Check if pruning factor is higher than the lowest one which
            # didn't work and adjust the pruning factor if it's true
            if (lowest_pruning_factor_not_working -
                    (pruning_factor - last_pruning_step) <= 2):
                print("Pruningfactor dense and conv: " +
                      str(pruning_factor - last_pruning_step))
                return pruned_model
            elif (lowest_pruning_factor_not_working -
                    (pruning_factor - last_pruning_step) <= 5):
                pruning_factor = (pruning_factor - last_pruning_step) + 2
                last_pruning_step = 2

        if all_pruning_factors.count(pruning_factor) >= 1:
            # Check if the pruning factor for next iteration was already
            # applied
            if history.history['val_accuracy'][-1] < req_acc:
                # If required accuracy wasn't reached, the pruning factor is
                # lowered in the step before. If the new pruning factor was
                # already applied, this is one which worked, so you increase
                # it a little step.
                if last_pruning_step == 2 or last_pruning_step == 5:
                    pruning_factor += 2
                    last_pruning_step = 2
                elif last_pruning_step == 10:
                    pruning_factor += 5
                    last_pruning_step = 5
                elif last_pruning_step == 15:
                    pruning_factor += 10
                    last_pruning_step = 10
            else:
                # If required accuracy was reached, the pruning factor is
                # increased in the step before. If the new pruning factor was
                # already applied, this is one which didn't work, so you lower
                # it a little step.
                if last_pruning_step == 2 or last_pruning_step == 5:
                    pruning_factor -= 3
                    last_pruning_step = 2
                elif last_pruning_step == 10:
                    pruning_factor -= 5
                    last_pruning_step = 5
                elif last_pruning_step == 15:
                    pruning_factor -= 10
                    last_pruning_step = 10

        all_pruning_factors.append(pruning_factor)

    return pruned_model


def stepwise_factor_pruning(keras_model, x_train, y_train=None, x_test=None,
                            y_test=None, prun_factor_dense=10, prun_factor_conv=10,
                            num_steps=5, metric='L1', comp=None, label_one_hot=False):
    """
    A given keras model get stepwise pruned. You can define how many times the
    model should get pruned and finetuned afterwards. The factor for dense and conv
    says how many percent of the dense and conv layers should be deleted for each step.

    Args:
        keras_model:        Model which should be pruned
        x_train:            Training data to retrain the model after pruning
        y_train:            Labels of training data if data loader used leave
                            it "None"
        x_test:             Test data to retrain the model after pruning
        y_test:             Labels of test data if data loader used leave
                            it "None"
        prun_factor_dense:  Integer which says how many percent of the neurons
                            should be deleted
        prun_factor_conv:   Integer which says how many percent of the filters
                            should be deleted
        num_steps:          Integer which says how many pruning steps should be
                            performed.
        metric:             Metric which should be used for model pruning
        comp:               Dictionary with compiler settings
        label_one_hot:      Boolean value if labels are one hot encoded or not

    Return:
        pruned_model:      New model after pruning
    """
    
    if callable(getattr(keras_model, "predict", None)):
        model = keras_model
    elif isinstance(keras_model, str) and ".h5" in keras_model:
        model = load_model(keras_model)
    else:
        print("No model given to prune")

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    for i in range(num_steps):
        print(f"pruning step: {i+1}/{num_steps}")
        if i != 0:
            keras_model = pruned_model

        pruned_model = factor_pruning(keras_model, prun_factor_dense, prun_factor_conv,
                                      metric, comp, label_one_hot)

        pruned_model.fit(x=x_train, y=y_train, batch_size=64,
                         validation_data=(x_val, y_val),
                         epochs=10)

        if not isinstance(x_test, type(None)) and not isinstance(y_test, type(None)):
            pruned_model.evaluate(x_test, y_test)
        
    
    return pruned_model


def stepwise_accuracy_pruning(keras_model, x_train, y_train=None, x_val=None,
                            y_val=None, pruning_acc=None, max_acc_loss=5,
                            prun_factor_dense=10, prun_factor_conv=10, 
                            metric='L1', comp=None, label_one_hot=False):
    """
    A given keras model get stepwise pruned. Either an accuracy value (in %)
    can be specified, which the minimized model has to still achieve. Or the maximum
    loss of accuracy (in %) that the minimized model may experience. The model
    is reduced step by step until the accuracy value is under run or the
    accuracy loss is exceeded.
    
    Args:
        keras_model:        Model which should be pruned
        x_train:            Training data to retrain the model after pruning
        y_train:            Labels of training data if data loader used leave
                            it "None"
        x_val:              Validation data
        y_val:              Labels of validation data if data loader used leave
                            it "None"
        pruning_acc:        Integer which says which accuracy value (in %)
                            should not be fall below.
        max_acc_loss:       Integer which says which accuracy loss (in %)
                            should not be exceed, default is -5%
        prun_factor_dense:  Integer which says how many percent of the neurons
                            should be deleted
        prun_factor_conv:   Integer which says how many percent of the filters
                            should be deleted
        metric:             Metric which should be used for model pruning
        comp:               Dictionary with compiler settings
        label_one_hot:      Boolean value if labels are one hot encoded or not

    Return:
        pruned_model:      New model after pruning
    """
    
    original_model_acc = None
    req_acc = None

    if callable(getattr(keras_model, "predict", None)):
        original_model = keras_model
    elif isinstance(keras_model, str) and ".h5" in keras_model:
        original_model = load_model(keras_model)
    else:
        print("No model given to prune")

    original_model.compile(**comp)

    if not isinstance(pruning_acc, type(None)):
        req_acc = pruning_acc / 100
    else:
        if isinstance(x_train, np.ndarray) or isinstance(x_train, list):
            if not isinstance(x_val, type(None)) and not isinstance(y_val, type(None)):
                original_model_acc = original_model.evaluate(
                    x_val, y_val)[-1]
            else:
                x_train, x_val, y_train, y_val = train_test_split(
                    x_train, y_train, test_size=0.2)
                original_model_acc = original_model.evaluate(
                    x_val, y_val)[-1]
        else:
            original_model_acc = original_model.evaluate(
                x_val)[-1]
        print(f"Start model accuracy: {original_model_acc * 100:.2f} %")
        req_acc = original_model_acc - (max_acc_loss / 100)
        print(f"Minimum required model accuracy: {req_acc * 100:.2f} %")

        print(f"prun_factor_dense: {prun_factor_dense} %")
        print(f"prun_factor_conv: {prun_factor_conv} %")

    train_epochs = 10
    threshold = ThresholdCallback(req_acc)
    callbacks = [threshold]
    pruning_step = 1

    model = original_model

    while True:
        print(f"pruning step: {pruning_step}")

        pruned_model = factor_pruning(model, prun_factor_dense=prun_factor_dense,
                            prun_factor_conv=prun_factor_conv, metric=metric,
                            comp=comp, label_one_hot=label_one_hot)

        if isinstance(x_train, np.ndarray) or isinstance(x_train, list):
            history = pruned_model.fit(x=x_train, y=y_train, batch_size=64,
                                validation_data=(x_val, y_val),
                                epochs=train_epochs, callbacks=callbacks)
        else:
            history = pruned_model.fit_generator(
                x_train, steps_per_epoch=len(x_train),
                validation_data=x_val,
                validation_steps=len(x_val),
                epochs=train_epochs, callbacks=callbacks)

        if history.history['val_accuracy'][-1] < req_acc:
            if pruning_step == 0:
                print("No pruning possible with these factors")
                return original_model
            else:
                return model
        else:
            # Required accuracy is reached
            # Check if number of parameters of pruned model are less than 
            # 100%-mean(pruning factors)*1.5 of the model of step before.
            # If not pruning stuck --> Increase pruning factors:
            pruning_factor_mean = (prun_factor_dense + prun_factor_conv) / 2
            weighted_model_params = (1.0 - (pruning_factor_mean*1.5/100)) * model.count_params()
            if (weighted_model_params <= pruned_model.count_params()):
                prun_factor_dense += 5
                prun_factor_conv += 5
                print(f"pruning_factor_mean: {pruning_factor_mean} %")
                print(f"weighted_model_params: {weighted_model_params}")
                print(f"pruned_model.count_params(): {pruned_model.count_params()}")
                print("Increased pruning factors")
                print(f"prun_factor_dense: {prun_factor_dense} %")
                print(f"prun_factor_conv: {prun_factor_conv} %")
                continue

            model = pruned_model
        
        pruning_step += 1
