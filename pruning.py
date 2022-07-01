'''Copyright [2020] Hahn-Schickard-Gesellschaft fuer angewandte Forschung e.V.,
                    Daniel Konegen + Marcus Rueb
   Copyright [2021] Karlsruhe Institute of Technology, Daniel Konegen
   Copyright [2022] Hahn-Schickard-Gesellschaft fuer angewandte Forschung e.V.,
                    Daniel Konegen + Marcus Rueb
   SPDX-License-Identifier: Apache-2.0
============================================================================'''

import os, copy
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

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


def get_layer_shape_dense(new_model_param, layer):
    """
    Gets the structure of the new generated model and return the shape of the
    current layer.

    Args:
        new_model_param:    The params of the new generated model
        layer:              The current layer we want the shape from

    Return:
        shape of the current layer
    """
    return new_model_param[layer][0].shape[1]


def get_layer_shape_conv(new_model_param, layer):
    """
    Gets the structure of the new generated model and return the shape of the
    current layer.

    Args:
        new_model_param:    The params of the new generated model
        layer:              The current layer we want the shape from

    Return:
        shape of the current layer
    """
    return new_model_param[layer][0].shape[3]


def get_last_layer_with_params(layer_params):
    """
    Get the index of the last layer containing parameter

    Args:
        layer_params:   Stores the current weights of the model

    Return:
        index of last layer containing parameter
    """
    last_params_index = 0
    for i in range(len(layer_params)):
        if len(layer_params[i]) > 0:
            last_params_index = i
    return last_params_index


def load_model_param(model):
    """
    Gets layer names, layer weights and output_shape of each layer from the
    given keras model. The weights of all layers are stored in layer_params.
    This array will be used to delete the neurons and reload the weights
    later. The type of all layers are stored in layer_types to search for dense
    and conv layers. The output shape of each layer is also needed to set the
    right number of parameters in layers like max_pool.

    Args:
        model:  Model which should be pruned

    Return:
        layer_types:        Type of all layers of the model
        layer_params:       All weight matrices of the model
        layer_output_shape: Output shape of all layers of the model
        layer_bias:         Bool list if layers contain bias or not
        network_structure:  NetStructure containing information about parent and child nodes
    """
    layer_params = []
    layer_types = []
    layer_names = []
    layer_parent_idx = []
    layer_output_shape = []
    layer_bias = []

    model_config = model.get_config()
    layers_dict = dict()
    is_seq = model.__class__.__name__ == "Sequential"

    for idx, layer in enumerate(model.layers):
        layer_types.append(layer.__class__.__name__)
        layers_dict[layer.name] = idx
        layer_params.append(layer.get_weights())
        layer_output_shape.append(list(layer.output_shape))
        try:
            layer_bias.append(layer.use_bias)
        except:
            layer_bias.append(None)
        # count inbound and outbound nodes
        if idx != 0:
            if not is_seq:
                # is there a way to do this without the model config?
                num_inbound = np.shape(model_config['layers'][idx]['inbound_nodes'])[1]
                parents = []
                parents_idx = []
                for p_idx in range(num_inbound):
                    # is there a way to do this without the model config?
                    temp_name = model_config['layers'][idx]['inbound_nodes'][0][p_idx][0]
                    parents.append(temp_name)
                    parents_idx.append(layers_dict[temp_name])
                layer_parent_idx.append(parents_idx)
            else:
                layer_parent_idx.append([idx-1])
        else:
            layer_parent_idx.append([])

    # create child index
    layer_child_idx = [[] for i in range(len(layer_parent_idx))]
    for idx, parents in enumerate(layer_parent_idx):
        for x in parents:
            layer_child_idx[x].append(idx)

    # combine all of it nicely
    network_structure = []
    for idx in range(len(layer_parent_idx)):
        item = NetStructure(
            layer_parent_idx[idx],
            layer_child_idx[idx]
        )
        network_structure.append(item)

    return np.array(layer_types), np.array(layer_params), \
           layer_output_shape, layer_bias, network_structure


def delete_dense_neuron(new_model_param, layer_types, layer_output_shape,
                        layer_bias, layer, netstr, neuron):
    """
    Deletes a given neuron if the layer is a dense layer.

    Args:
        new_model_param:    Stores the current weights of the model
        layer_types:        If layer_types is dense, neuron will be removed
        layer_output_shape: Stores the current output shapes of all layers of
                            the model
        layer:              Integer of layer number (0,1,2, ...)
        netstr:             Stores the network structure (parent and child nodes, split and merge nodes)
        neuron:             Integer which says which neuron of the given layer
                            (if dense) should be deleted

    Return:
        new_model_param:    New model params after deleting a neuron
        layer_output_shape: New output shapes of the model
    """
    '''If the current layer is a dense layer, weights and the bias are removed
       for the given layer and neuron.'''
    if layer_types[layer] == "Dense":
        new_model_param[layer][0] = np.delete(
            new_model_param[layer][0], neuron, axis=1)  # Weight
        if layer_bias[layer]:
            new_model_param[layer][1] = np.delete(
                new_model_param[layer][1], neuron, axis=0)  # Bias

        "The new output shape of the layer is restored"
        layer_output_shape[layer][1] = get_layer_shape_dense(new_model_param,
                                                             layer)

        """Check if there is a dense layer after the current.
           The parameters of the next dense layer were connected
           to the removed neuron and also have to be removed"""

        for i in range(layer + 1, len(new_model_param)):
            # check if layer is parent of (layer + i) - important for non-sequential models
            # TODO check if anything here is necessary, since there are usually no skip connections after dense layers
            #if layer not in netstr[i].parents:
            #    continue

            if layer_types[i] == "Dense":
                # Parameters also have to be deleted from next weight matrix
                new_model_param[i][0] = np.delete(
                    new_model_param[i][0], neuron, axis=0)
                return new_model_param, layer_output_shape

            """If there is a layer with no parameters like max_pool between the
               current and the next dense layer the output neurons are the same
               as those of the current dense layer"""
            if np.array(new_model_param[i]).size == 0:
                layer_output_shape[i][1] = get_layer_shape_dense(
                    new_model_param, layer)

    else:
        print("No dense layer")

    return new_model_param, layer_output_shape


def delete_filter(new_model_param, layer_types, layer_output_shape,
                  layer_bias, layer, netstr, filter, collected_indices):
    """
    Deletes a given filter if the layer is a conv layer.

    Args:
        new_model_param:    Stores the current weights of the model
        layer_types:        If layer_types is Conv2D, filter will be removed
        layer_output_shape: Stores the current output shapes of all layers of
                            the model
        layer:              Integer of layer number
        netstr:             Stores the network structure (parent and child nodes, split and merge nodes)
        filter:             Integer which says which filter of the given layer
                            (if conv) should be deleted

    Return:
        new_model_param:    New model params after deleting a filter
        layer_output_shape: New output shapes of the model
    """
    '''Don't change filters of output layers'''
    if len(netstr[layer].children) == 0:  # output conv layers don't have child nodes
        print("Skipping output conv layer")
        return new_model_param, layer_output_shape, collected_indices

    '''If the current layer is a conv layer, weights and the bias are removed
       for the given layer and filter'''
    if layer_types[layer] == "Conv2D":
        new_model_param[layer][0] = np.delete(
            new_model_param[layer][0], filter, axis=3)  # Delete Filter
        if layer_bias[layer]:
            new_model_param[layer][1] = np.delete(
                new_model_param[layer][1], filter, axis=0)  # Delete Bias

        "The new output shape of the layer is restored"
        layer_output_shape[layer][3] = get_layer_shape_conv(new_model_param,
                                                            layer)

        """Check if there is a dense/conv layer after the current.
           The parameters of the next dense layer were connected to the
           removed neuron and also have to be removed"""

        if len(netstr[layer].children) == 1:
            next_layer = netstr[layer].children[0]
        else:
            print("Something is wrong: non-output conv layer should have exactly 1 child!")
            return new_model_param, layer_output_shape, collected_indices

        # contains all branches that need processing as tuples (start_node, parent_node, child_node)
        branches_to_process = []
        for ch in netstr[next_layer].children:
            branches_to_process.append([next_layer, layer, ch])

        while len(branches_to_process) != 0: # TODO replace by queue
            next_branch = branches_to_process[0]
            del branches_to_process[0]

            terminated = False
            current_layer_idx = next_branch[0]
            parent_idx = next_branch[1]
            child_idx = next_branch[2]
            channel_offset = 0  # used to correct filter index after concatenation layers

            while not terminated:
                if len(new_model_param[current_layer_idx]) != 0:
                    if layer_types[current_layer_idx] == "Dense":
                        new_model_param[current_layer_idx][0] = np.delete(new_model_param[current_layer_idx][0], filter,axis=0)
                        terminated = True
                    if layer_bias[current_layer_idx] is None:
                        for i in range(0, len(new_model_param[current_layer_idx])):
                            new_model_param[current_layer_idx][i] = np.delete(new_model_param[current_layer_idx][i], filter,axis=0)
                        layer_output_shape[current_layer_idx][3] = get_layer_shape_conv(new_model_param, layer)
                    else:
                        if layer_types[current_layer_idx] == "Conv2D":
                            # NOTE: do not remove filters, just collect their indices
                            if current_layer_idx not in collected_indices:
                               collected_indices[current_layer_idx] = [filter + channel_offset]
                            else:
                               collected_indices[current_layer_idx].append(filter + channel_offset)
                            terminated = True
                        else:
                            new_model_param[current_layer_idx][0] = np.delete(new_model_param[current_layer_idx][0],
                                                                              filter, axis=2)
                            layer_output_shape[current_layer_idx][3] = get_layer_shape_conv(new_model_param, layer)
                else:
                    if layer_types[current_layer_idx] == "Dense":
                        terminated = True # this case should never happen: dense layers have params
                    elif layer_types[current_layer_idx] == "Flatten":
                        layer_output_shape[current_layer_idx][1] = np.prod(layer_output_shape[parent_idx][1:4])
                        for i in range(np.multiply(np.prod(layer_output_shape[parent_idx][1:3]), filter - 1),
                                       np.multiply(np.prod(layer_output_shape[parent_idx][1:3]), filter)):
                            new_model_param[child_idx][0] = np.delete(new_model_param[child_idx][0], i, axis=0)
                        terminated = True
                    elif layer_types[current_layer_idx] == "Concatenate":
                        channel_sum = 0
                        for cp_idx in netstr[current_layer_idx].parents:
                            if cp_idx == parent_idx:
                                channel_offset = channel_sum
                            channel_sum += layer_output_shape[cp_idx][3]
                        layer_output_shape[current_layer_idx][3] = channel_sum
                    else:
                        if len(layer_output_shape[current_layer_idx]) == 4:
                            layer_output_shape[current_layer_idx][3] = (get_layer_shape_conv(new_model_param, layer))
                        elif len(layer_output_shape[current_layer_idx]) == 2:
                            layer_output_shape[current_layer_idx][1] = (get_layer_shape_conv(new_model_param, layer))

                # update values for next step
                if child_idx == -1:  # we have processed the last output layer
                    terminated = True
                else:
                    parent_idx = current_layer_idx
                    current_layer_idx = child_idx
                    new_child_nodes = netstr[current_layer_idx].children

                    if len(new_child_nodes) > 1:
                        for nci in new_child_nodes[1:]:
                            branches_to_process.append([current_layer_idx, parent_idx, nci])
                    if len(new_child_nodes) == 0:
                        child_idx = -1
                    else:
                        child_idx = new_child_nodes[0]
    else:
        print("No conv layer")

    return new_model_param, layer_output_shape, collected_indices


def get_neurons_to_prune_l1(layer_params, prun_layer, prun_factor):
    """
    Calculate the neurons which get pruned with the L1 norm.

    Args:
        layer_params:   Stores the current weights of the model
        prun_layer:     Integer of layer number
        prun_factor:    Integer which says how many percent of the dense
                        neurons should be deleted

    Return:
        prune_neurons:  Get indizies of neurons to prune
        num_new_neuron: New shape of the weight Matrix
    """
    new_layer_param = layer_params[prun_layer]
    avg_neuron_w = []

    """Absolute average of the weights arriving at a neuron are written into
       an array."""
    for i in range(0, new_layer_param[0].shape[-1]):
        avg_neuron_w.append(np.average(np.abs(
            new_layer_param[0][:, i])))

    """Absolute average of the weights are sorted and a percantage of these
       which is given through the prune factor are stored in prune_neurons,
       these neurons will be pruned."""
    prun_neurons = sorted(range(new_layer_param[0].shape[-1]), key=lambda k:
                          avg_neuron_w[k])[:int((
                              prun_factor * new_layer_param[0].shape[-1]) /
                              100)]
    prun_neurons = np.sort(prun_neurons)

    "The number of the new units of the dense layer are stored"
    num_new_neurons = new_layer_param[0].shape[-1] - len(prun_neurons)
    return prun_neurons, num_new_neurons


def get_neurons_to_prune_l2(layer_params, prun_layer, prun_factor):
    """
    Calculate the neurons which get pruned with the L2 norm.

    Args:
        layer_params:   Stores the current weights of the model
        prun_layer:     Integer of layer number
        prun_factor:    Integer which says how many percent of the dense
                        neurons should be deleted

    Return:
        prune_neurons:  Get indizies of neurons to prune
        num_new_neuron: New shape of the weight Matrix
    """
    new_layer_param = layer_params[prun_layer]
    avg_neuron_w = []

    """Absolute average of the weights arriving at a neuron are written into
       an array"""
    for i in range(0, new_layer_param[0].shape[-1]):
        avg_neuron_w.append(np.linalg.norm(new_layer_param[0][:, i]))

    """Absolute average of the weights are sorted and a percantage of these
       which is given through the prune factor are stored in prune_neurons,
       these neurons will be pruned"""
    prun_neurons = sorted(range(new_layer_param[0].shape[-1]), key=lambda k:
                          avg_neuron_w[k])[:int((
                              prun_factor * new_layer_param[0].shape[-1]) /
                              100)]
    prun_neurons = np.sort(prun_neurons)

    "The number of the new units of the dense layer are stored"
    num_new_neurons = new_layer_param[0].shape[-1] - len(prun_neurons)
    return prun_neurons, num_new_neurons


def prun_neurons_dense(layer_types, layer_params, layer_output_shape,
                       layer_bias, prun_layer, netstr, prun_factor,
                       metric):
    """
    Deletes neurons from the dense layer. The prun_factor is telling how much
    percent of the neurons of the dense layer should be deleted.

    Args:
        layer_types:        If layer_types is dense neurons will be removed
        layer_params:       Stores the current weights of the model
        layer_output_shape: Stores the current output shapes of all layers of
                            the model
        netstr:             Stores the network structure (parent and child nodes, split and merge nodes)
        prun_layer:         Integer of layer number
        prun_factor:        Integer which says how many percent of the dense
                            neurons should be deleted

    Return:
        new_model_param:    New model params after deleting the neurons
        num_new_neurons:    New number of neurons of the dense layers
        layer_output_shape: New output shapes of the model
    """
    'Check if layer to prune is a Dense layer'
    if layer_types[prun_layer] != "Dense":
        print("No dense layer!")
        return None, None, None

    if prun_factor > 0:
        if metric == 'L1':
            prun_neurons, num_new_neurons = get_neurons_to_prune_l1(
                layer_params, prun_layer, prun_factor)
        elif metric == 'L2':
            prun_neurons, num_new_neurons = get_neurons_to_prune_l2(
                layer_params, prun_layer, prun_factor)
        else:
            prun_neurons, num_new_neurons = get_neurons_to_prune_l1(
                layer_params, prun_layer, prun_factor)

        "Deleting neurons, beginning with the neuron with the highest index"
        if len(prun_neurons) > 0:
            for i in range(len(prun_neurons) - 1, -1, -1):
                new_model_param, layer_output_shape = delete_dense_neuron(
                    layer_params, layer_types, layer_output_shape, layer_bias,
                    prun_layer, netstr, prun_neurons[i])

        else:
            new_model_param = layer_params
            print("No neurons to prune increase prune factor for dense layers")

    else:
        new_model_param = layer_params
        num_new_neurons = layer_params[prun_layer][0].shape[-1]
        print("No pruning implemented for dense layers")

    return new_model_param, num_new_neurons, layer_output_shape


def get_filter_to_prune_avarage(layer_params, prun_layer, prun_factor):
    """
    Calculate the filters which get pruned by average.

    Args:
        layer_params:   Stores the current weights of the model
        prun_layer:     Integer of layer number
        prun_factor:    Integer which says how many percent of the dense
                        neurons should be deleted

    Return:
        prun_filter:    Get indizies of filter to prune
        num_new_filter: New shape of the weight Matrix
    """
    '''Load the filters of the conv layer and add a array where the
       absolut average filter values will be stored'''
    filters = layer_params[prun_layer]
    avg_filter_w = []
    "Absolute average of the filter values are written into an array"
    for i in range(0, filters[0].shape[-1]):
        avg_filter_w.append(np.average(np.abs(filters[0][:, :, :, i])))

    """Absolute average of the filter values are sorted and a percantage of
       these which is given through the prune factor are stored in
       prune_filters, these filters will be pruned"""
    prun_filter = sorted(range(filters[0].shape[-1]), key=lambda k:
                         avg_filter_w[k])[:int((
                             prun_factor * filters[0].shape[-1]) /
                             100)]
    prun_filter = np.sort(prun_filter)

    "The number of the new filters of the conv layer are stored"
    num_new_filter = filters[0].shape[-1] - len(prun_filter)
    return prun_filter, num_new_filter


def get_filter_to_prune_l2(layer_params, prun_layer, prun_factor):
    """
    Calculate the filters which get pruned with the L2 norm.

    Args:
        layer_params:   Stores the current weights of the model
        prun_layer:     Integer of layer number
        prun_factor:    Integer which says how many percent of the dense
                        neurons should be deleted

    Return:
        prun_filter:    Get indizies of filter to prune
        num_new_filter: New shape of the weight Matrix
    """
    '''Load the filters of the conv layer and add a array where the
       absolut average filter values will be stored'''
    filters = layer_params[prun_layer]
    avg_filter_w = []
    "Absolute average of the filter values are written into an array"
    for i in range(0, filters[0].shape[-1]):
        avg_filter_w.append(np.linalg.norm(filters[0][:, :, :, i]))

    """Absolute average of the filter values are sorted and a percantage of
       these which is given through the prune factor are stored in
       prune_filters, these filters will be pruned"""
    prun_filter = sorted(range(filters[0].shape[-1]), key=lambda k:
                         avg_filter_w[k])[:int((
                             prun_factor * filters[0].shape[-1]) /
                             100)]
    prun_filter = np.sort(prun_filter)

    "The number of the new filters of the conv layer are stored"
    num_new_filter = filters[0].shape[-1] - len(prun_filter)
    return prun_filter, num_new_filter


def prun_filters_conv(layer_types, layer_params, layer_output_shape,
                      layer_bias, prun_layer, netstr, prun_factor, collected_indices,
                      metric='L1'):
    """
    Deletes filters from the conv layer. The prun_factor is telling how much
    percent of the filters of the conv layer should be deleted.

    Args:
        layer_types:        If layer_types is Conv2D, filters will be removed
        layer_params:       Stores the current weights of the model
        layer_output_shape: Stores the current output shapes of all layers of
                            the model
        netstr:             Stores the network structure (parent and child nodes, split and merge nodes)
        prun_layer:         Integer of layer number
        prun_factor:        Integer which says how many percent of the filters
                            should be deleted

    Return:
        new_model_param:    New model params after deleting the filters
        num_new_filters:    New number of filters of the conv layers
        layer_output_shape: New output shapes of the model
    """
    'Check if layer to prune is a Conv layer'
    if layer_types[prun_layer] != "Conv2D":
        print("No Conv layer!")
        return None, None

    # second condition prevents pruning output layers with parameters
    if prun_factor > 0 and len(netstr[prun_layer].children) > 0:
        if metric == 'L1':
            prun_filter, num_new_filter = get_filter_to_prune_avarage(
                layer_params, prun_layer, prun_factor)
        elif metric == 'L2':
            prun_filter, num_new_filter = get_filter_to_prune_l2(
                layer_params, prun_layer, prun_factor)
        else:
            prun_filter, num_new_filter = get_filter_to_prune_avarage(
                layer_params, prun_layer, prun_factor)

        """Deleting the filters, beginning with the filter with the highest
           index"""
        if len(prun_filter) > 0:
            for i in range(len(prun_filter) - 1, -1, -1):
                new_model_param, layer_output_shape, collected_indices = delete_filter(
                    layer_params, layer_types, layer_output_shape,
                    layer_bias, prun_layer, netstr, prun_filter[i], collected_indices)
            # NOTE here layer_output_shape is correct, but new_model_param has the old shape (input dimension is wrong)
        else:
            new_model_param = layer_params
            num_new_filter = layer_params[prun_layer][0].shape[-1]
            print("No filter to prune increase prune factor for conv layers")

    else:
        new_model_param = layer_params
        num_new_filter = layer_params[prun_layer][0].shape[-1]
        collected_indices = collected_indices

    return new_model_param, num_new_filter, layer_output_shape, collected_indices


def model_pruning(layer_types, layer_params, layer_output_shape, layer_bias,
                  netstr, num_new_neurons, num_new_filters,
                  prun_factor_dense, prun_factor_conv, metric):
    """
    Deletes neurons and filters from all dense and conv layers. The two
    prunfactors are telling how much percent of the neurons and the filters
    should be deleted.

    Args:
        layer_types:        The types of all layers of the model
        layer_params:       Stores the current weights of the model
        layer_output_shape: Stores the current output shapes of all layers of
                            the model
        layer_bias:	   Stores the current biases of the model
        netstr:             Stores the network structure (parent and child nodes, split and merge nodes)
        num_new_neurons:    Number of neurons of the dense layers
        num_new_filters:    Number of filters of the conv layers
        prun_factor_dense:  Integer which says how many percent of the neurons
                            should be deleted
        prun_factor_conv:   Integer which says how many percent of the filters
                            should be deleted
        metric:             Metric which should be used for model pruning            

    Return:
        layer_params:       New model params after deleting the neurons and
                            filters
        num_new_neurons:    New number of filters of the dense layers
        num_new_filters:    New number of filters of the conv layers
        layer_output_shape: New output shapes of the model after deleting
                            neurons and filters
    """

    collected_indices = dict()
    original_layer_output_shape = copy.deepcopy(layer_output_shape)

    for i in range(0, get_last_layer_with_params(layer_params)):
        if layer_types[i] == "Dense":
            layer_params, num_new_neurons[i], layer_output_shape = (
                prun_neurons_dense(layer_types, layer_params,
                                   layer_output_shape, layer_bias, i,
                                   netstr,
                                   prun_factor_dense, metric))
        elif layer_types[i] == "Conv2D":
            layer_params, num_new_filters[i], layer_output_shape, collected_indices = (
                prun_filters_conv(layer_types, layer_params,
                                  layer_output_shape, layer_bias, i,
                                  netstr,
                                  prun_factor_conv, collected_indices, metric))
        else:
            ("No pruning for this layer")

    for key, val in collected_indices.items():
        num_orig_input_filters = original_layer_output_shape[netstr[key].parents[0]][3]
        num_new_input_filters = layer_output_shape[netstr[key].parents[0]][3]
        num_filters_to_delete = num_orig_input_filters - num_new_input_filters
        # TODO make this smarter: instead of taking the front of the list, first
        #      take all indices that appear multiple times in the list
        indices_to_delete = list(set(val))[:num_filters_to_delete]
        indices_to_delete = sorted(indices_to_delete, reverse=True)

        for del_filter_idx in indices_to_delete:
            layer_params[key][0] = np.delete(layer_params[key][0], del_filter_idx, axis=2)
            layer_output_shape[key][3] = layer_output_shape[netstr[key].parents[0]][3]
    del collected_indices

    return layer_params, num_new_neurons, num_new_filters, layer_output_shape


def build_pruned_model(model, new_model_param, layer_types, num_new_neurons,
                       num_new_filters, comp):
    """
    The new number of neurons and filters are changed in the model config.
    Load the new weight matrices into the model.

    Args:
        model:              Model which should be pruned
        new_model_param:    Stores the new weights of the model
        layer_types:        The types of all layers of the model
        num_new_neurons:    Number of neurons of the dense layers
        num_new_filters:    Number of filters of the conv layers

    Return:
        pruned_model:   New model after pruning all dense and conv layers
    """

    model_config = model.get_config()

    """
    For functional model first layer (fl) is the input layer.
    For sequential model the first layer is the layer after the input layer
    """
    fl = 1
    if layer_types[0] == 'InputLayer':
        fl = 0

    for i in range(0, get_last_layer_with_params(new_model_param)):
        if model_config['layers'][i + fl]['class_name'] == "Dense":
            #print("Dense")
            model_config['layers'][i + fl]['config']['units'] = (
                num_new_neurons[i])

        elif model_config['layers'][i + fl]['class_name'] == "Conv2D":
            #print("Conv2D")
            model_config['layers'][i + fl]['config']['filters'] = (
                num_new_filters[i])

        elif model_config['layers'][i + fl]['class_name'] == "Reshape":
            temp_list = list(model_config['layers'][i + fl]['config']
                             ['target_shape'])
            cur_layer = i
            cur_filters = num_new_filters[cur_layer]
            # Get number of filters of last Conv layer
            if cur_filters == 0:
                while cur_filters == 0:
                    cur_layer -= 1
                    cur_filters = num_new_filters[cur_layer]
            temp_list[2] = cur_filters
            temp_tuple = tuple(temp_list)
            model_config['layers'][i + fl]['config']['target_shape'] = (
                temp_tuple)
        else:
            pass
            #print("No Dense or Conv2D")

    print("Before pruning:")
    model.summary()

    if "Sequential" in str(model):
        pruned_model = Sequential.from_config(model_config)
    elif "Functional" in str(model):
        pruned_model = Model.from_config(model_config)

    print("After pruning:")
    pruned_model.summary()

    for i in range(0, len(pruned_model.layers)):
        if len(new_model_param[i]) != 0:
            pruned_model.layers[i].set_weights(new_model_param[i])
        else:
            None

    if comp is not None:
        pruned_model.compile(**comp)

    return pruned_model


def pruning(keras_model, x_train, y_train, comp, fit, prun_factor_dense=10,
            prun_factor_conv=10, metric='L1'):
    """
    A given keras model get pruned. The factor for dense and conv says how
    many percent of the dense and conv layers should be deleted. After pruning
    the model will be retrained.

    Args:
        keras_model:        Model which should be pruned
        x_train:            Training data to retrain the model after pruning
        y_train:            Labels of training data to retrain the model after
                            pruning
        prun_factor_dense:  Integer which says how many percent of the neurons
                            should be deleted
        prun_factor_conv:   Integer which says how many percent of the filters
                            should be deleted

    Return:
        pruned_model:   New model after pruning and retraining
    """

    if callable(getattr(keras_model, "predict", None)):
        model = keras_model
    elif isinstance(keras_model, str) and ".h5" in keras_model:
        model = load_model(keras_model)
    else:
        print("No model given to prune")

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

    pruned_model.fit(x_train, y_train, **fit)

    return pruned_model


def pruning_for_acc(keras_model, x_train, x_val_y_train, comp,
                    pruning_acc=None, max_acc_loss=5, num_classes=None,
                    label_one_hot=None, data_loader_path=None):
    """
    A given keras model gets pruned. Either an accuracy value (in %) can be
    specified, which the minimized model has to still achieve. Or the maximum
    loss of accuracy (in %) that the minimized model may experience. The model
    is reduced step by step until the accuracy value is under run or the
    accuracy loss is exceeded.

    Args:
        keras_model:        Model which should be pruned
        x_train:            Training data to retrain the model after pruning
        x_val_y_train:      Labels of training data or validation data to
                            retrain the model after pruning (depends on whether
                            the data is a data loader or a numpy array)
        comp:               Compiler settings
        pruning_acc:        Integer which says which accuracy value (in %)
                            should not be fall below. If pruning_acc is not
                            defined, default is -5%
        max_acc_loss:       Integer which says which accuracy loss (in %)
                            should not be exceed
        num_classes:        Number of different classes of the model
        label_one_hot:      Boolean value if labels are one hot encoded or not
        data_loader_path:   Path of the folder or file with the training data

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

    if pruning_acc is not None:
        req_acc = pruning_acc / 100
    else:
        if os.path.isfile(data_loader_path):
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, x_val_y_train, test_size=0.2)
            original_model_acc = original_model.evaluate(
                x_val, y_val)[-1]
        elif os.path.isdir(data_loader_path):
            original_model_acc = original_model.evaluate(x_val_y_train)[-1]
        print("Start model accuracy: " + str(original_model_acc * 100) + "%")
        req_acc = original_model_acc - (max_acc_loss / 100)

    train_epochs = 10
    threshold = ThresholdCallback(req_acc)
    callbacks = [threshold]

    while pruning_factor <= 95:

        print("Next pruning factors: " + str(pruning_factor))

        model = prune_model(original_model, prun_factor_dense=pruning_factor,
                            prun_factor_conv=pruning_factor, metric='L1',
                            comp=comp, num_classes=num_classes,
                            label_one_hot=label_one_hot)

        if os.path.isfile(data_loader_path):
            history = model.fit(x=x_train, y=y_train, batch_size=64,
                                validation_data=(x_val, y_val),
                                epochs=train_epochs, callbacks=callbacks)
        elif os.path.isdir(data_loader_path):
            history = model.fit_generator(
                x_train, steps_per_epoch=len(x_train),
                validation_data=x_val_y_train,
                validation_steps=len(x_val_y_train),
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


def prune_model(keras_model, prun_factor_dense=10, prun_factor_conv=10,
                metric='L1', comp=None, num_classes=None, label_one_hot=None):
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
        num_classes:        Number of different classes of the model
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

    if num_classes <= 2 and comp is None:
        comp = {
            "optimizer": 'adam',
            "loss": tf.keras.losses.BinaryCrossentropy(),
            "metrics": 'accuracy'}
    elif num_classes > 3 and comp is None:
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
