'''Copyright [2020] Hahn-Schickard-Gesellschaft fuer angewandte Forschung e.V.,
                    Daniel Konegen + Marcus Rueb
   Copyright [2021] Karlsruhe Institute of Technology, Daniel Konegen
   Copyright [2022] Hahn-Schickard-Gesellschaft fuer angewandte Forschung e.V.,
                    Daniel Konegen + Marcus Rueb
   SPDX-License-Identifier: Apache-2.0
============================================================================'''

import numpy as np


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
