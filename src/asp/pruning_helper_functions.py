'''Copyright [2020] Hahn-Schickard-Gesellschaft fuer angewandte Forschung e.V.,
                    Daniel Konegen + Marcus Rueb
   Copyright [2021] Karlsruhe Institute of Technology, Daniel Konegen
   Copyright [2022] Hahn-Schickard-Gesellschaft fuer angewandte Forschung e.V.,
                    Daniel Konegen + Marcus Rueb
   SPDX-License-Identifier: Apache-2.0
============================================================================'''

import copy
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model

from .pruning_helper_classes import NetStructure
from .pruning_helper_functions_dense import prun_neurons_dense
from .pruning_helper_functions_conv import prun_filters_conv


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
