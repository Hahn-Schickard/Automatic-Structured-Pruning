"""Copyright [2020] Hahn-Schickard-Gesellschaft fuer angewandte Forschung e.V.,
                    Daniel Konegen + Marcus Rueb
   Copyright [2021] Karlsruhe Institute of Technology, Daniel Konegen
   Copyright [2022] Hahn-Schickard-Gesellschaft fuer angewandte Forschung e.V.,
                    Daniel Konegen + Marcus Rueb
   SPDX-License-Identifier: Apache-2.0
============================================================================"""

import numpy as np


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


def delete_filter(
    new_model_param,
    layer_types,
    layer_output_shape,
    layer_bias,
    layer,
    netstr,
    filter,
    collected_indices,
):
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
    """Don't change filters of output layers"""
    if len(netstr[layer].children) == 0:  # output conv layers don't have child nodes
        print("Skipping output conv layer")
        return new_model_param, layer_output_shape, collected_indices

    """If the current layer is a conv layer, weights and the bias are removed
       for the given layer and filter"""
    if layer_types[layer] == "Conv2D":
        new_model_param[layer][0] = np.delete(
            new_model_param[layer][0], filter, axis=3
        )  # Delete Filter
        if layer_bias[layer]:
            new_model_param[layer][1] = np.delete(
                new_model_param[layer][1], filter, axis=0
            )  # Delete Bias

        "The new output shape of the layer is restored"
        layer_output_shape[layer][3] = get_layer_shape_conv(new_model_param, layer)

        """Check if there is a dense/conv layer after the current.
           The parameters of the next dense layer were connected to the
           removed neuron and also have to be removed"""

        if len(netstr[layer].children) == 1:
            next_layer = netstr[layer].children[0]
        else:
            print(
                "Something is wrong: non-output conv layer should have exactly 1 child!"
            )
            return new_model_param, layer_output_shape, collected_indices

        # contains all branches that need processing as tuples (start_node, parent_node, child_node)
        branches_to_process = []
        for ch in netstr[next_layer].children:
            branches_to_process.append([next_layer, layer, ch])

        while len(branches_to_process) != 0:  # TODO replace by queue
            next_branch = branches_to_process[0]
            del branches_to_process[0]

            terminated = False
            current_layer_idx = next_branch[0]
            parent_idx = next_branch[1]
            child_idx = next_branch[2]
            channel_offset = (
                0  # used to correct filter index after concatenation layers
            )

            while not terminated:
                if len(new_model_param[current_layer_idx]) != 0:
                    if layer_types[current_layer_idx] == "Dense":
                        new_model_param[current_layer_idx][0] = np.delete(
                            new_model_param[current_layer_idx][0], filter, axis=0
                        )
                        terminated = True
                    if layer_bias[current_layer_idx] is None:
                        for i in range(0, len(new_model_param[current_layer_idx])):
                            new_model_param[current_layer_idx][i] = np.delete(
                                new_model_param[current_layer_idx][i], filter, axis=0
                            )
                        layer_output_shape[current_layer_idx][3] = get_layer_shape_conv(
                            new_model_param, layer
                        )
                    else:
                        if layer_types[current_layer_idx] == "Conv2D":
                            # NOTE: do not remove filters, just collect their indices
                            if current_layer_idx not in collected_indices:
                                collected_indices[current_layer_idx] = [
                                    filter + channel_offset
                                ]
                            else:
                                collected_indices[current_layer_idx].append(
                                    filter + channel_offset
                                )
                            terminated = True
                        else:
                            new_model_param[current_layer_idx][0] = np.delete(
                                new_model_param[current_layer_idx][0], filter, axis=2
                            )
                            layer_output_shape[current_layer_idx][3] = (
                                get_layer_shape_conv(new_model_param, layer)
                            )
                else:
                    if layer_types[current_layer_idx] == "Dense":
                        terminated = True  # this case should never happen: dense layers have params
                    elif layer_types[current_layer_idx] == "Flatten":
                        layer_output_shape[current_layer_idx][1] = np.prod(
                            layer_output_shape[parent_idx][1:4]
                        )
                        for i in range(
                            np.multiply(
                                np.prod(layer_output_shape[parent_idx][1:3]), filter - 1
                            ),
                            np.multiply(
                                np.prod(layer_output_shape[parent_idx][1:3]), filter
                            ),
                        ):
                            new_model_param[child_idx][0] = np.delete(
                                new_model_param[child_idx][0], i, axis=0
                            )
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
                            layer_output_shape[current_layer_idx][3] = (
                                get_layer_shape_conv(new_model_param, layer)
                            )
                        elif len(layer_output_shape[current_layer_idx]) == 2:
                            layer_output_shape[current_layer_idx][1] = (
                                get_layer_shape_conv(new_model_param, layer)
                            )

                # update values for next step
                if child_idx == -1:  # we have processed the last output layer
                    terminated = True
                else:
                    parent_idx = current_layer_idx
                    current_layer_idx = child_idx
                    new_child_nodes = netstr[current_layer_idx].children

                    if len(new_child_nodes) > 1:
                        for nci in new_child_nodes[1:]:
                            branches_to_process.append(
                                [current_layer_idx, parent_idx, nci]
                            )
                    if len(new_child_nodes) == 0:
                        child_idx = -1
                    else:
                        child_idx = new_child_nodes[0]
    else:
        print("No conv layer")

    return new_model_param, layer_output_shape, collected_indices


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
    """Load the filters of the conv layer and add a array where the
       absolut average filter values will be stored"""
    filters = layer_params[prun_layer]
    avg_filter_w = []
    "Absolute average of the filter values are written into an array"
    for i in range(0, filters[0].shape[-1]):
        avg_filter_w.append(np.average(np.abs(filters[0][:, :, :, i])))

    """Absolute average of the filter values are sorted and a percantage of
       these which is given through the prune factor are stored in
       prune_filters, these filters will be pruned"""
    prun_filter = sorted(range(filters[0].shape[-1]), key=lambda k: avg_filter_w[k])[
        : int((prun_factor * filters[0].shape[-1]) / 100)
    ]
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
    """Load the filters of the conv layer and add a array where the
       absolut average filter values will be stored"""
    filters = layer_params[prun_layer]
    avg_filter_w = []
    "Absolute average of the filter values are written into an array"
    for i in range(0, filters[0].shape[-1]):
        avg_filter_w.append(np.linalg.norm(filters[0][:, :, :, i]))

    """Absolute average of the filter values are sorted and a percantage of
       these which is given through the prune factor are stored in
       prune_filters, these filters will be pruned"""
    prun_filter = sorted(range(filters[0].shape[-1]), key=lambda k: avg_filter_w[k])[
        : int((prun_factor * filters[0].shape[-1]) / 100)
    ]
    prun_filter = np.sort(prun_filter)

    "The number of the new filters of the conv layer are stored"
    num_new_filter = filters[0].shape[-1] - len(prun_filter)
    return prun_filter, num_new_filter


def prun_filters_conv(
    layer_types,
    layer_params,
    layer_output_shape,
    layer_bias,
    prun_layer,
    netstr,
    prun_factor,
    collected_indices,
    metric="L1",
):
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
    "Check if layer to prune is a Conv layer"
    if layer_types[prun_layer] != "Conv2D":
        print("No Conv layer!")
        return None, None

    # second condition prevents pruning output layers with parameters
    if prun_factor > 0 and len(netstr[prun_layer].children) > 0:
        if metric == "L1":
            prun_filter, num_new_filter = get_filter_to_prune_avarage(
                layer_params, prun_layer, prun_factor
            )
        elif metric == "L2":
            prun_filter, num_new_filter = get_filter_to_prune_l2(
                layer_params, prun_layer, prun_factor
            )
        else:
            prun_filter, num_new_filter = get_filter_to_prune_avarage(
                layer_params, prun_layer, prun_factor
            )

        """Deleting the filters, beginning with the filter with the highest
           index"""
        if len(prun_filter) > 0:
            for i in range(len(prun_filter) - 1, -1, -1):
                new_model_param, layer_output_shape, collected_indices = delete_filter(
                    layer_params,
                    layer_types,
                    layer_output_shape,
                    layer_bias,
                    prun_layer,
                    netstr,
                    prun_filter[i],
                    collected_indices,
                )
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
