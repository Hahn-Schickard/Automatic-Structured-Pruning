'''Copyright [2020] Hahn-Schickard-Gesellschaft fuer angewandte Forschung e.V.,
                    Daniel Konegen + Marcus Rueb
   Copyright [2021] Karlsruhe Institute of Technology, Daniel Konegen
   Copyright [2022] Hahn-Schickard-Gesellschaft fuer angewandte Forschung e.V.,
                    Daniel Konegen + Marcus Rueb
   SPDX-License-Identifier: Apache-2.0
============================================================================'''

from .pruning import factor_pruning, accuracy_pruning, \
    stepwise_factor_pruning, stepwise_accuracy_pruning
from .pruning_helper_classes import NetStructure, ThresholdCallback
from .pruning_helper_functions import get_last_layer_with_params, \
    load_model_param, model_pruning, build_pruned_model
from .pruning_helper_functions_dense import get_layer_shape_dense, \
    delete_dense_neuron, get_neurons_to_prune_l1, get_neurons_to_prune_l2, \
        prun_neurons_dense
from .pruning_helper_functions_conv import get_layer_shape_conv, \
    delete_filter, get_filter_to_prune_avarage, get_filter_to_prune_l2, \
        prun_filters_conv

from pkg_resources import get_distribution, DistributionNotFound


# read the current version of the library from the setup.cfg
__version__ = ''
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass  # package is not installed

# remove links to functions which are only needed inside the __init__.py
del get_distribution, DistributionNotFound
