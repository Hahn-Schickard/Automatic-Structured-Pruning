#PRUNING NEU

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
import numpy as np
from keras.callbacks import EarlyStopping
import tensorflow as tf


def get_layer_shape_dense(new_model_param,layer):	
    """	
    Gets the struture of the new generated model and return the shape of the current layer	
    	
    Args: 	
        new_model_param: The params of the new generated model	
        layer: the current layer we want the shape from	
            	
    Return: 	
        shape of the current layer	
    """	
    return new_model_param[layer][0].shape[1]	


def get_layer_shape_conv(new_model_param,layer):	
    """	
    Gets the struture of the new generated model and return the shape of the current layer	
    	
    Args: 	
        new_model_param: The params of the new generated model	
        layer: the current layer we want the shape from	
            	
    Return: 	
        shape of the current layer	
    """	
    return new_model_param[layer][0].shape[3]



def load_model_param(model):
    """
    Gets layer names, layer weights and output_shape of each layer from the given keras model.
    The weights of all layers are stored in layer_params. This array will be used to delete the neurons and reload
    the weights later
    The type of all layers are stored in layer_types to search for dense and conv layers.
    The output shape of each layer is also needed to set the right number of parameters in layers like max_pool
    
    Args: 
        model: Model which should be pruned
            
    Return: 
        layer_types (np.array): Type of all layers of the model	
        layer_params (np.array): All weight matrices of the model	
        layer_output_shape (list): Output shape of all layers of the model
    """
    
    layer_params = []
    layer_types = []
    layer_output_shape = []
    layer_bias = []

    for layer in model.layers:	
        layer_types.append(layer.__class__.__name__)	
        layer_params.append(layer.get_weights())	
        layer_output_shape.append(list(layer.output_shape))
        try:
            layer_bias.append(layer.use_bias)
        except:
            layer_bias.append(None)
        
    return np.array(layer_types), np.array(layer_params), layer_output_shape, layer_bias



def delete_dense_neuron(new_model_param, layer_types, layer_output_shape, layer_bias, layer, neuron):
    """
    Deletes a given neuron if the layer is a dense layer
    
    Args: 
        new_model_param: Stores the current weights of the model
        layer_types: If layer_types is dense, neuron will be removed
        layer_output_shape: Stores the current output shapes of all layers of the model
        layer: Integer of layer number (0,1,2, ...)
        neuron: Integer which says which neuron of the given layer (if dense) should be deleted
            
    Return: 
        new_model_param: New model params after deleting a neuron
        layer_output_shape: New output shapes of the model
    """
    
    "If the current layer is a dense layer, weights and the bias are removed for the given layer and neuron"
    if layer_types[layer] == "Dense":
        new_model_param[layer][0] = np.delete(new_model_param[layer][0], neuron, axis=1)   #weight
        if layer_bias[layer] == True:
            new_model_param[layer][1] = np.delete(new_model_param[layer][1], neuron, axis=0)   #Bias
        
        "The new output shape of the layer is restored"
        layer_output_shape[layer][1] = get_layer_shape_dense(new_model_param, layer)
        
        "Check if there is a dense layer after the current. The parameters of the next dense layer were connected"	
        "to the removed neuron and also have to be removed"	
        #layer_index = 0	
        "If there is a layer with no parameters like max_pool between the current and the next dense layer"	
        "the output neurons are the same as those of the current dense layer" 
        
        for i in range(layer+1,len(new_model_param)):
            if layer_types[i] == "Dense":
                new_model_param[i][0] = np.delete(new_model_param[i][0], neuron, axis=0)   #Parameter müssen auch aus nächster Gewichtsmatrix gelöscht werden
                return new_model_param, layer_output_shape
            
            "If there is a layer with no parameters like max_pool between the current and the next dense layer"
            "the output neurons are the same as those of the current dense layer"            
            if np.array(new_model_param[i]).size == 0:
                layer_output_shape[i][1] = get_layer_shape_dense(new_model_param, layer)
            
    else:
        print("No dense layer")
        
    return new_model_param, layer_output_shape
    
def delete_filter(new_model_param, layer_types, layer_output_shape, layer_bias, layer, filter):
    """
    Deletes a given filter if the layer is a conv layer
    
    Args: 
        new_model_param: Stores the current weights of the model
        layer_types: If layer_types is Conv2D, filter will be removed
        layer_output_shape: Stores the current output shapes of all layers of the model
        layer: Integer of layer number
        filter: Integer which says which filter of the given layer (if conv) should be deleted
            
    Return: 
        new_model_param: New model params after deleting a filter
        layer_output_shape: New output shapes of the model
    """
    
    
    "If the current layer is a conv layer, weights and the bias are removed for the given layer and filter"
    if layer_types[layer] == "Conv2D":
        new_model_param[layer][0] = np.delete(new_model_param[layer][0], filter, axis=3)   #Delete Filter
        if layer_bias[layer] == True:
            new_model_param[layer][1] = np.delete(new_model_param[layer][1], filter, axis=0)   #Delete Bias
        
        
        "The new output shape of the layer is restored"
        layer_output_shape[layer][3] = get_layer_shape_conv(new_model_param, layer)
        
        "Check if there is a dense/conv layer after the current. The parameters of the next dense layer were connected"
        "to the removed neuron and also have to be removed"
        for dense_layer in range(layer+1,len(new_model_param)):
            
            if layer_types[dense_layer] == "Conv2D":
                new_model_param[dense_layer][0] = np.delete(new_model_param[dense_layer][0], filter, axis=2)
                return new_model_param, layer_output_shape
            
            
            elif layer_types[dense_layer] == "Dense":
                print(layer_output_shape[dense_layer-2][1]*layer_output_shape[dense_layer-2][2])
                print(filter)
                for j in range(0,layer_output_shape[dense_layer-2][1]*layer_output_shape[dense_layer-2][2]):   #layer before is flatten, we need output shape before layer flatten
                    new_model_param[dense_layer][0] = np.delete(new_model_param[dense_layer][0], filter, axis=0)
                return new_model_param, layer_output_shape
            
            elif np.array(new_model_param[dense_layer]).size == 0:
                for next_layer in range(dense_layer+1,len(new_model_param)):
                    if layer_types[next_layer] == "Conv2D": 
                        layer_output_shape[dense_layer][3] = get_layer_shape_conv(new_model_param, layer)
                        break
                    elif "flatten" in layer_types[next_layer]:
                        layer_output_shape[dense_layer][3] = get_layer_shape_conv(new_model_param, layer)
                        #layer_output_shape[next_layer][1] = layer_output_shape[dense_layer][1] * layer_output_shape[dense_layer][2] * layer_output_shape[dense_layer][3]
                        layer_output_shape[next_layer][1] = np.prod(layer_output_shape[dense_layer][1:4])
                        break
            
    else:
        print("No conv layer")
    
    return new_model_param, layer_output_shape


def get_neuros_to_prune_l1(layer_params,prun_layer,prun_factor):
    """
    Calculate the neurons who get Pruned with the L1 Norm 
    
    Args:
        layer_params: Stores the current weights of the model
        prun_layer: Integer of layer number
        prun_factor: Integer which says how many percent of the dense neurons should be deleted    
            
    Return: 
        prune_neurons: get indizies of neurons to prune
        num_new_neuron: New shape of the weight Matrix
    """
    new_layer_param = layer_params[prun_layer]
    avg_neuron_w = []

    'Absolute average of the weights arriving at a neuron are written into an array'
    for i in range (0,new_layer_param[0].shape[-1]):
        avg_neuron_w.append(np.average(np.abs(new_layer_param[0][:,i]))) 

    
    'Absolute average of the weights are sorted and a percantage of these which is given'
    'through the prune factor are stored in prune_neurons, these neurons will be pruned'
    prun_neurons = sorted(range(new_layer_param[0].shape[-1]), key=lambda k: avg_neuron_w[k])[:int((prun_factor*new_layer_param[0].shape[-1])/100)]
    prun_neurons = np.sort(prun_neurons)

    'The number of the new units of the dense layer are stored'
    num_new_neurons = new_layer_param[0].shape[-1] - len(prun_neurons)
    return prun_neurons,num_new_neurons


def get_neuros_to_prune_l2(layer_params,prun_layer,prun_factor):
    """
    Calculate the neurons who get Pruned with the L1 Norm 
    
    Args:
        layer_params: Stores the current weights of the model
        prun_layer: Integer of layer number
        prun_factor: Integer which says how many percent of the dense neurons should be deleted    
            
    Return: 
        prune_neurons: get indizies of neurons to prune
        num_new_neuron: New shape of the weight Matrix
    """    
    new_layer_param = layer_params[prun_layer]
    avg_neuron_w = []

    'Absolute average of the weights arriving at a neuron are written into an array'
    for i in range (0,new_layer_param[0].shape[-1]):
        avg_neuron_w.append(np.linalg.norm(new_layer_param[0][:,i])) 

    
    'Absolute average of the weights are sorted and a percantage of these which is given'
    'through the prune factor are stored in prune_neurons, these neurons will be pruned'
    prun_neurons = sorted(range(new_layer_param[0].shape[-1]), key=lambda k: avg_neuron_w[k])[:int((prun_factor*new_layer_param[0].shape[-1])/100)]
    prun_neurons = np.sort(prun_neurons)

    'The number of the new units of the dense layer are stored'
    num_new_neurons = new_layer_param[0].shape[-1] - len(prun_neurons)
    return prun_neurons,num_new_neurons




def prun_neurons_dense(layer_types, layer_params, layer_output_shape, layer_bias, prun_layer, prun_factor,metric):
    """
    Deletes neurons from the dense layer. The prun_factor is telling how much percent of the 
    neurons of the dense layer should be deleted.
    
    Args: 
        layer_types: If layer_types is dense neurons will be removed
        layer_params: Stores the current weights of the model
        layer_output_shape: Stores the current output shapes of all layers of the model
        prun_layer: Integer of layer number
        prun_factor: Integer which says how many percent of the dense neurons should be deleted
        
    Return: 
        new_model_param: New model params after deleting the neurons
        num_new_neurons: New number of neurons of the dense layers
        layer_output_shape: New output shapes of the model
    """
    
    'Check if layer to prune is a Dense layer'
    if layer_types[prun_layer] != "Dense":
        print("No dense layer!")
        return None, None
    
    if prun_factor > 0:
        'Load the weights of the dense layer and add an array where the' 
        'absolut average of the weights for each neurons will be stored'
        new_layer_param = layer_params[prun_layer]
        avg_neuron_w = []

        if metric == 'L1':
            prun_neurons,num_new_neurons=get_neuros_to_prune_l1(layer_params,prun_layer,prun_factor)
        elif metric == 'L2':
            prun_neurons,num_new_neurons=get_neuros_to_prune_l2(layer_params,prun_layer,prun_factor)
        else:
            prun_neurons,num_new_neurons=get_neuros_to_prune_l1(layer_params,prun_layer,prun_factor)

        '''
        'Absolute average of the weights arriving at a neuron are written into an array'
        for i in range (0,new_layer_param[0].shape[-1]):
            avg_neuron_w.append(np.average(np.abs(new_layer_param[0][:,i]))) 

        'Absolute average of the weights are sorted and a percantage of these which is given'
        'through the prune factor are stored in prune_neurons, these neurons will be pruned'
        prun_neurons = sorted(range(new_layer_param[0].shape[-1]), key=lambda k: avg_neuron_w[k])[:int((prun_factor*new_layer_param[0].shape[-1])/100)]
        prun_neurons = np.sort(prun_neurons)

        'The number of the new units of the dense layer are stored'
        num_new_neurons = new_layer_param[0].shape[-1] - len(prun_neurons)
        '''

        'Deleting the neurons, beginning with the neuron with the highest index'
        if len(prun_neurons) > 0:
            for i in range(len(prun_neurons)-1,-1,-1):
                new_model_param, layer_output_shape = delete_dense_neuron(layer_params, layer_types, layer_output_shape, layer_bias, prun_layer, prun_neurons[i])

        else:
            new_model_param = layer_params
            print("No neurons to prune increase prune factor for dense layers")
        
    else:
        new_model_param = layer_params
        num_new_neurons = layer_params[prun_layer][0].shape[-1]
        print("No pruning implemented for dense layers")
    
    return new_model_param, num_new_neurons, layer_output_shape

def get_filter_to_prune_avarage(layer_params,prun_layer,prun_factor):
    'Load the filters of the conv layer and add a array where the' 
    'absolut average filter values will be stored'
    filters = layer_params[prun_layer]
    avg_filter_w = []
    'Absolute average of the filter values are written into an array'
    for i in range (0,filters[0].shape[-1]):
        avg_filter_w.append(np.average(np.abs(filters[0][:,:,:,i])))

    'Absolute average of the filter values are sorted and a percantage of these which is given'
    'through the prune factor are stored in prune_filters, these filters will be pruned'
    prun_filter = sorted(range(filters[0].shape[-1]), key=lambda k: avg_filter_w[k])[:int((prun_factor*filters[0].shape[-1])/100)]
    prun_filter = np.sort(prun_filter)

    'The number of the new filters of the conv layer are stored'
    num_new_filter = filters[0].shape[-1] - len(prun_filter)
    return prun_filter,num_new_filter
    
def get_filter_to_prune_L2(layer_params,prun_layer,prun_factor):
    'Load the filters of the conv layer and add a array where the' 
    'absolut average filter values will be stored'
    filters = layer_params[prun_layer]
    avg_filter_w = []
    'Absolute average of the filter values are written into an array'
    for i in range (0,filters[0].shape[-1]):
        avg_filter_w.append(np.average(np.abs(filters[0][:,:,:,i])))

    'Absolute average of the filter values are sorted and a percantage of these which is given'
    'through the prune factor are stored in prune_filters, these filters will be pruned'
    prun_filter = sorted(range(filters[0].shape[-1]), key=lambda k: avg_filter_w[k])[:int((prun_factor*filters[0].shape[-1])/100)]
    prun_filter = np.sort(prun_filter)

    'The number of the new filters of the conv layer are stored'
    num_new_filter = filters[0].shape[-1] - len(prun_filter)
    return prun_filter,num_new_filter

    

def prun_filters_conv(layer_types, layer_params, layer_output_shape, layer_bias, prun_layer, prun_factor,metric='L1'):
    """
    Deletes filters from the conv layer. The prun_factor is telling how much percent of the 
    filters of the conv layer should be deleted.
    
    Args: 
        layer_types: If layer_types is Conv2D, filters will be removed
        layer_params: Stores the current weights of the model
        layer_output_shape: Stores the current output shapes of all layers of the model
        prun_layer: Integer of layer number
        prun_factor: Integer which says how many percent of the filters should be deleted
        
    Return: 
        new_model_param: New model params after deleting the filters
        num_new_filters: New number of filters of the conv layers
        layer_output_shape: New output shapes of the model
    """
    
    'Check if layer to prune is a Conv layer'
    if layer_types[prun_layer] != "Conv2D":
        print("No Conv layer!")
        return None, None
    #print(prun_factor)
    if prun_factor > 0:
        if metric == 'L1':
            prun_filter,num_new_filter=get_filter_to_prune_avarage(layer_params,prun_layer,prun_factor)
        elif metric == 'L2':
            prun_filter,num_new_filter=get_filter_to_prune_L2(layer_params,prun_layer,prun_factor)
        else:
            prun_filter,num_new_filter=get_filter_to_prune_avarage(layer_params,prun_layer,prun_factor)
        
        'Deleting the filters, beginning with the filter with the highest index'
        if len(prun_filter) > 0:
            for i in range(len(prun_filter)-1,-1,-1):
                new_model_param, layer_output_shape = delete_filter(layer_params, layer_types, layer_output_shape, layer_bias, prun_layer, prun_filter[i])

        else:
            new_model_param = layer_params
            print("No filter to prune increase prune factor for conv layers")
        
    else:
        new_model_param = layer_params
        num_new_filter = layer_params[prun_layer][0].shape[-1]
        print("No pruning implemented for conv layers")
    
    return new_model_param, num_new_filter, layer_output_shape



def model_pruning(layer_types, layer_params, layer_output_shape, layer_bias, num_new_neurons, num_new_filters, prun_factor_dense, prun_factor_conv,metric):
    """
    Deletes neurons and filters from all dense and conv layers. The two prunfactors are 
    telling how much percent of the neurons and the filters should be deleted.
    
    Args: 
        layer_types: The types of all layers of the model
        layer_params: Stores the current weights of the model
        layer_output_shape: Stores the current output shapes of all layers of the model
        num_new_neurons: Number of neurons of the dense layers
        num_new_filters: Number of filters of the conv layers
        prun_factor_dense: Integer which says how many percent of the neurons should be deleted
        prun_factor_conv: Integer which says how many percent of the filters should be deleted
        
    Return: 
        layer_params: New model params after deleting the neurons and filters
        num_new_neurons: New number of filters of the dense layers
        num_new_filters: New number of filters of the conv layers
        layer_output_shape: New output shapes of the model after deleting neurons and filters
    """
    
    for i in range(0,len(layer_params)-2):
        if layer_types[i] == "Dense":
            layer_params, num_new_neurons[i], layer_output_shape = prun_neurons_dense(layer_types, layer_params, layer_output_shape, layer_bias, i, prun_factor_dense,metric)

        elif layer_types[i] == "Conv2D":
            layer_params, num_new_filters[i], layer_output_shape = prun_filters_conv(layer_types, layer_params, layer_output_shape, layer_bias, i, prun_factor_conv,metric)

        else:
            ("No pruning for this layer")
            
    return layer_params, num_new_neurons, num_new_filters, layer_output_shape



def build_pruned_model(pruned_model, new_model_param, layer_types, num_new_neurons, num_new_filters,comp):
    """
    The new number of neurons and filters are changed in the model config.
    Load the new weight matrices into the model.
    
    Args: 
        pruned_model: Model which should be pruned
        new_model_param: Stores the new weights of the model
        layer_types: The types of all layers of the model
        num_new_neurons: Number of neurons of the dense layers
        num_new_filters: Number of filters of the conv layers
        
    Return: 
        pruned_model: New model after pruning all dense and conv layers
    """
    
    model_config = pruned_model.get_config()

    print(num_new_neurons)
    for i in range(0,len(model_config['layers'])-3):
        if model_config['layers'][i+1]['class_name'] == "Dense":
            print("Dense")
            model_config['layers'][i+1]['config']['units'] = num_new_neurons[i]

        elif model_config['layers'][i+1]['class_name'] == "Conv2D":
            print("Conv")
            model_config['layers'][i+1]['config']['filters'] = num_new_filters[i]

        else:
            print("No dense or conv")
            
    print("Before pruning:")        
    pruned_model.summary()
    
    pruned_model = Sequential.from_config(model_config)
    
    print("After pruning:")
    pruned_model.summary()
    
    pruned_model.compile(**comp)
    
    
    for i in range(0,len(pruned_model.layers)):
        if layer_types[i] == 'Conv2D' or layer_types[i] == 'Dense':
            pruned_model.layers[i].set_weights(new_model_param[i])
    
    return pruned_model



def pruning(keras_model, x_train, y_train,comp,fit, prun_factor_dense=10, prun_factor_conv=10,metric='L1'):
    """
    A given keras model get pruned. The factor for dense and conv says how many percent
    of the dense and conv layers should be deleted. After pruning the model will be
    retrained.
    
    Args: 
        keras_model: Model which should be pruned
        x_train: Training data to retrain the model after pruning
        y_train: Labels of training data to retrain the model after pruning
        prun_factor_dense: Integer which says how many percent of the neurons should be deleted
        prun_factor_conv: Integer which says how many percent of the filters should be deleted
        
    Return: 
        pruned_model: New model after pruning and retraining
    """
    
    if callable(getattr(keras_model, "predict", None)) :
        model = keras_model
    elif isinstance(keras_model, str) and ".h5" in keras_model:
        model = load_model(keras_model)
    else:
        print("No model given to prune")
    
    
    layer_types, layer_params, layer_output_shape, layer_bias = load_model_param(model)
    num_new_neurons = np.zeros(shape=len(layer_params), dtype=np.int16)
    num_new_filters = np.zeros(shape=len(layer_params), dtype=np.int16)

    layer_params, num_new_neurons, num_new_filters, layer_output_shape = model_pruning(layer_types, layer_params, layer_output_shape, layer_bias, num_new_neurons, num_new_filters, prun_factor_dense, prun_factor_conv,metric)

    print("Finish with pruning")

    pruned_model = build_pruned_model(model, layer_params, layer_types, num_new_neurons, num_new_filters,comp)

    #earlystopper = EarlyStopping(monitor='val_accuracy', min_delta= 1e-3, mode='min', verbose=1, patience=5, restore_best_weights=True)
    history = pruned_model.fit(x_train, y_train, **fit)
    
    return pruned_model


def pruning_for_acc(keras_model, x_train, y_train, x_test, y_test, comp,fit ,pruning_acc=None, max_acc_loss=1):
    """
    A given keras model gets pruned. Either an accuracy value (in %) can be specified, which 
    the minimized model must still achieve. Or the maximum loss of accuracy (in %) that 
    the minimized model may experience. The model is reduced step by step until the 
    accuracy value is underrun or the accuracy loss is exceeded.
    
    Args: 
        keras_model: Model which should be pruned
        x_train: Training data to retrain the model after pruning
        y_train: Labels of training data to retrain the model after pruning
        x_test: Test data for evaluation of the minimized model
        y_test: Labels of test data for evaluation of the minimized model
        pruning_acc: Integer which says which accuracy value (in %) should not be fall below. If pruning_acc is not defined, it is Baseline - 5%
        max_acc_loss: Integer which says which accuracy loss (in %) should not be exceed 
        
    Return: 
        pruned_model: New model after pruning and retraining
    """
    
    original_model = load_model(keras_model)
    original_model.compile(**comp)
    original_model_acc = original_model.evaluate(x_test,y_test)[-1]
    
    for i in range(5,100,5):
        model = pruning(original_model, x_train, y_train,comp,fit, prun_factor_dense=i, prun_factor_conv=i)
        
        if pruning_acc != None:
            if model.evaluate(x_test,y_test)[-1] < pruning_acc:
                print(i-5)
                if i == 5:
                    pruned_model = model
                return pruned_model
            pruned_model = model
            
        else:
            if model.evaluate(x_test,y_test)[-1] < (original_model_acc-(max_acc_loss/100)):
                print(i-5)
                return pruned_model
            pruned_model = model
    
    return pruned_model
    
    
def prune_model(keras_model, prun_factor_dense=10, prun_factor_conv=10,metric='L1',comp=None):
    """
    A given keras model get pruned. The factor for dense and conv says how many percent
    of the dense and conv layers should be deleted. After pruning the model will be
    retrained.
    
    Args: 
        keras_model: Model which should be pruned
        prun_factor_dense: Integer which says how many percent of the neurons should be deleted
        prun_factor_conv: Integer which says how many percent of the filters should be deleted
        
    Return: 
        pruned_model: New model after pruning 
    """
    
    if callable(getattr(keras_model, "predict", None)) :
        model = keras_model
    elif isinstance(keras_model, str) and ".h5" in keras_model:
        model = load_model(keras_model)
    else:
        print("No model given to prune")

    if comp is None:
        comp = {
        "optimizer": 'adam',
        "loss": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "metrics": 'accuracy'}    
    
    
    layer_types, layer_params, layer_output_shape, layer_bias = load_model_param(model)
    num_new_neurons = np.zeros(shape=len(layer_params), dtype=np.int16)
    num_new_filters = np.zeros(shape=len(layer_params), dtype=np.int16)

    layer_params, num_new_neurons, num_new_filters, layer_output_shape = model_pruning(layer_types, layer_params, layer_output_shape, layer_bias, num_new_neurons, num_new_filters, prun_factor_dense, prun_factor_conv,metric)

    print("Finish with pruning")

    pruned_model = build_pruned_model(model, layer_params, layer_types, num_new_neurons, num_new_filters,comp)
    
    return pruned_model
