# Automatic-Structured-Pruning
We have implemented a framework to support developers for an automatic structured pruning of their neural networks.
In previous pruning frameworks, only weight pruning is applied here. Although the unnecessary weights are set to zero, memory must still be provided for these weights, which does not result in a reduction of the weights. Thus, no reduction of the required memory space is achieved by this either.
For this reason, the Automatic-Structured-Pruning tool was developed. This tool makes it possible to perform pruning for convolutional and fully cross-linked layers. Individual filters or neurons are deleted directly from the respective layers. This results in a reduction of the weights as well as the memory requirements.
The tool allows two different pruning approaches:
- Factor: For the fully connected and convolutional layers, a factor is specified in each case, which indicates the percentage of neurons or filters to be deleted from the layer.
- Accuracy: The minimum accuracy or the maximum loss of accuracy is specified. This defines which accuracy the neural network should still reach after pruning.

The framework is still under development and will be extended from time to time.

Additionally, the Automatic-Structured-Pruning is part of the Tool [AutoFlow](https://github.com/Hahn-Schickard/AutoFlow), which is a tool that helps developers to implement machine learning (ML) faster and easier on embedded devices. The whole workflow of a data scientist should be covered. Starting from building the ML model to the selection of the target platform to the optimization and implementation of the model on the target platform.

# Features:
- Prune of filters / channels
- Prune of neurons
- Automatic pruning to minimum accuracy
- Prune metric: L1/L2 Norm for neuron Pruning


# Restrictions:
- The tool is tested with Tensorflow 2.5
- It only supports magnitude-based pruning


# Upcoming functions:
- More prune methods not only l1/l2 norm
- not only Sequential models to prune
- Resnet support
- Train from scratch
- Do you have an new idea? Please write to Marcus.Rueb@Hahn-Schickard.de, Daniel.Konegen@Hahn-Schickard.de or create an issue.



# How to start?
To make it easier to get started with the framework we have created a [notebook](https://github.com/Hahn-Schickard/Automatic-Structured-Pruning/examples/How_to_use_the_Framework.ipynb) which helps you to use the functions directly.
You can open the notebook in [Google Colab](https://colab.research.google.com/github/Hahn-Schickard/Automatic-Structured-Pruning/blob/master/examples/How_to_use_the_Framework.ipynb).
