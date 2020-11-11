# Automatic-Structured-Pruning
We have implemented a framework that supports developers to structured prune neural networks.
The framework is still under development and will be extended from time to time.


# Features:
- Prune of filters / channels
- Prune of neurons
- Automatic pruning to minimum accuracy
- Prune metric: L1/L2 Norm



# Restrictions:
- There are only SequentialModels Pruneable
- Tensorflow 2.1 to 2.3.1 is supported
- it only supports magnitude-based pruning


# Upcoming functions:
- More prune methods not only l1/l2 norm
- not only Sequential models to prune
- Resnet support
- Train from scratch
- do you have an idea? please write to Marcus.rueb@hahn-schickard.de or create an issue.



# How to start?
To make it easier to get started with the framework we have created a notebook which helps you to use the functions directly.
You can open the notebook in Google Colab.
