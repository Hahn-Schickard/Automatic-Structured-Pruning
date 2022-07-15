import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import pruning
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


print("Load CIFAR10 Dataset as test dataset")
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0


print("\nBuild model")
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(10, activation='softmax'))
model.summary()


print("\nCompile, train and evaluate model")
comp = {
"optimizer":'adam',
"loss": tf.keras.losses.SparseCategoricalCrossentropy(),
"metrics": ['accuracy']}

model.compile(**comp)
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]

model.fit(train_images, train_labels, validation_split=0.2, epochs=10, batch_size=128, callbacks=callbacks)

model_test_loss, model_test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Model accuracy after Training: {model_test_acc*100:.2f}%")


print("\nTest factor pruning")
dense_prune_rate=30
conv_prune_rate=40
pruned_model=pruning.factor_pruning(model, dense_prune_rate, conv_prune_rate,'L2')


print("\nCompile, retrain and evaluate pruned model")
comp = {
"optimizer":'adam',
"loss": tf.keras.losses.SparseCategoricalCrossentropy(),
"metrics": ['accuracy']}

pruned_model.compile(**comp)

pruned_model.fit(train_images, train_labels, epochs=10, validation_split=0.2)


print("\nCompare model before and after pruning")
model_test_loss, model_test_acc = model.evaluate(test_images,  test_labels, verbose=2)
pruned_model_test_loss, pruned_model_test_acc = pruned_model.evaluate(test_images,  test_labels, verbose=2)
print(f"Model accuracy before pruning: {model_test_acc*100:.2f}%")
print(f"Model accuracy after pruning: {pruned_model_test_acc*100:.2f}%")

print(f"\nTotal number of parameters before pruning: {model.count_params()}")
print(f"Total number of parameters after pruning: {pruned_model.count_params()}")
print(f"Pruned model contains only {(pruned_model.count_params()/model.count_params())*100:.2f} % of the original number of parameters.")


print("\nTest accuracy pruning")
comp = {
  "optimizer": 'adam',
  "loss": tf.keras.losses.SparseCategoricalCrossentropy(),
  "metrics": 'accuracy'
}

auto_model = pruning.accuracy_pruning(model, comp, train_images, train_labels, test_images,
                                     test_labels, pruning_acc=None, max_acc_loss=3,
                                     label_one_hot=False)


print("\nCompare model before and after pruning")
model_test_loss, model_test_acc = model.evaluate(test_images,  test_labels, verbose=2)
auto_model_test_loss, auto_model_test_acc = auto_model.evaluate(test_images,  test_labels, verbose=2)
print(f"Model accuracy before pruning: {model_test_acc*100:.2f}%")
print(f"Model accuracy after pruning: {auto_model_test_acc*100:.2f}%")

print(f"\nTotal number of parameters before pruning: {model.count_params()}")
print(f"Total number of parameters after pruning: {auto_model.count_params()}")
print(f"Pruned model contains only {(auto_model.count_params()/model.count_params())*100:.2f} % of the original number of parameters.")