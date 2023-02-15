# Details

Here, details about the implementation of the module will be given. The module will closely follow the API of Keras' Sequential model.

## Classes

A number of classes are required. There should be:

1. A class for the model.
2. A class for each layer.

### Model

This class is responsible for compiling, training and predicting. The class should have the following public methods:

- `compile`: This method will take the loss function and the metrics as arguments. It will compile the model.
- `fit`: This method will take the training data, the number of epochs and the batch size as arguments. It will train the model.
- `predict`: This method will take the test data as an argument and return the predictions.
- `summary`: This method will print a summary of the model.
- `score`: This method will take the test data and the test labels as arguments and return the score of the model.

The class is to implement such that it can be used as follows:

```python
from nn import Sequential

model = Sequential()
model.add(Dense(32, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(10))

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)

predictions = model.predict(X_test)
```

### Layer

The `Layer` class is the base class for all the layers. It has the following methods:

- `__init__`: The constructor of the layer. It will take the parameters of the layer as arguments.
- `forward`: The forward pass of the layer. It will take the input as an argument and return the output.
