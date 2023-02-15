# The API

## API Details

The API of the module will follow that of Keras' Sequential model. So, I'll have somethng like this:

```python
from nn import Sequential

model = Sequential()
model.add(Dense(32, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(10))

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)
```

`model.summary()` will also be available.

Or something similar to this. Details will come as I implement the module.

## Implementation Details

An abstract class `Layer` will be defined. It will have the following methods:

- `__init__`: The constructor of the layer. It will take the parameters of the layer as arguments.
- `forward`: The forward pass of the layer. It will take the input as an argument and return the output.
- `backward`: The backward pass of the layer. It will take the input and the gradient of the loss with respect to the output as arguments and return the gradient of the loss with respect to the input.
- `update`: The update function of the layer. It will take the learning rate as an argument and update the parameters of the layer.
- `get_params`: The function to get the parameters of the layer. It will return a list of the parameters of the layer.
