import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax
import scipy.io
from matplotlib import pyplot as plt

# %% Load data




# %% Model definition
conv = Conv3x3()                   # 28x28x1 -> 26x26x8
pool = MaxPool2()                  # 26x26x8 -> 13x13x8
softmax = Softmax()                # 13x13x8 -> 10

def forward(image, label):
  '''
  Completes a forward pass of the CNN and calculates the accuracy and
  cross-entropy loss.
  - image is a 2d numpy array
  - label is a digit
  '''
  # Transform the grayscale image from [0, 255] to [-0.5, 0.5] to make it easier
  # to work with.
  out = conv.forward()
  out = pool.forward()
  out = softmax.forward()

  # Compute cross-entropy loss and accuracy.
  loss = 
  acc = 

  return out, loss, acc
  
def train(im, label, lr=.005):
  '''
  A training step on the given image and label.
  Shall return the cross-entropy loss and accuracy.
  - image is a 2d numpy array
  - label is a digit
  - lr is the learning rate
  '''
  # Forward
  out, loss, acc = forward()

  # Calculate initial gradient
  gradient = 
  gradient[label] = -1 / out[label]

  # Backprop
  gradient = softmax.backprop(gradient, lr)
  gradient = pool.backprop(gradient)
  gradient = conv.backprop(gradient, lr)

  return loss, acc

# %% Training the model for 3 epochs
for epoch in range(3):
  print('--- Epoch %d ---' % (epoch + 1))

  # Shuffle the training data
  permutation = np.random.permutation(len(train_images))
  train_images = train_images[permutation]
  train_labels = train_labels[permutation]

  # Train!
  loss = 0
  num_correct = 0

  for i, (im, label) in enumerate(zip()):
    if i % 100 == 0:
    print(
      '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
      (i, loss / 100, num_correct))
    loss = 0
    num_correct = 0

    op, l, acc = train(im, label)
    loss = 
    num_correct = 
    
# Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip():
  _, l, acc = forward(im, label)
  loss
  num_correct

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)

# %% Plotting
''' Plot some of the images with predicted and actual labels'''



