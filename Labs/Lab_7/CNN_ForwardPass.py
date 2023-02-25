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

# %% Training the model
loss = 0
num_correct = 0
pred = np.zeros(test_labels.shape)

for i, (im, label) in enumerate(zip()):
  # Do a forward pass.
  op, l, acc = forward()
  pred[i] = 
  loss = 
  num_correct = 

  # Print stats every 100 steps.
  if i % 100 == 0:
    print(
      '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
      (i, loss / 100, num_correct))
    loss = 0
    num_correct = 0
    
# %% Plotting
''' Plot some of the images with predicted and actual labels'''



