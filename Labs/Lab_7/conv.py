import numpy as np

class Conv3x3:
  # A Convolution layer using 3x3 filters.

  def __init__(self, num_filters):
    self.num_filters = num_filters

    # filters is a 3d array with dimensions (num_filters, 3, 3)
    # Divide the filter by 9 to reduce the variance of initial values
    self.filters = 

  def iterate_regions(self, image):
    '''
    Generates all possible 3x3 image regions without using padding.
    - image is a 2d numpy array (hint: You can use 'yield' statement in Python)
    '''
    h, w = image.shape

    



  def forward(self, input):
    '''
    Performs a forward pass of the conv layer using the given input.
    Returns a 3d numpy array with dimensions (h, w, num_filters).
    - input is a 2d numpy array
    '''
    h, w = input.shape
    output = np.zeros(( , , self.num_filters))

    



    return output