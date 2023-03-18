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
    
    h1, w1 = ??
    
    for i in range(h1):
      for j in range(w1):
        im_region = image[i:(i + 3), j:(j + 3)]
        yield im_region, i, j
    



  def forward(self, input):
    '''
    Performs a forward pass of the conv layer using the given input.
    Returns a 3d numpy array with dimensions (h, w, num_filters).
    - input is a 2d numpy array
    '''
    self.last_input = input
    
    h, w = input.shape
    output = np.zeros(( , , self.num_filters))

    



    return output
    
  def backprop(self, d_L_d_out, learn_rate):
    '''
    Performs a backward pass of the conv layer.
    - d_L_d_out is the loss gradient for this layer's outputs.
    - learn_rate is a float.
    '''
    d_L_d_filters = np.zeros(self.filters.shape)

    for im_region, i, j in self.iterate_regions(self.last_input):
      for f in range(self.num_filters):

          

    # Update filters
    self.filters -= learn_rate * d_L_d_filters

    return None