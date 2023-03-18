import numpy as np

class MaxPool2:
  # A Max Pooling layer using a pool size of 2.

  def iterate_regions(self, image):
    '''
    Generates non-overlapping 2x2 image regions to pool over.
    - image is a 2d numpy array
    hint: You can use 'yield' statement in Python to create iterate regions
    '''
    h, w, _ = image.shape
    
    
    
    
    
    

  def forward(self, input):
    '''
    Performs a forward pass of the maxpool layer using the given input.
    Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
    - input is a 3d numpy array with dimensions (h, w, num_filters)
    '''
    h, w, num_filters = input.shape
    self.last_input = input
    output = np.zeros(( , , num_filters))

    




    return output
    
  def backprop(self, d_L_d_out):
    '''
    Performs a backward pass of the maxpool layer.
    Returns the loss gradient for this layer's inputs.
    - d_L_d_out is the loss gradient for this layer's outputs.
    '''
    d_L_d_input = np.zeros(self.last_input.shape)

    for im_region, i, j in self.iterate_regions(self.last_input):
      h, w, f = 
      amax = 

    # Iterate over every pixel of 'im_region' and check whether pixel value
    # matches one of the elements of amax. If true, copy the gradient to that location.

                
            

    return d_L_d_input
