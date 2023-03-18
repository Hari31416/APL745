import numpy as np

class Softmax:
  # A standard fully-connected layer with softmax activation.

  def __init__(self, input_len, output_len):
    # After randomly intializing the weights, divide by input_len to reduce the variance
    self.weights = 
    self.biases = 

  def forward(self, input):
    '''
    Performs a forward pass of the softmax layer using the given input.
    Returns a 1d numpy array containing the respective probability values.
    - input can be any array with any dimensions.
    '''
    self.last_input_shape = input.shape
    self.last_input = input

    
    
    
    
    
    
    return prob
    
  def backprop(self, d_L_d_out, learn_rate):
    '''
    Performs a backward pass of the softmax layer.
    Returns the loss gradient for this layer's inputs.
    - d_L_d_out is the loss gradient for this layer's outputs.
    - learn_rate is a float.
    - Only 1 element of d_L_d_out will be nonzero.
    '''

    for i, gradient in enumerate(d_L_d_out):
      if gradient == 0:
        continue  # Skips the current iteration of the loop

      # e^totals (Evaluate exponents for the values passed into softmax activation function)
      t_exp = 

      # Sum of all e^totals
      S = 

      # Gradients of out[i] against totals (Compute the gradients of the outputs of softmax layer wrt totals)
      d_out_d_t = 
      d_out_d_t[i] = 

      # Gradients of totals against weights/biases/input
      d_t_d_w = 
      d_t_d_b = 
      d_t_d_inputs = 

      # Gradients of loss against totals
      d_L_d_t = 

      # Gradients of loss against weights/biases/input
      d_L_d_w = 
      d_L_d_b = 
      d_L_d_inputs = 

      # Update weights / biases
      self.weights -= learn_rate * d_L_d_w
      self.biases -= learn_rate * d_L_d_b

      return d_L_d_inputs.reshape(self.last_input_shape)
