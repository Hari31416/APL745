import torch
import numpy as np
import matplotlib.pyplot as plt

class InversePhysicsInformedBarModel:
    """
    A class used for the definition of the data driven approach for Physics Informed Models for one dimensional bars. 
    EA is estimated.
    """

    def __init__(self, x, u, L, dist_load):
        """Construct a InversePhysicsInformedBarModel model"""

        '''
         Enter your code
         Task : initialize required variables for the class
        '''

    def predict(self, x, u):
        """Predict parameter EA of the differential equation."""

        '''
        Params: 
            x - input spatial value
            u - input displacement value at x
            ea - model predicted value
        '''

        '''
        Enter your code
        '''

        return ea

    def cost_function(self, x, u, EA_pred):
        """Compute the cost function."""

        '''
        Params:
            x - input spatial value
            u - displacement value at x
            EA_pred - model predicted EA value
            differential_equation_loss - calculated physics loss
        '''

        '''
        Enter your code
        '''

        return differential_equation_loss
    
    def train(self, epochs, optimizer, **kwargs):
        """Train the model."""

        '''
        This function is used for training the network. While updating the model params use "cost_function" 
        function for calculating loss
        Params:
            epochs - number of epochs
            optimizer - name of the optimizer
            **kwarhs - additional params

        This function doesn't have any return values. Just print the losses during training
        
        '''

        '''
            Enter your code        
        '''
