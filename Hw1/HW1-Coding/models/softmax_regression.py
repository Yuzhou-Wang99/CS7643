# Do not use packages that are not in standard distribution of python
import numpy as np

from ._base_network import _baseNetwork

class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28*28, num_classes=10):
        '''
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => ReLU activation => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        '''
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        '''
        initialize weights of the single layer regression network. No bias term included.
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the linear layer of shape (num_features, hidden_size)
        '''
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):
        '''
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of image (N, 28x28)
        :param y: labels of images in the batch (N,)
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        '''
        loss = None
        gradient = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process and compute the Cross-Entropy loss    #
        # Hint:                                                                     #
        #   Store your intermediate outputs before ReLU for backwards               #
        #############################################################################
        X = np.array(X)
        N = X.shape[0]
        X_flat = X.reshape(N, -1)  # Flatten the input images to (N, input_size)   
        
        # Compute scores using the linear layer
        scores = X_flat @ self.weights['W1']

        # Apply ReLU activation
        scores_ReLu = np.maximum(0, scores)
        
        # Compute softmax probabilities
        prob = self.softmax(scores_ReLu)
        
        loss = self.cross_entropy_loss(prob, y) 
        accuracy = self.compute_accuracy(prob, y)  
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        if mode != 'train':
            return loss, accuracy

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight and bias by chain rule         #
        #        2) Store the gradients in self.gradients                           #
        #############################################################################
        
        # compute the gradiants of the loss with respect to cross-entropy and sofmax
        dL_da = prob.copy()          # (N, num_classes)
        dL_da[np.arange(N), y] -= 1  # Subtract 1 at true class indices
        dL_da /= N                   # Average over batch

        # compute the gradiants of the loss with respect to scores_ReLu
        da_dz = self.ReLU_dev(scores)  # (N, num_classes)
        dL_dz = dL_da * da_dz  # Element-wise multiplication

        # compute the gradiants of the loss with respect to scores
        dz_dW1 = X_flat  # (N, input_size)
        dL_dW1 = dz_dW1.T @ dL_dz  # (input_size, num_classes) 

        self.gradients['W1'] = dL_dW1 # store the gradients 

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss, accuracy





        


