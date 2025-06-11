# Do not use packages that are not in standard distribution of python
import numpy as np
np.random.seed(1024)
from ._base_network import _baseNetwork

class TwoLayerNet(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10, hidden_size=128):
        super().__init__(input_size, num_classes)

        self.hidden_size = hidden_size
        self._weight_init()


    def _weight_init(self):
        '''
        initialize weights of the network
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the first layer of shape (num_features, hidden_size)
        - b1: The bias term of the first layer of shape (hidden_size,)
        - W2: The weight matrix of the second layer of shape (hidden_size, num_classes)
        - b2: The bias term of the second layer of shape (num_classes,)
        '''

        # initialize weights
        self.weights['b1'] = np.zeros(self.hidden_size)
        self.weights['b2'] = np.zeros(self.num_classes)
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.hidden_size)
        np.random.seed(1024)
        self.weights['W2'] = 0.001 * np.random.randn(self.hidden_size, self.num_classes)

        # initialize gradients to zeros
        self.gradients['W1'] = np.zeros((self.input_size, self.hidden_size))
        self.gradients['b1'] = np.zeros(self.hidden_size)
        self.gradients['W2'] = np.zeros((self.hidden_size, self.num_classes))
        self.gradients['b2'] = np.zeros(self.num_classes)

    def forward(self, X, y, mode='train'):
        '''
        The forward pass of the two-layer net. The activation function used in between the two layers is sigmoid, which
        is to be implemented in self.,sigmoid.
        The method forward should compute the loss of input batch X and gradients of each weights.
        Further, it should also compute the accuracy of given batch. The loss and
        accuracy are returned by the method and gradients are stored in self.gradients

        :param X: a batch of images (N, input_size)
        :param y: labels of images in the batch (N,)
        :param mode: if mode is training, compute and update gradients;else, just return the loss and accuracy
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
            self.gradients: gradients are not explicitly returned but rather updated in the class member self.gradients
        '''
        loss = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process:                                      #
        #        1) Call sigmoid function between the two layers for non-linearity  #
        #        2) The output of the second layer should be passed to softmax      #
        #        function before computing the cross entropy loss                   #
        #    2) Compute Cross-Entropy Loss and batch accuracy based on network      #
        #       outputs                                                             #
        #############################################################################
        X = np.array(X)  # Ensure X is a numpy array
        N = X.shape[0]
        X_flat = X.reshape(N,-1)

        L1 = X_flat @ self.weights['W1'] + self.weights['b1']  # Linear transformation for first layer
        L1_sigmoid = self.sigmoid(L1)  # Apply sigmoid activation function

        L2 = L1_sigmoid @ self.weights['W2'] + self.weights['b2']  # Linear transformation for second layer
        
        prob = self.softmax(L2)  # Apply softmax to get probabilities

        loss = self.cross_entropy_loss(prob, y)  # Compute Cross-Entropy Loss
        accuracy = self.compute_accuracy(prob, y)  # Compute accuracy
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
        #    HINT: You will need to compute gradients backwards, i.e, compute       #
        #          gradients of W2 and b2 first, then compute it for W1 and b1      #
        #          You may also want to implement the analytical derivative of      #
        #          the sigmoid function in self.sigmoid_dev first                   #
        #############################################################################
        # compute the gradiants of the loss with respect to cross-entropy and sofmax
        dL_dL2 = prob.copy()          # (N, num_classes)
        dL_dL2[np.arange(N), y] -= 1  # Subtract 1 at true class indices
        dL_dL2 /= N                   # Average over batch

        # compute the gradiants of the loss with respect to L2
        dL_dW2 = L1_sigmoid.T @ dL_dL2
        dL_db2 = np.sum(dL_dL2, axis=0)

        self.gradients['W2'] = dL_dW2  # store the gradients wrt W_2
        self.gradients['b2'] = dL_db2  # store the bias wrt b_2

        # Gradient w.r.t. L1_sigmoid
        dL_dL1_sigmoid = dL_dL2 @ self.weights['W2'].T  # (N, hidden_size)

        # Gradient w.r.t. L1 pre-sigmoid
        dL_dL1 = dL_dL1_sigmoid * self.sigmoid_dev(L1)  # Apply derivative of sigmoid

        dL_dW1 = X_flat.T @ dL_dL1
        dL_db1 = np.sum(dL_dL1, axis=0)

        self.gradients['W1'] = dL_dW1  # store the gradients wrt W_1
        self.gradients['b1'] = dL_db1  # store the bias wrt b_1
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


        return loss, accuracy


