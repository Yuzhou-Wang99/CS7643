import numpy as np

class MaxPooling:
    '''
    Max Pooling of input
    '''
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        '''
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        '''
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        N,C,H,W = x.shape

        H_out = (H - self.kernel_size)//self.stride + 1
        W_out = (W - self.kernel_size)//self.stride + 1

        out = np.zeros((N, C, H_out, W_out))

        for i in range(N):
            for j in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        window = x[i, j, h_start:h_end, w_start:w_end]
                        
                        out[i, j, h, w] = np.max(window)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        '''
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        '''
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        N, C, H, W = x.shape
        dx = np.zeros_like(x)

        for i in range(N):
            for j in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size

                        window = x[i, j, h_start:h_end, w_start:w_end]
                        max_index = np.unravel_index(np.argmax(window), window.shape)

                        dx[i, j, h_start + max_index[0], w_start + max_index[1]] += dout[i, j, h, w]
        
        self.dx = dx
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
