import numpy as np

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        '''
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        '''
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        '''
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        
        #set up the output shape
        N, C, H, W = x.shape
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = np.zeros((N, self.out_channels, H_out, W_out))

        #pad the input
        if self.padding > 0:
            pad_width = [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)]
            x_padded = np.pad(x, pad_width, mode='constant', constant_values=0)
        else:
            x_padded = x
        
        #perform convolution
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        out[:, i, h, w] += np.sum(x_padded[:, j, h_start:h_end, w_start:w_end] * self.weight[i, j], axis=(1, 2))
            out[:, i] += self.bias[i]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        '''
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        '''
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        N, C, H, W = x.shape

        if self.padding > 0:
            pad_width = [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)]
            x_padded = np.pad(x, pad_width, mode='constant', constant_values=0)
        else:
            x_padded = x
        
        db = np.zeros(self.out_channels)
        dw = np.zeros_like(self.weight)
        dx_padded = np.zeros_like(x_padded)
        
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        for i in range(self.out_channels):
            for j in range(self.in_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # accumulate gradients for weights
                        #For a weight self.weight[i, j, p, q], it contributes to out[n, i, h, w] via x_padded[n, j, h*stride + p, w*stride + q].
                        #Thus, dout[n, i, h, w]/∂weight[i, j, p, q] = x_padded[n, j, h*stride + p, w*stride + q].
                        dw[i, j] += np.sum(x_padded[:, j, h_start:h_end, w_start:w_end] * dout[:, i, h, w][:, None, None], axis=0)
                        
                        # accumulate gradients for input
                        dx_padded[:, j, h_start:h_end, w_start:w_end] += dout[:, i, h, w][:, None, None] * self.weight[i, j]
        
        db = np.sum(dout,axis =(0,2,3))  # sum over N, H_out, W_out

        if self.padding > 0:
            # remove padding from dx_padded
            self.dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            self.dx = dx_padded
        
        self.dw = dw
        self.db = db
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################