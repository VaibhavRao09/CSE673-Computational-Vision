import numpy as np

class baselayer:
    def __init__(self):
        pass
    def forward():
        pass
    def backword():
        pass


class Conv(baselayer):
    def __init__(self, in_channels,out_channels,kernel_size,stride=1,padding=1):
        self.pad = padding
        self.stride = stride
        self.dw, self.db = None, None
        self.x_orig = None
        self.H_f, self.W_f = kernel_size
        self.in_c = in_channels
        self.out_c = out_channels
        self.w = np.random.randn(self.H_f, self.W_f, in_channels, out_channels) * 0.1
        self.b = np.random.randn(out_channels) * 0.1
        
        
    def forward(self, X):
        (n, H_in, W_in, *_) = X.shape
        X_pad = np.pad(X, self.pad, mode='constant')
        #Add a new axis here so that we can vectorize some of 
        #our Calculations Below
        if self.in_c == 1:
            X_pad = X_pad[:,:,:,np.newaxis]
        self.x_orig = X_pad.shape
        h_out = ((H_in - self.H_f + 2*(self.pad)) // self.stride) + 1
        w_out = ((W_in - self.W_f + 2*(self.pad)) // self.stride) + 1
        n_f = self.out_c
        output_shape = (n, h_out, w_out, n_f)
        out_image = np.zeros(output_shape)
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.H_f
                w_start = j * self.stride
                w_end = w_start + self.W_f
                
                out_image[:, i, j, :] = np.sum(X_pad[:, h_start:h_end, w_start:w_end, :, np.newaxis] *
                    self.w[np.newaxis, :, :, :],
                    axis=(1,2,3)
                )
        return out_image + self.b
    
    def backward(self, x, grad):
        n, H_in, W_in, _ = self.x_orig
        X_pad = np.pad(x, self.pad, mode='constant')

        if self.in_c == 1:
            X_pad = X_pad[:,:,:,np.newaxis]
        output = np.zeros_like(X_pad)
        _, h_out, w_out, _ = grad.shape
        self.db = grad.sum(axis=(0, 1, 2)) / n
        self.dw = np.zeros_like(self.w)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.H_f
                w_start = j * self.stride
                w_end = w_start + self.W_f
                output[:, h_start:h_end, w_start:w_end, :] += np.sum(
                    self.w[np.newaxis, :, :, :, :]*
                    grad[:, i:i+1, j:j+1, np.newaxis, :],
                    axis=4
                )
                self.dw += np.sum(
                    X_pad[:, h_start:h_end, w_start:w_end, :,np.newaxis]*
                    grad[:, i:i+1, j:j+1, np.newaxis, :],
                    axis=0
                )

        self.dw /= n
        self.w = self.dw
        self.b = self.db
        assert output.shape == self.x_orig
        return output


class AvgPool():
    def __init__(self, in_channels,kernel_size,stride=1,padding=1):
        self.pad = padding
        self.stride = stride
        self._dw, self._db = None, None
        self.x_orig = None
        self.H_f, self.W_f = kernel_size
        self.in_c = in_channels
    
    def forward(self, X):
      self.x_orig = X.shape
      n, H_in, W_in, c = X.shape
      h_out = ((H_in - self.H_f + 2*(self.pad)) // self.stride) + 1
      w_out = ((W_in - self.W_f + 2*(self.pad)) // self.stride) + 1
      output = np.zeros((n, h_out, w_out, c))
      self.cache = np.zeros_like(X)
      self.cache.reshape(n, H_in * W_in, c)

      for i in range(h_out):
          for j in range(w_out):
              h_start = i * self.stride
              h_end = h_start + self.H_f
              w_start = j * self.stride
              w_end = w_start + self.W_f
              X_slice = X[:, h_start:h_end, w_start:w_end, :]
              output[:, i, j, :] = np.mean(X_slice, axis=(1, 2))
      return output

    def backward(self, x, grad):
        output = np.zeros_like(x)
        _, h_out, w_out, _ = grad.shape

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.H_f
                w_start = j * self.stride
                w_end = w_start + self.W_f
                output[:, h_start:h_end, w_start:w_end, :] += grad[:, i:i + 1, j:j + 1, :]
        return output


class MaxPool():
    
    def __init__(self, in_channels,kernel_size,stride=1,padding=1):
        self.pad = padding
        self.stride = stride
        self._dw, self._db = None, None
        self.x_orig = None
        self.H_f, self.W_f = kernel_size
        self.in_c = in_channels
    
    def forward(self, X):
      self.x_orig = X.shape
      n, H_in, W_in, c = X.shape
      h_out = ((H_in - self.H_f + 2*(self.pad)) // self.stride) + 1
      w_out = ((W_in - self.W_f + 2*(self.pad)) // self.stride) + 1
      output = np.zeros((n, h_out, w_out, c))
      for i in range(h_out):
          for j in range(w_out):
              h_start = i * self.stride
              h_end = h_start + self.H_f
              w_start = j * self.stride
              w_end = w_start + self.W_f
              X_slice = X[:, h_start:h_end, w_start:w_end, :]
              idx = np.argmax(X_slice, axis=1)
              max_ele = np.max(X_slice, axis=(1, 2))
              output[:, i, j, :] = max_ele
      return output
  
    def backward(self, x, grad):
        output = np.zeros_like(x)
        _, h_out, w_out, _ = grad.shape

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.H_f
                w_start = j * self.stride
                w_end = w_start + self.W_f
                output[:, h_start:h_end, w_start:w_end, :] += grad[:, i:i + 1, j:j + 1, :]
        return output


class Dense(baselayer):
    
    def __init__(self,input_units,out_units):
        self.w = np.random.normal(loc=0.0, scale=np.sqrt(2/(input_units+out_units)), size=(input_units,out_units))
        self.b = np.zeros(out_units)
        self.learning_rate = 0.1
        
    def forward(self, x):
        layeroutput = np.dot(x,self.w) + self.b       
        return layeroutput
        
    def backward(self, x, grad_output):
        grad_input = np.dot(grad_output, self.w.T)
        grad_weights = np.dot(x.T, grad_output)
        grad_biases = grad_output.mean(axis=0)*x.shape[0]
        self.w = self.w - self.learning_rate * grad_weights
        self.b = self.b - self.learning_rate * grad_biases
        return grad_input


class Flatten(baselayer):
    def __init__(self):
        self.shape = None
    def forward(self, inp):
        self.shape = inp.shape
        return np.ravel(inp).reshape(inp.shape[0], -1)
    def backward(self, x, grad):
        grad = grad.reshape(self.shape)
        return grad

class Dropout(baselayer):
    def __init__(self, drop_ratio=.2):
        self.points = None
        self.drop_ratio = drop_ratio
    def forward(self, x):
        self.coef = 1 / (1 - self.drop_ratio)
        self.points = (np.random.rand(*x.shape) > self.drop_ratio) * self.coef
        return x * self.points
    def backward(self, x, grad):
        return grad * self.points

class Sigmoid(baselayer):
    def __init__(self):
        self.sig = None
        pass
    def forward(self, x):
        self.sig = 1/(1 + np.exp(-x))
        return self.sig
    def backward(self, x, grad):
        return np.multiply(np.multiply(self.sig, (1 - self.sig)), grad)
    
class Softmax(baselayer):
    def __init__(self):
        self.soft = None
        pass
    def forward(self, x):
        self.soft = np.exp(x)/np.sum(np.exp(x),axis=-1,keepdims=True)
        return self.soft
    def backward(self, x, grad):
        g = (self.soft - grad) / grad.shape[0]
        return g
    
class ReLU(baselayer):
    def __init__(self):
        pass    
    def forward(self, x):
        x = np.maximum(x, 0)
        return x
    def backward(self, x, grad):
        g = np.array(grad, copy = True)
        g[np.where(x<=0)] = 0;
        return g;
    