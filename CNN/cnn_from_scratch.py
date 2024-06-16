
import math
import random
import time

def normal(loc=0.0, scale=1.0, size=None):
    if size is None:
        size = ()
    elif isinstance(size, int):
        size = (size,)
    else:
        size = tuple(size)

    # Generate random numbers from a standard normal distribution
    samples = []
    for _ in range(math.prod(size) if size else 1):
        u1 = random.random()  # Uniform random variable 1
        u2 = random.random()  # Uniform random variable 2
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)  # Standard normal random variable
        samples.append(z)

    # Scale and shift the samples to match the desired mean and standard deviation
    samples = [s * scale + loc for s in samples]

    # Reshape the list into the specified shape
    if size:
        if len(size) == 1:
            return samples[0] if samples else None
        elif len(size) == 2:
            return [samples[i * size[1]:(i + 1) * size[1]] for i in range(size[0])]
        else:
            raise ValueError("Shape dimensions greater than 2 are not supported.")
    else:
        return samples[0] if samples else None

def randn(*args):
    shape = args if len(args) > 0 else ()
    size = 1
    for dim in shape:
        size *= dim

    # Generate random values from standard normal distribution
    samples = []
    for _ in range(size):
        u1 = random.random()  # Uniform random variable 1
        u2 = random.random()  # Uniform random variable 2
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)  # Standard normal random variable
        samples.append(z)

    # Reshape the list into the specified shape
    if shape:
        return reshape(samples, shape)
    else:
        return samples[0] if samples else None


def ones(shape):
    if len(shape) == 0:
        raise ValueError("shape must be a non-empty tuple")

    result = []
    for _ in range(shape[0]):
        if len(shape) > 1:
            result.append(ones(shape[1:]))
        else:
            result.append(1)
    return result
def dot(A, B):
    # Check if the arrays can be multiplied
    if len(A.shape) != 1 and A.shape[1] != B.shape[0]:
        raise ValueError("Matrices dimensions are not aligned for dot product")
    result = 0
    for i in range(A.shape[0]):
        result += A[i] * B[i]

    return result
def reshape(arr, new_shape):
    # Determine total number of elements in arr
    total_elements = len(arr) if isinstance(arr, list) else 1

    # Check if the total number of elements matches the new shape
    if total_elements != new_shape[0] * new_shape[1]:
        raise ValueError(f"Total size of new array must be unchanged (got {total_elements} but expected {new_shape[0] * new_shape[1]})")

    # Create the reshaped array filled with zeros
    reshaped = zeros(new_shape)

    # Flatten the original array if it's 2D to iterate over its elements
    flat_arr = arr if isinstance(arr[0], list) else [arr]

    # Fill the reshaped array with the elements of the flattened original array
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            reshaped[i][j] = flat_arr[i * new_shape[1] + j]

    return reshaped

def zeros(shape):
    return [[0] * shape[1] for _ in range(shape[0])]


def array(lst):
    # Check if lst is a list of lists or a single list
    if isinstance(lst[0], list):
        # Recursively create array
        return [array(sublst) for sublst in lst]
    else:
        return lst
def arange(start, stop=None, step=1):
    if stop is None:
        stop = start
        start = 0

    result = []
    current = start

    while current < stop:
        result.append(current)
        current += step

    return result
def flatten(arr):
    flat_list = []
    for elem in arr:
        if isinstance(elem, list):
            flat_list.extend(flatten(elem))  # Recursively flatten sublists
        else:
            flat_list.append(elem)
    return flat_list

class ForwardLayer():
    def __init__(self, input_shape=None, neurons=1, bias=None, weights=None, activation=None, is_bias = True):
        random.seed(100)
        self.input_shape = input_shape
        self.neurons = neurons
        self.has_bias = is_bias
        self.name = ""
        self.w = weights
        self.b = bias
        if input_shape != None:
            self.output_shape = neurons
        if self.input_shape != None:
            self.weights = weights if weights != None else randn(self.input_shape, neurons)
            self.parameters = self.input_shape *  self.neurons + self.neurons if self.has_bias else 0
        if(is_bias):
            self.biases = bias if bias != None else randn(neurons)
        else:
            self.biases = 0
    #derivative function
    def activation_dfc(self, r):

        if self.activation is None:
            return ones(r.shape)
        if self.activation == 'tanh':
            return 1 - r ** 2
        if self.activation == 'sigmoid':
            # r = self.activation_fc(r)
            return r * (1 - r)
        if self.activation == "softmax":
            soft = self.activation_fc(r)
            diag_soft = soft*(1- soft)
            return diag_soft
        if self.activation == 'relu':
            r[r < 0] = 0
            r[r>=1]=1
            return r
        return r
    #action function
    def activation_fc(self, r):
        if self.activation == 'relu':
            r[r < 0] = 0
            return r
        if self.activation == None or self.activation == "linear":
            return r
        if self.activation == 'tanh':
            return math.tanh(r)
        if self.activation == 'sigmoid':
            return 1 / (1 + math.exp(-r))
        if self.activation == "softmax":
            r = r - math.max(r)
            s = math.exp(r)
            return s / math.sum(s)
    def apply_activation(self, x):
        soma = dot(x, self.weights) + self.biases
        self.out = self.activation_fc(soma)
        return self.out

    def set_n_input(self):
        self.weights = self.w if self.w != None else randn(size=(self.input_shape, self.neurons))

    def backpropagation(self, nx_layer):
        self.error = math.dot(nx_layer.weights, nx_layer.delta)
        self.delta = self.error * self.activation_dfc(self.out)
        self.d_weights += dot(self.input.T, self.delta)
        self.d_biases += self.delta

    def set_output_shape(self):
        self.set_n_input()
        self.output_shape = self.neurons
        self.get_parameters()

    def get_parameters(self):
        self.parameters = self.input_shape *  self.neurons + self.neurons if self.has_bias else 0
        return self.parameters

class Conv2d():
  def __init__(self, input_shape=None, filters=1, kernel_size = (3, 3), has_bias=True, activation=None, stride=(1, 1), padding="zero", kernel=None, bias=None):
        self.input_shape = input_shape
        self.filters = filters
        self.has_bias = has_bias
        self.activation = activation
        self.stride = stride
        self.padding = padding
        self.p = 1 if padding != None else 0
        self.bias = bias
        self.kernel = kernel
        if input_shape != None:
            self.kernel_size = (kernel_size[0], kernel_size[1], input_shape[2], filters)
            self.output_shape = (int((input_shape[0] - kernel_size[0] + 2 * self.p) / stride[0]) + 1,
                                int((input_shape[1] - kernel_size[1] + 2 * self.p) / stride[1]) + 1, filters)
            self.set_variables()
            self.out = zeros(self.output_shape)
        else:
            self.kernel_size = (kernel_size[0], kernel_size[1])
  def set_variables(self):
    self.weights = self.init_param(self.kernel_size)
    self.biases = self.init_param((self.filters, 1))
    self.parameters = math.multiply.reduce(self.kernel_size) + self.filters if self.has_bias else 1
    self.d_weights = zeros(self.kernel_size)
    self.d_biases = zeros(self.biases.shape)
  def init_param(self, size):
    stddev = 1/math.sqrt(math.prod(size))
    return normal(loc=0, scale=stddev, size=size)
  def activation_fc(self, r):
    if self.activation == None:
        return r
    if self.activation == 'tanh': #tanh
        return math.tanh(r)
    if self.activation == 'sigmoid':  # sigmoid
        return 1 / (1 + math.exp(-r))
    if self.activation == "softmax":# softmax
        r = r - math.max(r)
        s = math.exp(r)
        return s / math.sum(s)
  def set_output_shape(self):
        self.output_shape = (int((self.input_shape[0] - self.kernel_size[0] + 2 * self.p) / self.stride[0] + 1),
                            int((self.input_shape[1] - self.kernel_size[1] + 2 * self.p) / self.stride[1] + 1),self.input_shape[2])
        self.parameters = self.filters * self.input_shape[2] * self.kernel_size[0] * self.kernel_size[1] + self.filters
  def activation_dfc(self, r):
        if self.activation is None:
            return ones(r.shape)
        if self.activation == 'tanh':
            return 1 - r ** 2
        if self.activation == 'sigmoid':
            return r * (1 - r)
        if self.activtion == 'softmax':
            soft = self.activation_fc(r)
            return soft * (1 - soft)
        if self.activation == 'relu':
            r[r<0] = 0
            r[r>=1]=1
            return r
  def apply_activation(self, image):
        for f in range(self.filters):
            image = self.input
            kshape = self.kernel_size
            if kshape[0] % 2 != 1 or kshape[1] % 2 != 1:
                raise ValueError("Please provide odd length of 2d kernel.")
            if type(self.stride) == int:
                     stride = (stride, stride)
            else:
                stride = self.stride
            shape = image.shape
            if self.padding == "zero":
                zeros_h = zeros((shape[1], shape[2]))
                zeros_h = reshape(zeros_h, (  shape[1], shape[2]))
                zeros_v = zeros((shape[0]+2, shape[2]))
                zeros_v = reshape(zeros_v, (shape[0]+2, -1, shape[2]))
                padded_img = [zeros_h] + image + [zeros_h]
                padded_img = [zeros_v + row + zeros_v for row in padded_img]
                image = padded_img
                shape = image.shape
            elif self.padding == "same":
                hor1 = image[0].reshape(-1, shape[1], shape[2])
                hor2 = image[-1].reshape(-1, shape[1], shape[2])
                padded_img = [hor1] + image + [hor2]
                ver1 = image[:, 0].reshape(padded_img.shape[0], -1, shape[2])
                ver2 = image[:, -1].reshape(padded_img.shape[0], -1, shape[2])
                padded_img = [ver1 + row + ver2 for row in padded_img]
                image = padded_img
                shape = image.shape
            elif self.padding == None:
                pass
            rv = 0
            cimg = []
            for r in range(kshape[0], shape[0]+1, stride[0]):
                cv = 0
                for c in range(kshape[1], shape[1]+1, stride[1]):
                    chunk = image[rv:r, cv:c]
                    soma = chunk * self.weights[:, :, :, f]
                    summa = soma.sum()+self.biases[f]
                    cimg.append(summa)
                    cv+=stride[1]
                rv+=stride[0]
            cimg = array(cimg).reshape(int(rv/stride[0]), int(cv/stride[1]))
            self.out[:, :, f] = cimg
        self.out = self.activation_fc(self.out)
        return self.out
  def backpropagation(self, nx_layer):
        layer = self
        layer.delta = zeros((layer.input_shape[0], layer.input_shape[1], layer.input_shape[2]))
        image = layer.input
        for f in range(layer.filters):
            kshape = layer.kernel_size
            shape = layer.input_shape
            stride = layer.stride
            rv = 0
            i = 0
            for r in range(kshape[0], shape[0]+1, stride[0]):
                cv = 0
                j = 0
                for c in range(kshape[1], shape[1]+1, stride[1]):
                    chunk = image[rv:r, cv:c]
                    layer.delta_weights[:, :, :, f] += chunk * nx_layer.delta[i, j, f]
                    layer.delta[rv:r, cv:c, :] += nx_layer.delta[i, j, f] * layer.weights[:, :, :, f]
                    j+=1
                    cv+=stride[1]
                rv+=stride[0]
                i+=1
            layer.delta_biases[f] = math.sum(nx_layer.delta[:, :, f])
        layer.delta = layer.activation_dfc(layer.delta)

class Dropout:
    def __init__(self, prob = 0.5):
        self.input_shape=None
        self.output_shape = None
        self.input_data= None
        self.output = None
        self.has_bias = False
        self.activation = None
        self.parameters = 0
        self.delta = 0
        self.weights = 0
        self.bias = 0
        self.prob = prob
        self.d_weights = 0
        self.d_biases = 0
    def set_output_shape(self):
        self.output_shape = self.input_shape
        self.weights = 0
    def apply_activation(self, x, train=True):
        if train:
            self.input_data = x
            flat = array(self.input_data).flatten()
            random_indices = random.choice(len(flat), int(self.prob*len(flat)), replace=False)
            flat[random_indices] = 0
            self.output = flat.reshape(x.shape)
            return self.output
        else:
            self.input_data = x
            self.output = x / self.prob
            return self.output
    def activation_dfc(self, x):
        return x
    def backpropagation(self, nx_layer):
        if type(nx_layer).__name__ != "Conv2d":
            self.error = dot(nx_layer.weights, nx_layer.delta)
            self.delta = self.error * self.activation_dfc(self.out)
        else:
            self.delta = nx_layer.delta
        self.delta[self.output == 0] = 0

class Pool2d:
    def __init__(self, kernel_size = (2, 2), stride=None, kind="max", padding=None):
        self.input_shape=None
        self.output_shape = None
        self.input_data= None
        self.output = None
        self.has_bias = False
        self.activation = None
        self.parameters = 0
        self.delta = 0
        self.weights = 0
        self.bias = 0
        self.d_weights = 0
        self.d_biases = 0
        self.padding = padding
        self.p = 1 if padding != None else 0
        self.kernel_size = kernel_size
        if type(stride) == int:
                 stride = (stride, stride)
        self.stride = stride
        if self.stride == None:
            self.stride = self.kernel_size
        self.pools = ['max', "average", 'min']
        if kind not in self.pools:
            raise ValueError("Pool kind not understood.")
        self.kind = kind
    def set_output_shape(self):
        self.output_shape = (int((self.input_shape[0] - self.kernel_size[0] + 2 * self.p) / self.stride[0] + 1),
                            int((self.input_shape[1] - self.kernel_size[1] + 2 * self.p) / self.stride[1] + 1),self.input_shape[2])
    def apply_activation(self, image):
        stride = self.stride
        kshape = self.kernel_size
        shape = image.shape
        self.input_shape = shape
        self.set_output_shape()
        self.out = zeros((self.output_shape))
        for nc in range(shape[2]):
            cimg = []
            rv = 0
            for r in range(kshape[0], shape[0]+1, stride[0]):
                cv = 0
                for c in range(kshape[1], shape[1]+1, stride[1]):
                    chunk = image[rv:r, cv:c, nc]
                    if len(chunk) > 0:
                        if self.kind == "max":
                            chunk = math.max(chunk)
                        if self.kind == "min":
                            chunk = math.min(chunk)
                        if self.kind == "average":
                            chunk = math.mean(chunk)
                        cimg.append(chunk)
                    else:
                        cv-=cstep
                    cv+=stride[1]
                rv+=stride[0]
            cimg = math.array(cimg).reshape(int(rv/stride[0]), int(cv/stride[1]))
            self.out[:,:,nc] = cimg
        return self.out
    def backpropagation(self, nx_layer):
      layer = self
      stride = layer.stride
      kshape = layer.kernel_size
      image = layer.input
      shape = image.shape
      layer.delta = zeros(shape)
      cimg = []
      rstep = stride[0]
      cstep = stride[1]
      for f in range(shape[2]):
        i = 0
        rv = 0
        for r in range(kshape[0], shape[0]+1, rstep):
            cv = 0
            j = 0
            for c in range(kshape[1], shape[1]+1, cstep):
                chunk = image[rv:r, cv:c, f]
                dout = nx_layer.delta[i, j, f]
                if layer.kind == "max":
                    p = math.max(chunk)
                    index = math.argwhere(chunk == p)[0]
                    layer.delta[rv+index[0], cv+index[1], f] = dout
                if layer.kind == "min":
                    p = math.min(chunk)
                    index = math.argwhere(chunk == p)[0]
                    layer.delta[rv+index[0], cv+index[1], f] = dout
                if layer.kind == "average":
                    p = math.mean(chunk)
                    layer.delta[rv:r, cv:c, f] = dout
                j+=1
                cv+=cstep
            rv+=rstep
            i+=1

class Flatten:
    def __init__(self, input_shape=None):
        self.input_shape=None
        self.output_shape = None
        self.input_data= None
        self.output = None
        self.has_bias = False
        self.activation = None
        self.parameters = 0
        self.delta = 0
        self.weights = 0
        self.bias = 0
        self.delta_weights = 0
        self.delta_biases = 0
    def set_output_shape(self):
        self.output_shape = (self.input_shape[0] * self.input_shape[1] * self.input_shape[2])
        self.weights = 0
    def apply_activation(self, x):
        self.input_data = x
        self.output = array(self.input_data).flatten()
        return self.output
    def activation_dfc(self, x):
        return x
    def backpropagation(self, nx_layer):
        self.error = dot(nx_layer.weights, nx_layer.delta)
        self.delta = self.error * self.activation_dfc(self.out)
        self.delta = self.delta.reshape(self.input_shape)


#This optimizer get from this https://dataqoil.com/2020/06/05/writing-popular-machine-learning-optimizers-from-scratch-on-python/#google_vignette
class Optimizer:
    def __init__(self, layers, name=None, learning_rate=0.01, mr=0.001):
        self.name = name
        self.learning_rate = learning_rate
        self.mr = mr
        keys = ["sgd", "momentum", "rmsprop", "adagrad", "adam", "adamax", "adadelta"]
        values = [self.sgd, self.momentum, self.rmsprop, self.adagrad, self.adam, self.adamax, self.adadelta]
        self.opt_dict = {keys[i]: values[i] for i in range(len(keys))}
        if name is not None and name in keys:
            self.opt_dict[name](layers=layers, training=False)

    def sgd(self, layers, training=True):
        if training:
            for layer in layers:
                if hasattr(layer, 'weights'):
                    layer.weights -= self.learning_rate * layer.delta_weights
                    if layer.has_bias:
                        layer.biases -= self.learning_rate * layer.delta_biases

    def momentum(self, layers, learning_rate=0.1, beta1=0.9, weight_decay=0.0005, training=True):
        for l in layers:
            if l.parameters != 0:
                if training:
                    l.weights_momentum = beta1 * l.weights_momentum + learning_rate * l.delta_weights - weight_decay * learning_rate * l.weights
                    l.weights += l.weights_momentum
                    l.biases_momentum = beta1 * l.biases_momentum + learning_rate * l.delta_biases - weight_decay * learning_rate * l.biases
                    l.biases += l.biases_momentum
                else:
                    l.weights_momentum = 0
                    l.biases_momentum = 0

    def adagrad(self, layers, learning_rate=0.01, beta1=0.9, epsilon=1e-8, training=True):
        for l in layers:
            if l.parameters != 0:
                if training:
                    l.weights_adagrad += l.delta_weights ** 2
                    l.weights += learning_rate * (l.delta_weights / math.sqrt(l.weights_adagrad + epsilon))
                    l.biases_adagrad += l.delta_biases ** 2
                    l.biases += learning_rate * (l.delta_biases / math.sqrt(l.biases_adagrad + epsilon))
                else:
                    l.weights_adagrad = 0
                    l.biases_adagrad = 0

    def rmsprop(self, layers, learning_rate=0.001, beta1=0.9, epsilon=1e-8, training=True):
        for l in layers:
            if l.parameters != 0:
                if training:
                    l.weights_rms = beta1 * l.weights_rms + (1 - beta1) * (l.delta_weights ** 2)
                    l.weights += learning_rate * (l.delta_weights / math.sqrt(l.weights_rms + epsilon))
                    l.biases_rms = beta1 * l.biases_rms + (1 - beta1) * (l.delta_biases ** 2)
                    l.biases += learning_rate * (l.delta_biases / math.sqrt(l.biases_rms + epsilon))
                else:
                    l.weights_rms = 0
                    l.biases_rms = 0

    def adam(self, layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0, training=True):
        for l in layers:
            if l.parameters != 0:
                if training:
                    l.t += 1
                    if l.t == 1:
                        l.pdelta_biases = 0
                        l.pdelta_weights = 0
                    l.weights_adam1 = beta1 * l.weights_adam1 + (1 - beta1) * l.delta_weights
                    l.weights_adam2 = beta2 * l.weights_adam2 + (1 - beta2) * (l.delta_weights ** 2)
                    mcap = l.weights_adam1 / (1 - beta1 ** l.t)
                    vcap = l.weights_adam2 / (1 - beta2 ** l.t)
                    l.delta_weights = mcap / (math.sqrt(vcap) + epsilon)
                    l.weights += l.pdelta_weights * self.mr + learning_rate * l.delta_weights
                    l.pdelta_weights = l.delta_weights * 0
                    l.biases_adam1 = beta1 * l.biases_adam1 + (1 - beta1) * l.delta_biases
                    l.biases_adam2 = beta2 * l.biases_adam2 + (1 - beta2) * (l.delta_biases ** 2)
                    mcap = l.biases_adam1 / (1 - beta1 ** l.t)
                    vcap = l.biases_adam2 / (1 - beta2 ** l.t)
                    l.delta_biases = mcap / (math.sqrt(vcap) + epsilon)
                    l.biases += l.pdelta_biases * self.mr + learning_rate * l.delta_biases
                    l.pdelta_biases = l.delta_biases * 0
                else:
                    l.t = 0
                    l.weights_adam1 = 0
                    l.weights_adam2 = 0
                    l.biases_adam1 = 0
                    l.biases_adam2 = 0

    def adamax(self, layers, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8, training=True):
        for l in layers:
            if l.parameters != 0:
                if training:
                    l.weights_m = beta1 * l.weights_m + (1 - beta1) * l.delta_weights
                    l.weights_v = math.maximum(beta2 * l.weights_v, abs(l.delta_weights))
                    l.weights += (self.learning_rate / (1 - beta1)) * (l.weights_m / (l.weights_v + epsilon))
                    l.biases_m = beta1 * l.biases_m + (1 - beta1) * l.delta_biases
                    l.biases_v = math.maximum(beta2 * l.biases_v, abs(l.delta_biases))
                    l.biases += (self.learning_rate / (1 - beta1)) * (l.biases_m / (l.biases_v + epsilon))
                else:
                    l.weights_m = 0
                    l.biases_m = 0
                    l.weights_v = 0
                    l.biases_v = 0

    def adadelta(self, layers, learning_rate=0.01, beta1=0.9, epsilon=1e-8, training=True):
        for l in layers:
            if l.parameters != 0:
                if training:
                    l.weights_v = beta1 * l.weights_v + (1 - beta1) * (l.delta_weights ** 2)
                    l.delta_weights = math.sqrt((l.weights_m + epsilon) / (l.weights_v + epsilon)) * l.delta_weights
                    l.weights_m = beta1 * l.weights_m + (1 - beta1) * l.delta_weights
                    l.weights += l.delta_weights
                    l.biases_v = beta1 * l.biases_v + (1 - beta1) * (l.delta_biases ** 2)
                    l.delta_biases = math.sqrt((l.biases_m + epsilon) / (l.biases_v + epsilon)) * l.delta_biases
                    l.biases_m = beta1 * l.biases_m + (1 - beta1) * l.delta_biases
                    l.biases += l.delta_biases
                else:
                    l.weights_m = 0
                    l.biases_m = 0
                    l.weights_v = 0
                    l.biases_v = 0

class CNN():
    def __init__(self):
        self.layers = []
        self.info_df = {}
        self.column = ["LName", "Input Shape", "Output Shape", "Activation", "Bias"]
        self.parameters = []
        self.optimizer = ""
        self.loss = "mse"
        self.lr = 0.01
        self.mr = 0.0001
        self.metrics = []
        self.av_optimizers = ["sgd", "momentum", "adam"]
        self.av_metrics = ["mse", "accuracy", "cse"]
        self.av_loss = ["mse", "cse"]
        self.iscompiled = False
        self.model_dict = None
        self.out = []
        self.eps = 1e-15
        self.train_loss = {}
        self.val_loss = {}
        self.train_acc = {}
        self.val_acc = {}
    def add(self, layer):
        if(len(self.layers) > 0):
            prev_layer = self.layers[-1]
            if prev_layer.name != "Input Layer":
                prev_layer.name = f"{type(prev_layer).__name__}{len(self.layers) - 1}"
            if layer.input_shape == None:
                if type(layer).__name__ == "Flatten":
                        ops = prev_layer.output_shape[:]
                        if type(prev_layer).__name__ == "Pool2d":
                            ops = prev_layer.output_shape[:]
                elif type(layer).__name__ == "Conv2d":
                    ops = prev_layer.output_shape[:]
                    if type(prev_layer).__name__ == "Pool2d":
                        ops = prev_layer.output_shape
                elif type(layer).__name__ == "Pool2d":
                    ops = prev_layer.output_shape[:]
                    if type(prev_layer).__name__ == "Pool2d":
                        ops = prev_layer.output_shape[:]
                else:
                    ops = prev_layer.output_shape
                layer.input_shape = ops
                layer.set_output_shape()
            layer.name = f"Out Layer({type(layer).__name__})"
        else:
            layer.name = "Input Layer"
        if type(layer).__name__ == "Conv2d":
            if(layer.output_shape[0] <= 0 or layer.output_shape[1] <= 0):
                raise ValueError(f"The output shape became invalid [i.e. {layer.output_shape}].")
        self.layers.append(layer)
        self.parameters.append(layer.parameters)
    

m = CNN()
m.add(Conv2d(input_shape = (28, 28, 1), filters = 4, padding=None, kernel_size=(3, 3), activation="relu"))
m.add(Conv2d(filters=8, kernel_size=(3, 3), padding=None, activation="relu"))
m.add(Conv2d(filters=8, kernel_size=(2, 2), padding=None, activation="relu"))  # Replace Pool2d with Conv2d
m.add(Flatten())
m.add(ForwardLayer(neurons = 64, activation = "relu"))
m.add(Dropout(0.1))
m.add(ForwardLayer(neurons = 10, activation='softmax'))