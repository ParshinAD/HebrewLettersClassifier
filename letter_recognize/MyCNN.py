#!/usr/bin/env python
# coding: utf-8

# Importing needed library
import numpy as np
import pickle
from PIL import Image, ImageOps


def cnn_forward_naive(x, w, b, cnn_params):
    stride = cnn_params['stride']
    pad = cnn_params['pad']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    # Preparing cache for output
    cache = (x, w, b, cnn_params)

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

    # Calculating spatial sizes of output feature maps
    height_out = int(1 + (H + 2 * pad - HH) / stride)
    width_out = int(1 + (W + 2 * pad - WW) / stride)

    feature_maps = np.zeros((N, F, height_out, width_out))

    # For every image
    for n in range(N):
        # For every filter
        for f in range(F):
            height_index = 0
            for i in range(0, H, stride):
                width_index = 0
                for j in range(0, W, stride):
                    feature_maps[n, f, height_index, width_index] = np.sum(x_padded[n, :, i:i+HH, j:j+WW] * w[f, :, :, :]) + b[f]
                    # Increasing index for width
                    width_index += 1
                # Increasing index for height
                height_index += 1

    return feature_maps, cache


def absolute_error(x, y):
    return np.sum(np.abs(x - y))


def cnn_backward_naive(derivative_out, cache):
    """
    Defining function for Naive Backward Pass for Convolutional Layer.

    Input consists of following:

        derivatives_out - upstream derivatives.
        cache - tuple of shape (x, w, b, cnn_params), where:
            x of shape (N, C, H, W) - input, where N is number of input images
                and every of them with C channels, with height H and with width W.
            w of shape (F, C, HH, WW) - filters, with the help of which we convolve every input image
                with F different filters, where every filter spans all C channels and every filter
                has height HH and width WW.
            b of shape (F, ) - biases for every filter F.
            cnn_params - dictionary with parameters for convolution, where key stride is a step for sliding,
                and key pad is a zero-pad frame around input image.

    Function returns a tuple of (dx, dw, db):

        dx - gradient with respect to x.
        dw - gradient with respect to w.
        db - gradient with respect to b.
    """
    x, w, b, cnn_params = cache

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, height_out, weight_out = derivative_out.shape

    stride = cnn_params['stride']
    pad = cnn_params['pad']

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

    dx_padded = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

    # For every image
    for n in range(N):
        # For every filter
        for f in range(F):
            # Going through all input image through all channels
            for i in range(0, H, stride):
                for j in range(0, W, stride):
                    # Calculating gradients
                    dx_padded[n, :, i:i+HH, j:j+WW] += w[f, :, :, :] * derivative_out[n, f, i, j]
                    dw[f, :, :, :] += x_padded[n, :, i:i+HH, j:j+WW] * derivative_out[n, f, i, j]
                    db[f] += derivative_out[n, f, i, j]

    # Reassigning dx by slicing dx_padded
    dx = dx_padded[:, :, 1:-1, 1:-1]

    # Returning calculated gradients
    return dx, dw, db


def max_pooling_forward_naive(x, pooling_params):
    """
    Defining function for Naive Forward Pass for Max Pooling Layer.

    Input consists of following:

        x of shape (N, F, H, W) - input, where N is number of input images
            and every of them with F channels (number of feature maps after Convolutional Layer),
            with height H and with width W.
        pooling_params - dictionary with following keys:
            pooling_height - height of pooling region.
            pooling_width - width of pooling region.
            stride - step (distance) between pooling regions.

    Function returns a tuple of (pooled_output, cache):

        pooled_output - output resulted data with shape (N, F, H', W'), where:
            N - number of received batches of feature maps for every input image
                and is the same with number of input images.
            F - number of channels (number of feature maps after Convolutional Layer) for every input image.
            H' - height of received pooled data that is calculated by following equation:
                H' = 1 + (H + pooling_height) / stride
            W' - width of received pooled data that is calculated by following equation:
                W' = 1 + (W + pooling_width) / stride
        cache - tuple of shape (x, pooling_params), that is needed in backward pass.
    """
    N, F, H, W = x.shape

    pooling_height = pooling_params['pooling_height']
    pooling_width = pooling_params['pooling_width']
    stride = pooling_params['stride']

    cache = (x, pooling_params)

    height_pooled_out = int(1 + (H - pooling_height) / stride)
    width_polled_out = int(1 + (W - pooling_width) / stride)

    pooled_output = np.zeros((N, F, height_pooled_out, width_polled_out))

    # For every image
    for n in range(N):
        # Going through all input image through all channels
        for i in range(height_pooled_out):
            for j in range(width_polled_out):
                ii = i * stride
                jj = j * stride
                # Getting current pooling region with all channels F
                current_pooling_region = x[n, :, ii:ii+pooling_height, jj:jj+pooling_width]
                pooled_output[n, :, i, j] = np.max(current_pooling_region.reshape((F, pooling_height * pooling_width)), axis=1)

    return pooled_output, cache


def max_pooling_backward_naive(derivatives_out, cache):
    """
    Defining function for Naive Backward Pass for MAX Pooling Layer.

    Input consists of following:

        derivatives_out - upstream derivatives.
        cache - tuple of (x, pooling_params), where:
            x of shape (N, F, H, W) - input, where N is number of input images
                and every of them with F channels (number of feature maps after Convolutional Layer),
                with height H and with width W.
            pooling_params - dictionary with following keys:
                pooling_height - height of pooling region.
                pooling_width - width of pooling region.
                stride - step (distance) between pooling regions.

    Function returns derivatives calculated with Gradient Descent method:

        dx - gradient with respect to x.
    """
    x, pooling_params = cache
    N, F, H, W = x.shape

    pooling_height = pooling_params['pooling_height']
    pooling_width = pooling_params['pooling_width']
    stride = pooling_params['stride']

    height_pooled_out = int(1 + (H - pooling_height) / stride)
    width_polled_out = int(1 + (W - pooling_width) / stride)

    dx = np.zeros((N, F, H, W))

    # For every image
    for n in range(N):
        # For every channel
        for f in range(F):
            # Going through all pooled image by height and width
            for i in range(height_pooled_out):
                for j in range(width_polled_out):
                    ii = i * stride
                    jj = j * stride

                    current_pooling_region = x[n, f, ii:ii+pooling_height, jj:jj+pooling_width]
                    current_maximum = np.max(current_pooling_region)
                    temp = current_pooling_region == current_maximum
                    dx[n, f, ii:ii+pooling_height, jj:jj+pooling_width] += derivatives_out[n, f, i, j] * temp

    return dx



def fc_forward(x, w, b):
    """
    Defining function for Naive Forward Pass for Fully-Connected Layer (also known as Affine Layer).

    Input consists of following:

        x of shape (N, d1, ..., dk) - input data, where input x contains N batches and
            each batch x[i] has shape (d1, ..., dk).
        w of shape (D, M) - weights.
        b of shape (M,) - biases.
        We will reshape each input batch x[i] into vector of dimension D = d1 * ... * dk.
        As a result, input will be in form of matrix with shape (N, D).
        It is needed for calculation product of input matrix over weights.
        As weights' matrix has shape (D, M), then output resulted matrix will have shape (N, M).

    Function returns a tuple of (fc_output, cache):

        fc_output - output data of shape (N, M).
        cache - tuple of shape (x, w, b, cnn_params), that is needed in backward pass.
    """
    cache = (x, w, b)

    N = x.shape[0]
    x_reshaped = x.reshape(N, -1)
    fc_output = np.dot(x_reshaped, w) + b

    return fc_output, cache


def fc_backward(derivatives_out, cache):
    """
    Defining function for Naive Backward Pass for Fully-Connected Layer (also known as Affine Layer).

    Input consists of following:

        derivatives_out - upstream derivatives of shape (N, M).
        cache - tuple of (x, w, b), where:
            x of shape (N, d1, ..., dk) - input data.
            w of shape (D, M) - weights.
            b of shape (M,) - biases.

    Function returns a tuple of (dx, dw, db):

        dx - gradient with respect to x of shape (N, d1, ..., dk).
        dw - gradient with respect to w of shape (D, M).
        db - gradient with respect to b of shape (M,).
    """
    x, w, b = cache

    dx = np.dot(derivatives_out, w.T).reshape(x.shape)
    N = x.shape[0]
    x = x.reshape(N, -1)
    dw = np.dot(x.T, derivatives_out)
    db = np.dot(np.ones(dx.shape[0]), derivatives_out)

    return dx, dw, db


def relu_forward(x):
    """
    Defining function for Naive Forward Pass for ReLU activation.
    ReLU is the abbreviation for Rectified Linear Unit.

    Input consists of following:

        x of any shape - input data.

    Function returns a tuple of (relu_output, cache):

        relu_output - output data of the same shape as x.
        cache - is x, that is needed in backward pass.
    """
    cache = x
    relu_output = np.maximum(0, x)

    return relu_output, cache


def relu_backward(derivatives_out, cache):
    """
    Defining function for Naive Backward Pass for ReLU activation.

    Input consists of following:

        derivatives_out - upstream derivatives of any shape.
        cache - is x, of the same shape as derivatives_out.

    Function returns a tuple of (relu_output, cache):

        dx - gradient with respect to x.
    """
    x = cache

    temp = x > 0
    dx = temp * derivatives_out

    return dx


def softmax_loss(x, y):
    """
    Defining function for Softmax Classification Loss.

    Input consists of following:

        x of shape (N, C) - input data, where x[i, j] is score for the j-th class for the i-th input.
        y of shape (N, ) - vector of labels, where y[i] is the label for x[i] and 0 <= y[i] < C.

    Function returns a tuple of (loss, dx):

        loss - scalar giving the Logarithmic Loss.
        dx - gradient of loss with respect to x.
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probabilities = shifted_logits - np.log(z)
    probabilities = np.exp(log_probabilities)

    N = x.shape[0]

    loss = -np.sum(log_probabilities[np.arange(N), y]) / N

    dx = probabilities
    dx[np.arange(N), y] -= 1
    dx /= N

    return loss, dx


class ConvNet1(object):

    """""""""
    Initializing new Network
    Input consists of following:
        input_dimension of shape (C, H, W) - dimension of input data,
                                             where C channels, with height H and with width W.
        number_of_filters - number of filters to use in Convolutional Layer.
        size_of_filter - size of filter to use in Convolutional Layer.
        hidden_dimension - number of neurons to use in Fully-Connected Hidden Layer.
        number_of_classes - number of scores to produce from the final Fully-Connected Layer.
        weight_scale - scalar giving standard deviation for random initialization of weights.
        regularization - scala giving L2 regularization strength.
        dtype - numpy datatype to use for computation.
    """

    def __init__(self, input_dimension=(3, 32, 32), number_of_filters=32, size_of_filter=7,
                 hidden_dimension=100, number_of_classes=10, weight_scale=1e-3, regularization=0.0,
                 dtype=np.float32):

        self.params = {}
        self.regularization = regularization
        self.dtype = dtype

        C, H, W = input_dimension
        HH = WW = size_of_filter
        F = number_of_filters
        Hh = hidden_dimension
        Hclass = number_of_classes

        self.params['w1'] = weight_scale * np.random.rand(F, C, HH, WW)
        self.params['b1'] = np.zeros(F)

        # Defining parameters for Convolutional Layer (which is only one here)
        self.cnn_params = {'stride': 1, 'pad': int((size_of_filter - 1) / 2)}
        Hc = int(1 + (H + 2 * self.cnn_params['pad'] - HH) / self.cnn_params['stride'])
        Wc = int(1 + (W + 2 * self.cnn_params['pad'] - WW) / self.cnn_params['stride'])

        # Defining parameters for Max Pooling Layer:
        self.pooling_params = {'pooling_height': 2, 'pooling_width': 2, 'stride': 2}
        Hp = int(1 + (Hc - self.pooling_params['pooling_height']) / self.pooling_params['stride'])
        Wp = int(1 + (Wc - self.pooling_params['pooling_width']) / self.pooling_params['stride'])

        self.params['w2'] = weight_scale * np.random.rand(F * Hp * Wp, Hh)
        self.params['b2'] = np.zeros(Hh)

        self.params['w3'] = weight_scale * np.random.rand(Hh, Hclass)
        self.params['b3'] = np.zeros(Hclass)

        for d_key, d_value in self.params.items():
            self.params[d_key] = d_value.astype(dtype)


    def loss_for_training(self, x, y):
        """
        Evaluating loss for training ConvNet1.

        Input consists of following:

            x of shape (N, C, H, W) - input data,
                where N is number of images and every of them with C channels,
                with height H and with width W.
            y of shape (N, ) - vector of labels, where y[i] is the label for x[i].

        Function returns a tuple of (loss, gradients):

            loss - scalar giving the Logarithmic Loss.
            gradients - dictionary with the same keys as self.params,
                mapping parameter names to gradients of loss with respect to those parameters.
        """
        w1, b1 = self.params['w1'], self.params['b1']
        w2, b2 = self.params['w2'], self.params['b2']
        w3, b3 = self.params['w3'], self.params['b3']

        # Implementing forward pass for ConvNet1 and computing scores for every input
        # Forward pass:
        # Input --> Conv --> ReLU --> Pool --> FC --> ReLU --> FC --> Softmax
        cnn_output, cache_cnn = cnn_forward_naive(x, w1, b1, self.cnn_params)
        relu_output_1, cache_relu_1 = relu_forward(cnn_output)
        pooling_output, cache_pooling = max_pooling_forward_naive(relu_output_1, self.pooling_params)
        fc_hidden, cache_fc_hidden = fc_forward(pooling_output, w2, b2)
        relu_output_2, cache_relu_2 = relu_forward(fc_hidden)
        scores, cache_fc_output = fc_forward(relu_output_2, w3, b3)

        loss, d_scores = softmax_loss(scores, y)

        # Adding L2 regularization
        loss += 0.5 * self.regularization * np.sum(np.square(w1))
        loss += 0.5 * self.regularization * np.sum(np.square(w2))
        loss += 0.5 * self.regularization * np.sum(np.square(w3))

        # Backward pass through FC output
        dx3, dw3, db3 = fc_backward(d_scores, cache_fc_output)
        # Adding L2 regularization
        dw3 += self.regularization * w3

        # Backward pass through ReLU and FC Hidden
        d_relu_2 = relu_backward(dx3, cache_relu_2)
        dx2, dw2, db2 = fc_backward(d_relu_2, cache_fc_hidden)
        # Adding L2 regularization
        dw2 += self.regularization * w2

        # Backward pass through Pool, ReLU and Conv
        d_pooling = max_pooling_backward_naive(dx2, cache_pooling)
        d_relu_1 = relu_backward(d_pooling, cache_relu_1)
        dx1, dw1, db1 = cnn_backward_naive(d_relu_1, cache_cnn)
        # Adding L2 regularization
        dw1 += self.regularization * w1

        # Putting resulted derivatives into gradient dictionary
        gradients = dict()
        gradients['w1'] = dw1
        gradients['b1'] = db1
        gradients['w2'] = dw2
        gradients['b2'] = db2
        gradients['w3'] = dw3
        gradients['b3'] = db3

        return loss, gradients


    def scores_for_predicting(self, x):
        """
        Calculating Scores for Predicting.

        Input consists of following:

            x of shape (N, C, H, W) - input data,
                where N is number of images and every of them with C channels,
                with height H and with width W.

        Function returns:

            scores of shape (N, C) - classification scores,
                where score [i, C] is the classification score for x[i] and class C.
        """
        w1, b1 = self.params['w1'], self.params['b1']
        w2, b2 = self.params['w2'], self.params['b2']
        w3, b3 = self.params['w3'], self.params['b3']

        # Implementing forward pass for ConvNet1 and computing scores for every input
        # Forward pass:
        # Input --> Conv --> ReLU --> Pool --> FC --> ReLU --> FC --> Softmax
        cnn_output, _ = cnn_forward_naive(x, w1, b1, self.cnn_params)
        relu_output_1, _ = relu_forward(cnn_output)
        pooling_output, _ = max_pooling_forward_naive(relu_output_1, self.pooling_params)
        affine_hidden, _ = fc_forward(pooling_output, w2, b2)
        relu_output_2, _ = relu_forward(affine_hidden)
        scores, _ = fc_forward(relu_output_2, w3, b3)

        return scores


def adam(w, dw, config=None, learning_rate=1e-3):
    if config is None:
        config = {}

    config.setdefault('learning_rate', learning_rate)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))
    config.setdefault('v', np.zeros_like(w))
    config.setdefault('t', 0)

    config['t'] += 1
    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dw
    config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * (dw**2)

    mt = config['m'] / (1 - config['beta1']**config['t'])
    vt = config['v'] / (1 - config['beta2']**config['t'])

    next_w = w - config['learning_rate'] * mt / (np.sqrt(vt) + config['epsilon'])

    return next_w, config


class Train(object):

    """""""""
    Initializing new Train class
    Input consists of following required and Optional arguments.

    Required arguments consist of following:
        model - a modal object conforming parameters as described above,
        data - a dictionary with training and validating data.

    Optional arguments (**kwargs) consist of following:
        update_rule - a string giving the name of an update rule in optimize_rules.py,
        optimization_config - a dictionary containing hyperparameters that will be passed
                              to the chosen update rule. Each update rule requires different
                              parameters, but all update rules require a 'learning_rate' parameter.
        learning_rate_decay - a scalar for learning rate decay. After each epoch the 'learning_rate'
                              is multiplied by this value,
        batch_size - size of minibatches used to compute loss and gradients during training,
        number_of_epochs - the number of epoch to run for during training,
        print_every - integer number that corresponds to printing loss every 'print_every' iterations,
        verbose_mode - boolean that corresponds to condition whether to print details or not.

    """

    def __init__(self, model, data, **kwargs):
        self.model = model
        self.x_train = data['x_train']
        self.y_train = data['y_train']
        self.x_validation = data['x_validation']
        self.y_validation = data['y_validation']

        self.optimization_config = kwargs.pop('optimization_config', {})  # Default is '{}'
        self.learning_rate_decay = kwargs.pop('learning_rate_decay', 1.0)  # Default is '1.0'
        self.batch_size = kwargs.pop('batch_size', 100)  # Default is '100'
        self.number_of_epochs = kwargs.pop('number_of_epochs', 10)  # Default is '10'
        self.print_every = kwargs.pop('print_every', 10)  # Default is '10'
        self.verbose_mode = kwargs.pop('verbose_mode', True)  # Default is 'True'

        if len(kwargs) > 0:
            extra = ', '.join(k for k in kwargs.keys())
            raise ValueError('Extra argument:', extra)

        self.update_rule = adam
        self._reset()


    def _reset(self):
        # Setting up variables
        self.current_epoch = 0
        self.best_validation_accuracy = 0
        self.best_params = {}
        self.loss_history = []
        self.train_accuracy_history = []
        self.validation_accuracy_history = []

        self.optimization_configurations = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optimization_config.items()}
            self.optimization_configurations[p] = d


    def _step(self):
        number_of_training_images = self.x_train.shape[0]
        batch_mask = np.random.choice(number_of_training_images, self.batch_size)
        x_batch = self.x_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        loss, gradient = self.model.loss_for_training(x_batch, y_batch)

        self.loss_history.append(loss)

        for p, v in self.model.params.items():
            dw = gradient[p]
            config_for_current_p = self.optimization_configurations[p]
            next_w, next_configuration = self.update_rule(v, dw, config_for_current_p)
            self.model.params[p] = next_w
            self.optimization_configurations[p] = next_configuration


    def check_accuracy(self, x, y, number_of_samples=None, batch_size=100):
        """""""""
        Input consists of following:
            x of shape (N, C, H, W) - N data, each with C channels, height H and width W,
            y - vector of labels of shape (N,),
            number_of_samples - subsample data and test model only on this number of data,
            batch_size - split x and y into batches of this size to avoid using too much memory.

        Function returns:
            accuracy - scalar number giving percentage of images
                       that were correctly classified by model.
        """

        N = x.shape[0]

        if number_of_samples is not None and N > number_of_samples:
            batch_mask = np.random.choice(N, number_of_samples)
            N = number_of_samples
            x = x[batch_mask]
            y = y[batch_mask]

        number_of_batches = int(N / batch_size)
        if N % batch_size != 0:
            number_of_batches += 1

        y_predicted = []

        for i in range(number_of_batches):
            s = i * batch_size
            e = (i + 1) * batch_size
            scores = self.model.scores_for_predicting(x[s:e])
            y_predicted.append(np.argmax(scores, axis=1))

        y_predicted = np.hstack(y_predicted)
        accuracy = np.mean(y_predicted == y)
        return accuracy


    def run(self):
        number_of_training_images = self.x_train.shape[0]
        iterations_per_one_epoch = int(max(number_of_training_images / self.batch_size, 1))
        iterations_total = int(self.number_of_epochs * iterations_per_one_epoch)
        for t in range(iterations_total):
            self._step()
            if self.verbose_mode and t % self.print_every == 0:

                print('Iteration: ' + str(t + 1) + '/' + str(iterations_total) + ',',
                      'loss =', self.loss_history[-1])

            end_of_current_epoch = (t + 1) % iterations_per_one_epoch == 0
            if end_of_current_epoch:
                self.current_epoch += 1
                for k in self.optimization_configurations:
                    self.optimization_configurations[k]['learning_rate'] *= self.learning_rate_decay
            first_iteration = (t == 0)
            last_iteration = (t == iterations_total - 1)

            if first_iteration or last_iteration or end_of_current_epoch:
                training_accuracy = self.check_accuracy(self.x_train, self.y_train,
                                                        number_of_samples=1000)

                validation_accuracy = self.check_accuracy(self.x_validation, self.y_validation)

                self.train_accuracy_history.append(training_accuracy)
                self.validation_accuracy_history.append(validation_accuracy)

                if self.verbose_mode:

                    print('Epoch: ' + str(self.current_epoch) + '/' + str(self.number_of_epochs) + ',',
                          'Training accuracy = ' + str(training_accuracy) + ',',
                          'Validation accuracy = ' + str(validation_accuracy))

                if validation_accuracy > self.best_validation_accuracy:
                    self.best_validation_accuracy = validation_accuracy
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v

        self.model.params = self.best_params

        with open('model_params_ConvNet1.pickle', 'wb') as f:
            pickle.dump(self.model.params, f)

        history_dictionary = {'loss_history': self.loss_history,
                              'train_accuracy_history': self.train_accuracy_history,
                              'validation_history': self.validation_accuracy_history}
        with open('model_histories_ConvNet1.pickle', 'wb') as f:
            pickle.dump(history_dictionary, f)


def download_local_image(filename, folder='../data/'):
        imageFile = Image.open(folder + filename)
        return imageFile


def process_image(img):
    img = ImageOps.grayscale(img)
    bbox = Image.eval(img, lambda x: 255-x).getbbox()
    if bbox is None:
        new_image = Image.new('L', (28,28), 255)
        crop_img = img.resize((20, 20), Image.NEAREST)

        new_image.paste(crop_img, (0, 0))

        return np.array(new_image)

    widthlen = bbox[2] - bbox[0]
    heightlen = bbox[3] - bbox[1]

    if widthlen > heightlen:
        widthlen = int(20 * heightlen / widthlen)
        heightlen = 20
    else:
        heightlen = int(20 * widthlen / heightlen)
        widthlen = 20

    wstart = (28 - widthlen) // 2
    hstart = (28 - heightlen) // 2

    new_image = Image.new('L', (28,28), 255)
    crop_img = img.crop(bbox).resize((widthlen, heightlen), Image.NEAREST)

    new_image.paste(crop_img, (wstart, hstart))

    new_image = 1 - np.array(new_image)/255
    return np.expand_dims(new_image, -1)


def CNN_predict_on_one_image(image, parameters):
    img = process_image(image)
    img = np.moveaxis(img, 2, -3)
    img = np.expand_dims(img, 0)

    # Forward propagation
    model = ConvNet1(input_dimension=(28, 28, 1), weight_scale=1e-2, hidden_dimension=100, number_of_classes=22)
    model.params = parameters
    scores = model.scores_for_predicting(img)

    p = scores.argmax(axis=0)
    return scores


def CNN_train_on_one_image(image, parameters, Y, learning_rate=0.0025):
    img = process_image(image)
    img = np.moveaxis(img, 2, -3)
    img = np.expand_dims(img, 0)

    # Forward propagation
    model = ConvNet1(input_dimension=(28, 28, 1), weight_scale=1e-2, hidden_dimension=100, number_of_classes=22, regularization=1e-3)
    model.params = parameters
    train_step_CNN(img, [Y], model, adam, learning_rate)

    return model.params


def train_step_CNN(img, y, model, update_rule, learning_rate=0.0025):
    loss, gradient = model.loss_for_training(img, y)

    for p, v in model.params.items():
        dw = gradient[p]
        next_w, next_configuration = update_rule(v, dw, learning_rate=learning_rate)
        # Updating value in 'params'
        model.params[p] = next_w



