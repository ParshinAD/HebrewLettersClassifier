from .FCC_model import process_image, L_model_forward, L_model_backward, update_parameters
from .MyCNN import CNN_predict_on_one_image, CNN_train_on_one_image

import pickle
from PIL import Image
import io
import numpy as np


def softmax(x):
    return(np.exp(x)/np.exp(x).sum())


def predict_on_one_image(image,):
    imageStream = io.BytesIO(image)
    imagePIL = Image.open(imageStream)
    image = process_image(imagePIL)

    if image is None:
        return {'answer': "Can't predict, when nothing is drawn"}
    image_flatten = image.reshape(-1, 1)

    # Forward propagation for constant fcc
    const_parameters_fcc = np.load("tmp/parameters.npy", allow_pickle=True)[()]
    probas_const_fcc, _ = L_model_forward(image_flatten, const_parameters_fcc)

    # Forward propagation for trainable fcc
    trainable_parametes_fcc = np.load("parameters.npy", allow_pickle=True)[()]
    probas_train_fcc, _ = L_model_forward(image_flatten, trainable_parametes_fcc)

    # Forward propagation for constant CNN
    with open('tmp/model_params_ConvNet1.pickle', 'rb') as f:
        const_parameters_cnn = pickle.load(f, encoding='latin1')
    probas_const_cnn = CNN_predict_on_one_image(imagePIL, const_parameters_cnn)

    # Forward propagation for trainable CNN
    with open('model_params_ConvNet1_trainable.pickle', 'rb') as f:
        train_parameters_cnn = pickle.load(f, encoding='latin1')

    probas_trainable_cnn = CNN_predict_on_one_image(imagePIL, train_parameters_cnn)
    result = make_answer(probas_const_fcc, probas_train_fcc, probas_const_cnn, probas_trainable_cnn)
    return result


def make_answer(probas_const_fcc, probas_train_fcc, probas_const_cnn, probas_trainable_cnn):
    digit2labels = {0: 'Alef', 1: 'Ayin',  2: 'Bet',  3: 'Chet', 4: 'Dalet',
      5: 'Gimel', 6: 'He', 7: 'Kaf', 8: 'Lamed', 9: 'Mem', 10: 'Nun', 11: 'Pe',
      12: 'Qof', 13: 'Resh', 14: 'Samech', 15: 'Shin', 16: 'Tav', 17: 'Tet',
      18: 'Tsadi', 19: 'Vav', 20: 'Yod', 21: 'Zayin'}

    probas_const_fcc = sorted(enumerate(softmax(probas_const_fcc)), key=lambda x: -x[1])[:3]
    probas_train_fcc = sorted(enumerate(softmax(probas_train_fcc)), key=lambda x: -x[1])[:3]

    probas_const_cnn = probas_const_cnn.reshape([-1, 1])
    probas_const_cnn = sorted(enumerate(softmax(probas_const_cnn)), key=lambda x: -x[1])[:3]

    probas_trainable_cnn = probas_trainable_cnn.reshape([-1, 1])
    probas_trainable_cnn = sorted(enumerate(softmax(probas_trainable_cnn)), key=lambda x: -x[1])[:3]

    if probas_train_fcc[0][0] == probas_trainable_cnn[0][0]:
        answer = probas_train_fcc[0][0]
    elif probas_train_fcc[0][0] == probas_const_cnn[0][0] and probas_const_fcc[0][0] == probas_const_cnn[0][0]:
        answer = probas_train_fcc[0][0]
    elif probas_trainable_cnn[0][1] < 0.5:
        answer = probas_train_fcc[0][0]
    else:
        answer = probas_trainable_cnn[0][0]

    result = {'fnn': [f'{digit2labels[probas_const_fcc[0][0]]}: {round(float(probas_const_fcc[0][1])*100, 2)}%',
                        f'{digit2labels[probas_const_fcc[1][0]]}: {round(float(probas_const_fcc[1][1])*100, 2)}%;',
                        f'{digit2labels[probas_const_fcc[2][0]]}: {round(float(probas_const_fcc[2][1])*100, 2)}%;',],
            'fnn_t': [f'{digit2labels[probas_train_fcc[0][0]]}: {round(float(probas_train_fcc[0][1])*100, 2)}%',
                        f'{digit2labels[probas_train_fcc[1][0]]}: {round(float(probas_train_fcc[1][1])*100, 2)}%;',
                        f'{digit2labels[probas_train_fcc[2][0]]}: {round(float(probas_train_fcc[2][1])*100, 2)}%;',],
            'cnn': [f'{digit2labels[probas_const_cnn[0][0]]}: {round(float(probas_const_cnn[0][1])*100, 2)}%',
                        f'{digit2labels[probas_const_cnn[1][0]]}: {round(float(probas_const_cnn[1][1])*100, 2)}%;',
                        f'{digit2labels[probas_const_cnn[2][0]]}: {round(float(probas_const_cnn[2][1])*100, 2)}%;',],
            'cnn_t': [f'{digit2labels[probas_trainable_cnn[0][0]]}: {round(float(probas_trainable_cnn[0][1])*100, 2)}%',
                        f'{digit2labels[probas_trainable_cnn[1][0]]}: {round(float(probas_trainable_cnn[1][1])*100, 2)}%;',
                        f'{digit2labels[probas_trainable_cnn[2][0]]}: {round(float(probas_trainable_cnn[2][1])*100, 2)}%;',],
              'answer': digit2labels[answer]}
    return result


def one_step_train(image, Y, learning_rate=0.0005):
    parameters_FNN = np.load("parameters.npy", allow_pickle=True)[()]
    labels2digit = {'Alef': 0, 'Ayin': 1, 'Bet': 2, 'Chet': 3, 'Dalet': 4,
     'Gimel': 5, 'He': 6, 'Kaf': 7, 'Lamed': 8, 'Mem': 9, 'Nun': 10, 'Pe': 11,
      'Qof': 12, 'Resh': 13, 'Samech': 14, 'Shin': 15, 'Tav': 16, 'Tet': 17,
       'Tsadi': 18, 'Vav': 19, 'Yod': 20, 'Zayin': 21}
    Y = labels2digit[Y.split(':')[0]]

    imageStream = io.BytesIO(image)
    imageStream = Image.open(imageStream)
    # step for FNN
    image = process_image(imageStream)
    image = image.reshape(-1, 1)

    label = [0.0]*22
    label[Y] = 1
    label = np.array(label).reshape(-1, 1)

    AL, caches = L_model_forward(image, parameters_FNN)

    grads = L_model_backward(AL, label, caches)
    parameters_FNN = update_parameters(parameters_FNN, grads, learning_rate/10)
    np.save("parameters.npy", parameters_FNN)

    # step for CNN
    with open('model_params_ConvNet1_trainable.pickle', 'rb') as f:
        train_parameters_cnn = pickle.load(f, encoding='latin1')  # dictionary type
    train_parameters_cnn = CNN_train_on_one_image(imageStream, train_parameters_cnn, Y)
    with open('model_params_ConvNet1_trainable.pickle', 'wb') as f:
        pickle.dump(train_parameters_cnn, f)

    return 'Trained!'




