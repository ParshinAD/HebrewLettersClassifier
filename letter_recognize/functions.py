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
    const_parameters_fcc = np.load("tmp/parameters_with_sofits.npy", allow_pickle=True)[()]
    probas_const_fcc, _ = L_model_forward(image_flatten, const_parameters_fcc, 'softmax')

    # Forward propagation for trainable fcc
    trainable_parametes_fcc = np.load("parameters_with_sofits.npy", allow_pickle=True)[()]
    probas_train_fcc, _ = L_model_forward(image_flatten, trainable_parametes_fcc, 'softmax')

    # Forward propagation for constant CNN
    with open('tmp/model_params_ConvNet1_with_sofits.pickle', 'rb') as f:
        const_parameters_cnn = pickle.load(f, encoding='latin1')
    probas_const_cnn = CNN_predict_on_one_image(imagePIL, const_parameters_cnn)

    # Forward propagation for trainable CNN
    with open('model_params_ConvNet1_with_sofits_trainable.pickle', 'rb') as f:
        train_parameters_cnn = pickle.load(f, encoding='latin1')

    probas_trainable_cnn = CNN_predict_on_one_image(imagePIL, train_parameters_cnn)
    result = make_answer(probas_const_fcc, probas_train_fcc, probas_const_cnn, probas_trainable_cnn)
    return result


def make_answer(probas_const_fcc, probas_train_fcc, probas_const_cnn, probas_trainable_cnn):
    digit2labels = {0: 'Alef', 1: 'Ayin', 2: 'Bet', 3: 'Chet', 4: 'Dalet', 5: 'Gimel',
     6: 'He', 7: 'Kaf', 8: 'Kaf sofit', 9: 'Lamed', 10: 'Mem', 11: 'Mem sofit',
      12: 'Nun', 13: 'Nun sofit', 14: 'Pe', 15: 'Pe sofit', 16: 'Qof', 17: 'Resh',
       18: 'Samech', 19: 'Shin', 20: 'Tav', 21: 'Tet', 22: 'Tsadi', 23: 'Tsadi sofit',
        24: 'Vav', 25: 'Yod', 26: 'Zayin', -1: "Can't recognize this as a letter"}

    probas_const_fcc = sorted(enumerate(probas_const_fcc), key=lambda x: -x[1])[:3]
    probas_train_fcc = sorted(enumerate(probas_train_fcc), key=lambda x: -x[1])[:3]

    probas_const_cnn = probas_const_cnn.reshape([-1, 1])
    probas_const_cnn = sorted(enumerate(probas_const_cnn), key=lambda x: -x[1])[:3]

    probas_trainable_cnn = probas_trainable_cnn.reshape([-1, 1])
    probas_trainable_cnn = sorted(enumerate(probas_trainable_cnn), key=lambda x: -x[1])[:3]

    if probas_train_fcc[0][0] == probas_trainable_cnn[0][0]:
        answer = probas_train_fcc[0][0]
    elif probas_train_fcc[0][1] < 0.50 and probas_trainable_cnn[0][1] < 0.50:
        answer = -1
    elif probas_train_fcc[0][1] > probas_trainable_cnn[0][1]:
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


def one_step_train(image, Y,):
    parameters_FNN = np.load("parameters_with_sofits.npy", allow_pickle=True)[()]
    labels2digit = {'Alef': 0,  'Ayin': 1,  'Bet': 2,  'Chet': 3,  'Dalet': 4,
    'Gimel': 5, 'He': 6, 'Kaf': 7, 'Kaf sofit': 8, 'Lamed': 9, 'Mem': 10,
     'Mem sofit': 11, 'Nun': 12, 'Nun sofit': 13, 'Pe': 14, 'Pe sofit': 15,
      'Qof': 16, 'Resh': 17, 'Samech': 18, 'Shin': 19, 'Tav': 20, 'Tet': 21,
       'Tsadi': 22, 'Tsadi sofit': 23, 'Vav': 24, 'Yod': 25, 'Zayin': 26}
    Y = labels2digit[Y.split(':')[0]]

    imageStream = io.BytesIO(image)
    imageStream = Image.open(imageStream)
    # step for FNN
    image = process_image(imageStream)
    image = image.reshape(-1, 1)

    label = [0.0]*27
    label[Y] = 1
    label = np.array(label).reshape(-1, 1)

    AL, caches = L_model_forward(image, parameters_FNN)

    grads = L_model_backward(AL, label, caches)
    parameters_FNN = update_parameters(parameters_FNN, grads, learning_rate=0.0001)
    np.save("parameters_with_sofits.npy", parameters_FNN)

    # step for CNN
    with open('model_params_ConvNet1_with_sofits_trainable.pickle', 'rb') as f:
        train_parameters_cnn = pickle.load(f, encoding='latin1')  # dictionary type
    train_parameters_cnn = CNN_train_on_one_image(imageStream, train_parameters_cnn, Y, learning_rate=0.00003)
    with open('model_params_ConvNet1_with_sofits_trainable.pickle', 'wb') as f:
        pickle.dump(train_parameters_cnn, f)

    return 'Trained!'




