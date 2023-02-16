from .FCC_model import process_image, L_model_forward, L_model_backward, update_parameters
from PIL import Image
import io
import numpy as np


def softmax(x):
    return(np.exp(x)/np.exp(x).sum())


def predict_on_one_image(image,):
    imageStream = io.BytesIO(image)
    image = Image.open(imageStream)
    image = process_image(image)

    if image is None:
        return {'answer': "Can't predict, when nothing is drawn"}

    image = image.reshape(-1, 1)

    # Forward propagation for constant fcc
    const_parameters_fcc = np.load("tmp/parameters.npy", allow_pickle=True)[()]
    probas_const_fcc, caches = L_model_forward(image, const_parameters_fcc)
    # Forward propagation for trainable fcc
    trainable_parametes_fcc = np.load("parameters.npy", allow_pickle=True)[()]
    probas_train_fcc, caches = L_model_forward(image, trainable_parametes_fcc)

    result = make_answer(probas_const_fcc, probas_train_fcc)

    return result


def make_answer(probas_const_fcc, probas_train_fcc):
    digit2labels = {0: 'Alef', 1: 'Ayin',  2: 'Bet',  3: 'Chet', 4: 'Dalet',
      5: 'Gimel', 6: 'He', 7: 'Kaf', 8: 'Lamed', 9: 'Mem', 10: 'Nun', 11: 'Pe',
      12: 'Qof', 13: 'Resh', 14: 'Samech', 15: 'Shin', 16: 'Tav', 17: 'Tet',
      18: 'Tsadi', 19: 'Vav', 20: 'Yod', 21: 'Zayin'}

    probas_const_fcc = sorted(enumerate(softmax(probas_const_fcc)), key=lambda x: -x[1])[:3]
    probas_train_fcc = sorted(enumerate(softmax(probas_train_fcc)), key=lambda x: -x[1])[:3]

    result = {'fnn': [f'{digit2labels[probas_const_fcc[0][0]]}: {round(float(probas_const_fcc[0][1])*100, 2)}%',
                        f'{digit2labels[probas_const_fcc[1][0]]}: {round(float(probas_const_fcc[1][1])*100, 2)}%;',
                        f'{digit2labels[probas_const_fcc[2][0]]}: {round(float(probas_const_fcc[2][1])*100, 2)}%;',],
            'fnn_t': [f'{digit2labels[probas_train_fcc[0][0]]}: {round(float(probas_train_fcc[0][1])*100, 2)}%',
                        f'{digit2labels[probas_train_fcc[1][0]]}: {round(float(probas_train_fcc[1][1])*100, 2)}%;',
                        f'{digit2labels[probas_train_fcc[2][0]]}: {round(float(probas_train_fcc[2][1])*100, 2)}%;',],
              'answer': digit2labels[probas_train_fcc[0][0]]}
    return result


def one_step_train(image, Y, learning_rate=0.0075):
    parameters = np.load("parameters.npy", allow_pickle=True)[()]

    labels2digit = {'Alef': 0, 'Ayin': 1, 'Bet': 2, 'Chet': 3, 'Dalet': 4,
     'Gimel': 5, 'He': 6, 'Kaf': 7, 'Lamed': 8, 'Mem': 9, 'Nun': 10, 'Pe': 11,
      'Qof': 12, 'Resh': 13, 'Samech': 14, 'Shin': 15, 'Tav': 16, 'Tet': 17,
       'Tsadi': 18, 'Vav': 19, 'Yod': 20, 'Zayin': 21}
    Y = labels2digit[Y.split(':')[0]]

    imageStream = io.BytesIO(image)
    image = Image.open(imageStream)
    image = process_image(image)
    image = image.reshape(-1, 1)

    label = [0.0]*22
    label[Y] = 1
    label = np.array(label).reshape(-1, 1)

    AL, caches = L_model_forward(image, parameters)

    grads = L_model_backward(AL, label, caches)

    parameters = update_parameters(parameters, grads, learning_rate)

    np.save("parameters.npy", parameters)

    return 'Trained!'