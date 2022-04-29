# Simple Starter
STARTER = {
    'input': (128, 128),
    'convolutional': [
        {'input': None, 'output': 16, 'kernel': 3, 'stride': 1, 'padding': 1},
        {'input': 16, 'output': 32, 'kernel': 3, 'stride': 1, 'padding': 1},
    ],
    'pooling': [
        {'kernel': 2, 'stride': 2},
        {'kernel': 2, 'stride': 2}
    ],
    'hidden': [
        {'input': 32768, 'output': 2048}
    ],
    'output': None,
    'pool position': [1, 2]
}


# Inspired From:
# A. M. Badshah, J. Ahmad, N. Rahim and S. W. Baik, 
# "Speech Emotion Recognition from Spectrograms with Deep Convolutional Neural Network,"
# 2017 International Conference on Platform Technology and Service (PlatCon), 2017, 
# pp. 1-5, doi:10.1109/PlatCon.2017.7883728.
# https://www.researchgate.net/publication/315638464_Speech_Emotion_Recognition_from_Spectrograms_with_Deep_Convolutional_Neural_Network
REF_ONE = {
    'input': (256, 256),
    'convolutional': [
        {'input': None, 'output': 120, 'kernel': 11, 'stride': 3, 'padding': 5},
        {'input': 120, 'output': 256, 'kernel': 5, 'stride': 1, 'padding': 2},
        {'input': 256, 'output': 384, 'kernel': 3, 'stride': 1, 'padding': 1}
    ],
    'pooling': [
        {'kernel': 4, 'stride': 2},
        {'kernel': 4, 'stride': 2},
        {'kernel': 4, 'stride': 2}
    ],
    'hidden': [
        {'input': 31104, 'output': 8192},
        {'input': 8192, 'output': 2048}
    ],
    'output': None,
    'pool position': [1, 2, 3]
}


# Inspired From:
# Somayeh Shahsavarani, "Speech Emotion Recognition using Convolutional Neural Networks",
# Masters dissertation,The Graduate College, University of Nebraska, Lincoln 2018. 
# Accessed on: August 23, 2021. [Online]. Available:
# https://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=1165&context=computerscidiss
REF_TWO = {
    'input': (128, 128),
    'convolutional': [
        {'input': None, 'output': 8, 'kernel': 5, 'stride': 1, 'padding': 2},
        {'input': 8, 'output': 16, 'kernel': 5, 'stride': 1, 'padding': 2}
    ],
    'pooling': [
        {'kernel': 2, 'stride': 2},
        {'kernel': 2, 'stride': 2}
    ],
    'hidden': [
        {'input': 16384, 'output': 2048}
    ],
    'output': None,
    'pool position': [1, 2]
}


# Inspired From:
# Kim, N., Lee, J., Ha, H., Lee, G., Lee, J. and Kim, H., 2017.
# Speech Emotion Recognition Based on Multi-task Learning using a Convolutional Neural Network. 
# In: 2017 APSIPA Annual Summit and Conference. pp.704-707.
# https://www.researchgate.net/publication/323193422_Speech_emotion_recognition_based_on_multi-task_learning_using_a_convolutional_neural_network
REF_THREE = {
    'input': (128, 128),
    'convolutional': [
        {'input': None, 'output': 32, 'kernel': 5, 'stride': 1, 'padding': 2},
        {'input': 32, 'output': 32, 'kernel': 3, 'stride': 1, 'padding': 1}
    ],
    'pooling': [
        {'kernel': 2, 'stride': 2},
        {'kernel': 2, 'stride': 2}
    ],
    'hidden': [
        {'input': 32768, 'output': 2048}
    ],
    'output': None,
    'pool position': [1, 2]
}


# AlexNet:
# Krizhevsky, A., Sutskever, I. and Hinton, G., 2012. 
# ImageNet Classification with Deep Convolutional Neural Networks. 
# In: 25th International Conference on Neural Information Processing Systems. 
# [online] NY, United States: Curran Associates Inc., pp.1097–1105. 
# Available at: <https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf>.
REF_FOUR = {
    'input': (227, 227),
    'convolutional': [
        {'input': None, 'output': 96, 'kernel': 11, 'stride': 4, 'padding': 0},
        {'input': 96, 'output': 256, 'kernel': 5, 'stride': 1, 'padding': 2},
        {'input': 256, 'output': 384, 'kernel': 3, 'stride': 1, 'padding': 1},
        {'input': 384, 'output': 384, 'kernel': 3, 'stride': 1, 'padding': 1},
        {'input': 384, 'output': 256, 'kernel': 3, 'stride': 1, 'padding': 1}
    ],
    'pooling': [
        {'kernel': 3, 'stride': 2},
        {'kernel': 3, 'stride': 2},
        {'kernel': 3, 'stride': 2}
    ],
    'hidden': [
        {'input': 9216, 'output': 4096},
        {'input': 4096, 'output': 4096},
        {'input': 4096, 'output': 1000}
    ],
    'output': None,
    'pool position': [1, 2, 5]
}

