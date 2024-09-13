# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#
# import os
#
# import joblib
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from tensorflow import keras
# from tensorflow.keras import optimizers
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.models import load_model
#
#
# def create_nn_regressor(ninput=4, nhidden=10, noutput=1, dropout_rate=0.5,
#                         activation_hidden="sigmoid", activation_output="sigmoid"):
#     """ An easy way of creating DNN with 2 dense layers (1 hidden layer)
#
#     Parameters
#     ----------
#     ninput:
#         input shape
#     nhidden: tuple
#         number of neurons in dense layers
#     dropout_rate:
#         dropout rate
#     noutput:
#         output shape
#     activation_hidden:
#         the activation function used in hidden layers
#     activation_output:
#         the activation function used in output layers
#
#     Returns
#     -------
#
#     """
#     model = Sequential()
#     model.add(Input(shape=(ninput,),))
#     # model.add(Dropout(dropout_rate))
#     model.add(Dense(nhidden, activation=activation_hidden))
#     model.add(BatchNormalization())
#     model.add(Dropout(dropout_rate))
#     model.add(Dense(noutput, activation=activation_output))
#     return model
#
#
# def test_create_nn_regressor():
#     model = create_nn_regressor(4, 20, 1)
#     model.build()
#     model.summary()
#     model(tf.ones((10, 4)))
#     print(model.weights)
#
#
