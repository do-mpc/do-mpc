
from tensorflow import keras
import casadi
import numpy as np
import os

# import the python script from the corresponding folder
import sys
sys.path.append(os.path.join('..','..','..'))


import do_mpc

sys.path.append(os.path.join('..','..','..','do_mpc','sysid'))

import onnxconversion
import importlib
importlib.reload(onnxconversion)

model_input = keras.Input(shape=(3), name='input')
hidden_layer = keras.layers.Dense(5, activation='relu', name='hidden')(model_input)
output_layer = keras.layers.Dense(1, activation='linear', name='output')(hidden_layer)

keras_model = keras.Model(inputs=model_input, outputs=output_layer)

converter = onnxconversion.ONNXConversion(keras_model, from_keras=True)

converter.convert(input=np.ones((1,3)))

print(converter['output'])

x = casadi.SX.sym('x', 1,3)
converter.convert(input=x)

print(converter['output'])