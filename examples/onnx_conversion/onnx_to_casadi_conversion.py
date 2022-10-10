# In this script, a simple usage example of the class "ONNX2Casadi" in the python
# script "onnx_to_casadi.py" in the folder:
# "C:\Users\user\Documents\GitHub\do-mpc\do_mpc\tools\onnx"

# In this script, a simple usage example of the class "ONNX2Casadi" in the python
# script "onnx_to_casadi.py" in the folder:
# "C:\Users\user\Documents\GitHub\do-mpc\do_mpc\tools\onnx"

# This conversion example is carried out using a simple keras model, which is
# built as follows. The "ONNX2Casadi" class accepts, however ONNX models as well
# as input. These should be load using the following commands:
#
#    import onnx
#    onnx.load(path_to_saved_onnx_model.onnx)


from tensorflow import keras
import casadi
import numpy as np

# import the python script from the corresponding folder
import sys
sys.path.append('../../')
sys.path.append('../../do_mpc/sysid')
import do_mpc

import onnx2casadi
import importlib
importlib.reload(onnx2casadi)

import pdb


# Specify the Keras model layers
model_input1 = keras.Input(shape=(1,), name='first_input')
model_input2 = keras.Input(shape=(1,), name='second_input')
concat_layer = keras.layers.concatenate([model_input1, model_input2],name="concatination_layer")
hidden_layer = keras.layers.Dense(3, name='hidden_layer', activation='tanh')(concat_layer)
sum_layer = keras.layers.add([hidden_layer, model_input1], name="sum_layer")
hidden_layer2 = keras.layers.Dense(5, name='hidden_layer2', activation='relu')(hidden_layer)
model_output = keras.layers.Lambda(lambda x: x[0, :2], name="model_output")(hidden_layer2)


# Define the Keras model
model = keras.Model(inputs=[model_input1,model_input2], outputs=model_output, name="model")

# Define the CasADi converter
casadi_converter = onnx2casadi.ONNX2Casadi(model)

# Print information about converter
# - Input shapes and names
# - Layer names (to query with __getitem__)
print(casadi_converter)

# Define inputs with respective shapes
input1 = casadi.SX.sym("in1",1)
input2 = casadi.SX.sym("in2",1)

# input1 = np.array([1.0])
# input2 = np.array([2.0])

# Pass the input to the converter with specified model input names
# If verbose is set to True (default is False) a short message will be shown 
# after every successful conversion of each ONNX computational layer 
casadi_converter.convert(first_input=input1, second_input=input2, verbose=False)


# Get the specific symbolic expression of a specific defined layer (1st example)
# or computational node (2nd example). 
casadi_output = casadi_converter["model_output"]   


# Create function from output
casadi_function = casadi.Function("casadi_function", [input1, input2], [casadi_output])

# Evaluate the function and compare with NN:
test_inp1 = np.array([1.0])
test_inp2 = np.array([2.0])

print("Casadi output: ", casadi_function(test_inp1, test_inp2))
print("Keras output: ", model.predict([test_inp1, test_inp2]))

