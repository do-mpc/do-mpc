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
sys.path.append(os.path.join('..','..','..'))

import do_mpc

sys.path.append(os.path.join('..','..','..','do_mpc','sysid'))

import onnxconversion
import importlib
importlib.reload(onnxconversion)

import pdb

# Specify the Keras model layers
model_input1 = keras.Input(shape=(3), name='first_input')
model_input2 = keras.Input(shape=(2), name='second_input')
concat_layer = keras.layers.concatenate([model_input1, model_input2],name="concatenation_layer", axis=1)
hidden_layer = keras.layers.Dense(5, name='hidden_layer', activation='tanh')(concat_layer)
hidden_layer2 = keras.layers.Dense(5, name='hidden_layer2', activation='relu')(hidden_layer)
sum_layer = keras.layers.add([hidden_layer, hidden_layer2], name="sum_layer")
slice_layer = keras.layers.Lambda(lambda x: x[:, :2], name="slice_layer")(sum_layer)
model_output = keras.layers.Dense(5, name='model_output', activation='tanh')(slice_layer)


# Define the Keras model
model = keras.Model(inputs=[model_input1,model_input2], outputs=model_output, name="model")

# Define the CasADi converter
casadi_converter = onnxconversion.ONNXConversion(model, from_keras=True)

# Print information about converter
# - Input shapes and names
# - Layer names (to query with __getitem__)
print(casadi_converter)


# All tests in the following with these values:
test_inp1 = np.array([1.0, 2.0, 3.0]).reshape(1,-1)
test_inp2 = np.array([2.0, 2.0]).reshape(1,-1)

""" First test with coasadi arrays as input """
# Define inputs with respective shapes. Test either with numpy or casadi or mixed
input1 = casadi.SX.sym("in1",(1,3))
input2 = casadi.SX.sym("in2",(1,2))

casadi_converter.convert(first_input=input1, second_input=input2, verbose=False)

# Pass the input to the converter with specified model input names
# If verbose is set to True (default is False) a short message will be shown 
# after every successful conversion of each ONNX computational layer 
casadi_converter.convert(first_input=input1, second_input=input2, verbose=False)
# Get the specific symbolic expression of a specific defined layer (1st example)
# or computational node (2nd example). 
casadi_output = casadi_converter["model_output"]

# Create function from output
casadi_function = casadi.Function("casadi_function", [input1, input2], [casadi_output])

# Evaluate the function
print("Casadi output: ", casadi_function(test_inp1, test_inp2))


""" Second test with numpy arrays as input """

input1 = test_inp1
input2 = test_inp2

casadi_converter.convert(first_input=input1, second_input=input2, verbose=False)
numpy_output = casadi_converter["model_output"]
print("Numpy output: ", numpy_output)

""" Third test with mixed input """
input1 = casadi.SX.sym("in1",(1,3))
input2 = test_inp2

casadi_converter.convert(first_input=input1, second_input=input2, verbose=False)
mixed_output = casadi_converter["model_output"]
casadi_function = casadi.Function("casadi_function", [input1], [mixed_output])
print("Casadi output (with one numpy input): ", casadi_function(test_inp1))



print("Keras output: ", model.predict([test_inp1, test_inp2]))

