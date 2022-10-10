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
sys.path.append("..\\..\\do_mpc\\tools\\onnx")
import onnx_to_casadi

# Specify the Keras model layers
model_input1 = keras.Input(shape=(1,), name='first_input')
model_input2 = keras.Input(shape=(1,), name='second_input')
concat_layer = keras.layers.concatenate([model_input1, model_input2],name="concatination_layer")
hidden_layer = keras.layers.Dense(3, name='hidden_layer', activation='tanh')(concat_layer)
model_output = keras.layers.Dense(1, name='model_output', activation='linear')(hidden_layer)

# Define the Keras model
model = keras.Model(inputs=[model_input1,model_input2], outputs=model_output, name="model")

# Define the CasADi converter
casadi_converter = onnx_to_casadi.ONNX2Casadi(model)

# Show expected input shapes and names
print(casadi_converter.inputshape)

# Define inputs with respective shapes
input1 = casadi.SX.sym("in1",1)
input2 = casadi.SX.sym("in2",1)

# Pass the input to the converter with specified model input names
# If verbose is set to True (default is False) a short message will be shown 
# after every successful conversion of each ONNX computational layer 
casadi_converter.convert(first_input=input1, second_input=input2, verbose=True)

# Show names of all model computational nodes:
# A NN layer can consist of several computation nodes. Each computational node
# describes one specific logical or mathematical operation (logical:
# concatination, mathematical: bias addition, matrix multiplication, ...)
print(casadi_converter.layers)

# All model-equivalent symbolic expressions of each computational node are
# stored as a dictionary in:
casadi_converter.values

# Show model output layer names
print(casadi_converter.output_layers)

# The converter can as well try to guess the original NN layer names from the
# passed ONNX computational graph structure and set corresponding symbolic
# expressions. This can be shown as follows
print(casadi_converter.relevant_layers)
print(casadi_converter.relevant_values)

# Show CasADi symbolic expressions of the output layers only
casadi_converter.output_values

# Get the specific symbolic expression of a specific defined layer (1st example)
# or computational node (2nd example). 
casadi_converter["hidden_layer"]   
            # All NN layer names are stored in casadi_converter.relevant_layers
casadi_converter["model/hidden_layer/MatMul:0"] 
            # All ONNX node names are stored in casadi_converter.layers 

# Predict values using the overall CasADi function equivalent to the original
# model:
single_inp1 = 1
single_inp2 = -5
predicted_single_value = casadi_converter.casadi_function(single_inp1, single_inp2)

# Prediction can as well be carried out for a set of values
input_set1 = np.array([[1,2,5]])
input_set2 = np.array([[-4,2,8]])
predicted_value_set = casadi_converter.casadi_function(input_set1, input_set2)



