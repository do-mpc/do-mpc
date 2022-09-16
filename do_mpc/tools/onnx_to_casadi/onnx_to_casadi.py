import numpy as np
import pandas as pd
import tensorflow as tf
import numpy as np
import random
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import sklearn
from keras.layers import Input, Dense
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
import casadi
from tensorflow.python.framework import tensor_util
from onnx_tf.backend import prepare
import onnx2keras
from onnx2keras import onnx_to_keras
import keras
import onnxruntime as rt
import onnx
import tf2onnx
from casadi import *


class ONNX2Casadi:
    """ Transform ONNX model into casadi mathematical symbolic expressions.
    
    The ONNX2Casadi class is only then operative, if the model to be transformed
    does not include numerical arrays of a 3rd or higher order: Inputs and all
    computation results are either skalars or at most vectors and no matrices 
    (convolutional networks can not be converted).
    
    The __init__ method defines and initializes the object prperties, which are 
    relevant for the following CasADi-conversion steps. for better understanding
    of the open-source CasADi please visit https://web.casadi.org/
    
    The conversion of the model operations into CasADi expressions happens
    only through the __call__ method.
    
    **Example:**
    
    ::
        
        onnx_model = onnx.load("path_to_the_onnx_model.onnx")
        
    import the ONNX model (As this script is written in python,
    the __init__ method can alternatively get a keras model as input,
    which will be converted into its ONNX equivalent)
    
    ::
        
        converted_model = ONNX2Casadi(onnx_model)
    
    This initializes the conversion-relevant properties of the ONNX2Casadi-
    object: converted_model
    
    ::
        
        x = SX.sym('x', 1)
        y = SX.sym('y')
        z = SX.sym('z')
        conversion_results = initialized_converted_model(x=x,y=y,z=z)
        
    Initialize the inputs as CasADi symbolic variables and pass them respectively
    to the model inputs specified with corresponding input name. This will result
    in the list: conversion_results containing:
        - At index 0: a CasADi symbolic function which is mathnatically 
        equivalent to the model operation.
        - At index 1: the symbolic expression of the model output with respect
        to the symbolic input variables x, y and z.
        - At index 2: a dictionary with the names of all included ONNX-specific
        computation operations as keys and their CasADi symbolic expressions
        with respect to the symbolic input variables x,y and z.
        
    ::
        
        layer-specific_casadi_expression = converted_model['layer_name']
        
    This is how the CasADi conversion result of a specific layer or ONNX graph
    node can be accessed. Therefor the exact layer or graph node name is needed.
    To get all graph node names type:
        print(converted_model.layers)
    To get all layer names type:
        print(converted_model.relevant_layers))
    graph nodes are the computation steps as they are stored in the ONNX graph.
    Layer is the conventional layer concept from a neural network. Each layer 
    can be represented as several computational graph nodes in ONNX: One fully
    connected layer usually contain 3 ONNX graph nodes:
            - Weight multiplication
            - Bias addition
            - Activation function
            
            
    **Warning**
    
    1) If you are getting one of the following errors while running the script or
       installing the corresponding python package:
           - Powershell error concerning version conflict of the FlatBuffers
           - Python console error: "from_keras requires input_signature"
       try to upgrade your tensorflow package using the following poweshell command:
           pip install tensorflow --upgrade
           
    2) While passing a keras model to the class ONNX2Casadi a very long comment
    will be generated and printed in the python console. This is due to the function
    tf2onnx.convert.from_keras() and is fully normal and expected.
    """
    
    def __init__(self, model):
        """ Initializes the converted model.
        Pass eitehr a keras or ONNX model.
        """
        
        # In case of a keras model as input. convert it to an ONNX model
        if "keras" in str(type(model)).lower():
            self.onnx_model, _ = tf2onnx.convert.from_keras(model,opset=13,output_path="{}.onnx".format(model.name))
            self.name = model.name
        else:
            self.onnx_model = model
            self.name = "casadi_model"
            
        
        
        # From the ONNX model the graph and the nodes and the initializers are directly inheretited
        self.graph = self.onnx_model.graph
        self.nodes = list(self.graph.node)
        onnx_initializers = list(self.graph.initializer)
        
        # The intialized tensors are converted into the numpy readable format  befor assignment
        self.initialized_tensors = {}
        for initializer in onnx_initializers:
            self.initialized_tensors[initializer.name] = onnx.numpy_helper.to_array(initializer)
        
            
        # Determining the input shape 
        inputshape = {}
        for inpn in self.graph.input:
            if inpn.name not in self.initialized_tensors.keys():
                inputshape[inpn.name] = tuple([shape_dim.dim_value for shape_dim in inpn.type.tensor_type.shape.dim[1:]])
        
        self.inputshape = inputshape
        
         
        # Determining output layer names
        
        self.output_layers = [out.name for out in self.graph.output]
        all_layers = [n.name for n in list(self.graph.input)] + [n.output[0] for n in self.nodes]
        self.layers = all_layers
        
        # Initializing relevant layers
        self.relevant_layers = []
        self.relevant_values = {}

        


    


    
    def __call__(self, casadi_model_name="", verbose=True, symvar_type="SX", **kwargs):
        """ This method geneartes the actual casadi conversion.
        
        Args:
            casadi_model_name: The default name is "casadi_model" or the 
                original keras model name, if a keras model was passed.
                to __init__
            verbose: if True (default value), a message will be printed after
                each successful conversion of a ONNX graph operation.
            symvar_type: Chose between the casadi specific "SX" or "MX" symbolics
            kwargs: Input values as CasADi variables with specified input 
                variable name
                
        Returns:
            casadi_model: A CasADi symbolic function which is mathnatically 
                equivalent to the model operation.
            output_values: The symbolic expression of the model output with respect
                to the symbolic input variables x, y and z.
            self.values: A dictionary with the names of all included ONNX-specific
                computation operations as keys and their CasADi symbolic expressions
                with respect to the symbolic input variables x,y and z. 
        """
        
        
        # It is possible to rename the model
        if len(casadi_model_name) > 0:
             self.name = CasADi_model_name
                
             
        # Renaming the local variables for the purpose of better code readability
        graph = self.graph
        nodes = self.nodes
        init_tensors = self.initialized_tensors
        inputshape = self.inputshape
        # all_input_layers = self.all_input_layers
        output_layers = self.output_layers
        all_layers = self.layers 
        
        # Defining the CasADi symbolic variable type (SX or MX) 
        if symvar_type == "SX":
            symvar_type = casadi.SX
        elif symvar_type == "MX":
            symvar_type == casadi.MX
        
        
        # Sanity check: Are the dimensions of kwargs correct. Do we have the right names and number of inputs?
        
        self.input = {}
        if len(kwargs) == len(inputshape):
            if not all( layer_name in list(kwargs.keys()) for layer_name in list(inputshape.keys()) ):
                raise Exception("False input layer names.\nThe input layers are {}".format(list(inputshape.keys())))
                
            for input_name, shape in inputshape.items():
                if isinstance(kwargs[input_name],(numpy.ndarray)):
                    if kwargs[input_name].shape != inputshape[input_name]:
                        raise Exception("The shape of the input '{}' should be {}".foramt(input_element,inputshape[input_name]))
                
                elif isinstance(kwargs[input_name],(casadi.SX,casadi.MX)):
                    if ( (len(inputshape[input_name]) == 1) and (
                            inputshape[input_name][0] != kwargs[input_name].shape[0])) and (
                                inputshape[input_name] != kwargs[input_name].shape):
                        raise Exception("The shape of the input '{}' should be {}".foramt(input_element,inputshape[input_name]))
                    
                else:
                    raise Exception("Input should be of the datatype numpy.ndarray, casadi.casadi.MX or casadi.casadi.SX")
                
                self.input[input_name] = kwargs[input_name]
            
        else:
            raise Exception("The model takes {} inputs for the layers {}".format(len(inputshape.keys()),list(inputshape.keys())))
            
            
            
        # Defining activation functions (should be extended with, if the used ActFunc is not available here !)
        var = MX.sym("var",1)
        act_func_dict = {"Tanh":np.tanh,"Sigmoid":Function("Sigmoid",[var],[1/(1+np.exp(-var))]),
                     "Relu":Function("relu",[var],[(var>=0)*var]),
                     "Elu":Function("elu",[var],[var*(var>=0)+(np.exp(var)-1)*(var<=0)])}



        # Computation of all node values
        node_values = self.input.copy() # "node_values" contains only input node values as initial values
        all_values = self.input.copy()
        all_values.update(init_tensors) # "all_values" contains initializer and input node values as initial values
        for n in nodes[0:]:
            if verbose:
                print("\nProcessing of {}".format(n.name))
                
            ins = [] # "ins" collects all the input variables for the corresponding
                     # node in CasADi-form with corrected shape (in case input shape is (1,))
                     # These inputs could either be the input values from the last layer
                     # or the bias and weight values, which are saved in "init_tensors"
                     # "and all_values"
                     # Computation follows the node order given by the ONNX graph.
                     # Computed values from the previous node are initialized in "node_values"
                     # These are as well saved in "all_values" in addition to bias and weight values
                     # A computational node is different from a neural layer:
                     # ONNX graph reserves for each mathematical operation a separate
                     # node (bias addition and weight multiplication are 2 nodes) 
                     
            for input_layer_name in n.input:
                
                # conversion into CasADi and shape correction in case of (1,) as input shape
                if isinstance(all_values[input_layer_name],(numpy.ndarray)):
                    if len(all_values[input_layer_name].shape) == 1:
                        all_values[input_layer_name] = np.array([all_values[input_layer_name]])
                    all_values[input_layer_name] = symvar_type(all_values[input_layer_name])
                ins.append(all_values[input_layer_name]) # critical !!! input_layer_name should be already contained in all_values => Assumption: ONNX graph representation is correctly arrenged
                
            
            ## Determination of the operation type and subsequent computation
            
            # input-weight-multiplication 
            if n.op_type == "MatMul":
                out = ins[0]@ins[1]
                # out = ins[1]@ins[0]
                
            # Addition of bias to weight multiplication results from previous node
            elif n.op_type in ["Add","Sum"]:
                out = sum(ins)
                
            # Application of activation function
            elif n.op_type in list(act_func_dict.keys()):
                out = act_func_dict[n.op_type](ins[0])
                
            # Node result concatination (equivalennt to layer concatination)
            elif n.op_type == "Concat":
                if n.attribute[0].i == 0:
                    concat_f = vertcat
                else:
                    concat_f = horzcat
                out = concat_f(ins[0],ins[1])
            
            # The model in its actual actual capable of the conversion of other
            # ONNX node types and operations. This design should be sufficient
            # for the deep learning regression applications with at maximum
            # 2-dimensional arrays as input (no convulotional neural networks or
            # models with 3D and higher input arrays !)
            else:
                raise Exception("The layer type: {} is not yet included in the conversion tool".format(n.op_type))
            if isinstance(out,(numpy.ndarray)):
                out = symvar_type(out)
            
            # Assignement of the computation result to the variables "all_values"
            # and "node_values" as initialization for the next node operation
            all_values[n.output[0]] = out
            node_values[n.output[0]] = out
        
        # Assignement of all node output values to the class object 
        self.values = node_values
            
          
            
        # computation values of relevant nodes only
        # Relevant nodes are nodes, whose output results correspond to the
        # layer output results: node names of bias addition and weight 
        # multiplication operations are eliminated. hence, as layer results are
        # only results from concatination nodes or activation function nodes considered
        self.relevant_layers = []
        self.relevant_values = {}
        for layer, value in node_values.items():
            operation_type = layer[layer.rfind("/")+1:].lower() # Example: "MatMul:0" from the full node 
                                                                # name "model_1/1st_output/MatMul:0"
            if not any(irrelevant in operation_type for irrelevant in ["matmul","biasadd"]):
                relevant_layer_name = layer
                if layer.count("/")>1:
                    # Example: "model_1" from the full layer name "model_1/xy/Tanh:0"
                    relevant_layer_name = layer[layer.find("/")+1:layer.rfind("/")]
                
                # Assignement of the list of the relevant layer names and of their
                # respective values to the ONNX2CasADi object
                self.relevant_layers.append(relevant_layer_name)                
                self.relevant_values[relevant_layer_name] = self.values[layer]
                
     
        # Generation of the equivalent CasADi model (of the Outputs)
        output_values = []
        for l in output_layers:
            output_values.append(all_values[l])

        casadi_model = Function(self.name,list(self.input.values()),
                    output_values,list(self.input.keys()),output_layers)
        
        print("\nThe model '{}' was successfully converted".format(self.name))
        
        return casadi_model, output_values, self.values







    def __getitem__(self, key):
        """ Enables the output of the casadi expression of a specific layer or 
        graph operation node.
        """
        values = self.values
        values.update(self.relevant_values)
        out = {}
        for layer_name in values:
            if key == layer_name:                
                out[layer_name] = values[layer_name]

        return out
    
    
    
    


if __name__ == '__main__':
    
    # Keras model conversion example
    keras_model = tf.keras.models.load_model("keras_model_multioutput")
    model1 = ONNX2Casadi(keras_model)
    x = SX.sym('x', 1)
    y = SX.sym('y')
    xy_input = SX.sym('xy_input')
    cas_expr1 = model1(x=x,y=y,xy_input=xy_input)
    
    # ONNX model conversion example. The model was generated in Matlab.
    matlab_onnx = onnx.load("matlab onnx models/dlnet1.onnx")
    model2 = ONNX2Casadi(matlab_onnx)
    input2 = {'sequence1': SX.sym('s1', 1),
              'sequence2': SX.sym('s2', 1)}
    cas_expr2 = model2(**input2)
    


