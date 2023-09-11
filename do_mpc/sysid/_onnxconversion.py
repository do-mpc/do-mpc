import casadi
import onnx
from onnx import numpy_helper
import numpy as np
import pdb
import importlib
from typing import List, Dict, Tuple, Union, Callable, Any, Optional


class ONNXConversion:
    """ Transform `ONNX model <https://onnx.ai>`_. 
    The transformation returns a CasADi expression of the model and can be used e.g. in the :py:class:`do_mpc.model.Model` class.

    Warning:
        The feature is experimental and currently only has a limited number of supported operations.
        All supported operations can be found in the :py:class:`ONNXOperations` class.

        Other known limitations are listed at the end of this page.
    
    **How to use:** 

    1. Create an ONNX model in your favorite framework (e.g. `TensorFlow <https://www.tensorflow.org/>`_, `PyTorch <https://pytorch.org/>`_, `Keras <https://keras.io/>`_, `ONNX <https://onnx.ai/>`_).
    
    2. Initiate the :py:class:`ONNXConversion` class with the ONNX model as input.

    3. Obtain information about model inputs and ouputs by printing the class instance.

    4. Call the :py:meth:`ONNXConversion.convert` method, passing with keyword arguments the external inputs of the model. The inputs are propagated through the model and all node expressions are created.

    5. Query the class instance with the respective layer or node name to obtain the CasADi expression of the respective layer or node.


    **Example:**

    We start with a simple Tensorflow (with Keras) model:
    ::
    
        model_input = keras.Input(shape=(3), name='input')
        hidden_layer = keras.layers.Dense(5, activation='relu', name='hidden')(model_input)
        output_layer = keras.layers.Dense(1, activation='linear', name='output')(hidden_layer)

        keras_model = keras.Model(inputs=model_input, outputs=output_layer)

    We then proceed to export the model in the ONNX format, using the `tf2onnx <https://pypi.org/project/tf2onnx/>`_ package:

    ::

        model_input_signature = [
            tf.TensorSpec(np.array((1, 3)), name='input'),
        ]
        output_path = os.path.join('models', 'model.onnx')

        onnx_model, _ = tf2onnx.convert.from_keras(keras_model, 
            output_path=output_path, 
            input_signature=model_input_signature
        )

    We can now use the ONNX model (either directly or loaded from disc) to initialize the :py:class:`ONNXConversion` class:

    ::

        casadi_converter = do_mpc.sysid.ONNXConversion(onnx_model)

    Obtain information about the model inputs and outputs by calling ``print(casadi_converter)``, yielding, in this example:

    .. code-block:: console

        ONNX2Casadi model 'casadi_model' 
        ----------------------------------
        Call 'convert' by supplying the inputs with respective name and shape below.
        Input shape of 'input' is (1, 3)
        ----------------------------------
        Query the instance with the following keywords to obtain the CasADi expression of the respective layer or graph operation node:
        - 'input'
        - 'model_4/hidden/MatMul:0'
        - 'model_4/hidden/Relu:0'
        - 'output'

    Call the :py:meth:`ONNXConversion.convert` method, considering the name and shape of the inputs:

    :: 

        # Inputs can be numpy arrays
        casadi_converter.convert(input=np.ones((1,3)))

        # or CasADi expressions
        x = casadi.SX.sym('x',1,3)
        casadi_converter.convert(input=x)

    Query the instance with the respective layer or node name to obtain the CasADi expression of the respective layer or node:

    ::

        print(casadi_converter['output'])

    Args:
        model: An ONNX model.
        model_name: Name of the model

    """
    
    def __init__(self, model: onnx.onnx_ml_pb2.ModelProto, model_name: Optional[str]=None):  
        # In case of a keras model as input, convert it to an ONNX model
        
        if isinstance(model,(onnx.onnx_ml_pb2.ModelProto)):
            self.onnx_model = model
            self.name = "casadi_model" if not isinstance(model_name, (str)) else model_name
        else:
            raise Exception("Please pass a keras or onnx model as input. Please use the from_keras flag to convert a keras model to an onnx model.")
        
        # From the ONNX model the graph and the nodes and the initializers are directly inherited
        self.graph = self.onnx_model.graph
        self.nodes = list(self.graph.node)
        onnx_initializers = list(self.graph.initializer)
        
        # The initialized tensors are converted into the numpy readable format  before assignment
        self.initialized_tensors = {}
        for initializer in onnx_initializers:
            self.initialized_tensors[initializer.name] = numpy_helper.to_array(initializer)
        
            
        # Determining the input shape 
        self.inputshape = {}
        for inpn in self.graph.input:
            if inpn.name not in self.initialized_tensors.keys():
                self.inputshape[inpn.name] = tuple([shape_dim.dim_value for shape_dim in inpn.type.tensor_type.shape.dim])
        
         
        # Determining output layer names
        self.output_layers = [out.name for out in self.graph.output]
        self.layers = [n.name for n in list(self.graph.input)] + [n.output[0] for n in self.nodes]
        

        # Create instance of operations class
        self.operations = ONNXOperations()
        

    def __repr__(self) -> str:
        """ Prints information about the converter.

        Use this method to obtain information about the model inputs and outputs. 
        """

        # Create message
        repr_message = "ONNX2Casadi model '{}' \n".format(self.name)
        repr_message += "----------------------------------\n"
        repr_message += "Call 'convert' by supplying the inputs with respective name and shape below.\n"
        for name, shape in self.inputshape.items():
            repr_message += "Input shape of '{}' is {}\n".format(name, shape)
        repr_message += "----------------------------------\n"
        repr_message += "Query the instance with the following keywords to obtain the CasADi expression of the respective layer or graph operation node:\n"
        for name in self.layers:
            repr_message += " - '{}'\n".format(name)

        return repr_message
        
        

    def _determine_shape(self,raw_shape):
        #TODO: I am not sure we need this. In any case this should be a private method. The user wont activate it.
        """ This method helps to determine the relevant array shape from a given
        ambiguous shape representation.
        
        *Example:*
        
        ::
            
            
        (None,1) and (1, None) as input return (1,)
        (n,m) shapes with n and m not "None" values stays the same
        [n,m] returns (n,m) in a tuple representation
        """
        shape = []
        for dimension in raw_shape:
            #if dimension != None:
            shape.append(dimension)
        return tuple(shape)
    
    
            
    
    def convert(self, verbose=False, **kwargs) -> None:
        """ Evaluate ONNX model with inputs of type ``casadi.SX``, ``casadi.MX``, ``casadi.DM`` or ``numpy.ndarray``.

        The keyword arguments of this method refer to the names of the inputs of the model. 
        If these names are unknown, print the instance of the class to obtain the names.

        Convert does not return anything. The converted model is stored in the instance of the class.
        To obtain the results of the conversion at an arbitrary internal layer, query the instance with the respective layer name.
        Layer names can be obtained by printing the instance of the class.

        Args:
            verbose: If True, prints the conversion progress.
            **kwargs: Keyword arguments of the method refer to the names of the inputs of the model. The values of the keyword arguments are the inputs of the model and can be of type ``casadi.SX``, ``casadi.MX``, ``casadi.DM`` or ``numpy.ndarray``.
        """
        
        
        # Rename for shorther notation
        graph = self.graph
        nodes = self.nodes
        init_tensors = self.initialized_tensors
        inputshape = self.inputshape
        
            
        # Sanity check: Right number of inputs?
        if len(kwargs) != len(inputshape):
            raise Exception("The model takes {} inputs for the layers {}".format(len(inputshape.keys()),list(inputshape.keys())))

        # Sanity check: Right names for inputs
        if not all( layer_name in list(kwargs.keys()) for layer_name in list(inputshape.keys()) ):
            raise Exception("False input layer names.\n The input layers are {}".format(list(inputshape.keys())))

        # Sanity check: Right type for inputs
        if not all(isinstance(value,(casadi.SX,casadi.MX, casadi.DM, np.ndarray)) for value in kwargs.values()):
            raise Exception("Wrong input type. Please pass a CasADi variable or numpy array as input.")

        # Create dict "input" and check the shape of the inputs
        self.input = {}   

        for input_name, shape in inputshape.items():
            # TODO: Write comments on all checks
            check_1 = (len(inputshape[input_name]) == 1)
            check_2 = (inputshape[input_name][0] != kwargs[input_name].shape[0])
            check_3 = (inputshape[input_name] != kwargs[input_name].shape)
            if check_1 and check_2 and check_3:
                raise Exception("The shape of the input '{}' should be {}".format(input_name,inputshape[input_name]))

            self.input[input_name] = kwargs[input_name]



        # Computation of all node values
        node_values = self.input.copy() # "node_values" contains only input node values as initial values
        all_values = self.input.copy()
        all_values.update(init_tensors) # "all_values" contains initializer and input node values as initial values
        
        # Iterate over all nodes
        for n in nodes:
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
                
                # Conversion into CasADi and shape correction in case of (1,) as input shape
                if isinstance(all_values[input_layer_name],(np.ndarray)):
                    pass
                    #all_values[input_layer_name] = casadi.DM(np.atleast_2d(all_values[input_layer_name]))

                ins.append(all_values[input_layer_name]) # critical ! input_layer_name should be already contained in all_values => Assumption: ONNX graph representation is correctly arranged
                
            
            # Determination of the operation type and subsequent computation
            if hasattr(self.operations, n.op_type):
                out = getattr(self.operations, n.op_type)(*ins, attribute=n.attribute)
            else:
                raise Exception("Operation '{}' not implemented. Please consider the limited set of operations available to the tool.".format(n.op_type))
            

            all_values[n.output[0]] = out
            node_values[n.output[0]] = out
        
        # Assignment of all node output values to the class object 
        self.node_values = node_values
            
        

    def __getitem__(self, key: str):
        """ Enables the output of the CasADi expression of a specific layer or 
        graph operation node. 

        To learn about possible keywords, it is recommended to print the instance of the class:

        ::

            print(converter)

        Args:
            key: Name of the layer of the ONNX graph.


        """
        node_values = self.node_values

        if key in node_values.keys():
            out = node_values[key]
        else:
            raise Exception("The node '{}' is not contained in the ONNX graph.".format(key))

        return out



class ONNXOperations:
    """ CasADi operations, which are available in the :py:class:`ONNXConversion` class.
    See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md>`_ for a full list of operations.

    .. note::

        This class is not intended to be used directly. It is used by the :py:class:`ONNXConversion` class.
        The purpose of this class is to provide a list of all available operations in the :py:class:`ONNXConversion` class.

    """
    def __init__(self):
        pass

    def Tanh(self,x, attribute = None):
        return casadi.tanh(x)

    def Sigmoid(self,x, attribute = None):
        out = 1/(1+casadi.exp(-x))
        return out

    def Relu(self,x, attribute = None):
        return casadi.fmax(0,x)

    def Elu(self,x, attribute = None):
        return casadi.fmax(0,x) + casadi.fmin(0,casadi.exp(x)-1)

    def MatMul(self,*args, attribute = None):
        return casadi.mtimes(*args)

    def Add(self,*args, attribute = None):
        """Addition of two or more tensors.
        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#add>`_ for more details.
        """
        out = 0
        for arg in args:
            out += arg
        return out

    def Mul(self,*args, attribute = None):
        return args[0]*args[1]

    def Sub(self, *args, attribute = None):
        return args[0] - args[1]

    def Gemm(self, *args, attribute = None):
        """General Matrix Multiplication.
        See `ONNX documentation  <https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm>`_ for more details.
        """

        attr_dict = {
            k.name: k.i if k.type == 2 else k.f for k in attribute
        }

        A = args[0]
        B = args[1]
        C = args[2]

        if 'transA' in attr_dict.keys() and attr_dict['transA'] == 1:
            A = casadi.transpose(A)
        if 'transB' in attr_dict.keys() and attr_dict['transB'] == 1:
            B = casadi.transpose(B)
        if 'alpha' in attr_dict.keys():
            alpha = attr_dict['alpha']
        else:
            alpha = 1
        if 'beta' in attr_dict.keys():
            beta = attr_dict['beta']
        else:
            beta = 1

        if C.ndim == 1:
            C = C.reshape(1,-1)
        
        res = alpha*self.MatMul(A,B) + beta*C

        return res

    def Sum(self,*args, attribute = None):
        return  self.Add(*args)

    def Concat(self,*args, attribute = None):
        if attribute[0].i in (0,2):
            return casadi.vertcat(*args)
        else:
            return casadi.horzcat(*args)
    
    def Unsqueeze(self,*args, attribute = None):
        # TODO: Check if this is correct
        return args[0]

    def Squeeze(self, *args, attribute = None):
        # TODO: Check if this is correct
        return args[0]

    def Slice(self, *args, attribute = None):
        # TODO: Check if this is correct
        ax = attribute[0].ints
        ends = attribute[1].ints
        starts = attribute[2].ints

        slices = [slice(None,None),]*2

        for ax_k, start_k, end_k in zip(ax, starts, ends):
            slices[ax_k] = slice(start_k,end_k)

        return args[0][tuple(slices)]

    def Reshape(self, *args, attribute = None):
        # TODO: Check if this is correct
        data = args[0]
        shape = args[1]
        return data.reshape(tuple(shape))

    def Shape(self, *args, attribute = None):
        return args[0].shape


    

