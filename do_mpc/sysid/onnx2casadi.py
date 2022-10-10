import casadi
import onnx
import numpy as np
import tensorflow as tf
import pdb


class ONNX2Casadi:
    """ Transform ONNX model into casadi mathematical symbolic expressions.
    
    The ONNX2Casadi class is only then operative, if the model to be transformed
    does not include numerical arrays of a 3rd or higher order: Inputs and all
    computation results are either scalars or at most vectors and no matrices 
    (convolutional networks can not be converted).
    
    The __init__ method defines and initializes the object properties, which are 
    relevant for the following CasADi-conversion steps. For better understanding
    of the open-source software tool CasADi please visit https://web.casadi.org/
    
    The model is only converted by applying the method "convert".
    
    To better understand how this tool works, see
    https://github.com/do-mpc/do-mpc/tree/master/examples/onnx_conversion/onnx_to_casadi_conversion.py
            
            
    **Warning**
    
    1) If you are getting the following error while running the script or
       installing the corresponding python package:
           - Powershell error concerning version conflict of the FlatBuffers
       try to upgrade your tensorflow package using the following poweshell command:
           pip install tensorflow --upgrade
           
    2) While passing a keras model to the class ONNX2Casadi a very long comment
    will be generated and printed in the python console. This is due to the function
    tf2onnx.convert.from_keras() and is fully normal and expected.
    """
    
    def __init__(self, model, model_name=None):
        """ Initializes the converted model.
        Pass either a keras or ONNX model.
        """
        
        # In case of a keras model as input, convert it to an ONNX model
        if isinstance(model,(tf.keras.Model)):
            try:
                import tf2onnx
            except:
                raise Exception("The package 'tf2onnx' is not installed. Please install it, e.g. using 'pip install tf2onnx'.")

            model_input_signature = [tf.TensorSpec(np.array(self.determine_shape(inp_spec.shape)),
                                        name=inp_spec.name) for inp_spec in model.input_spec]
            self.onnx_model, _ = tf2onnx.convert.from_keras(model,output_path=None,
                                                            input_signature=model_input_signature)
            self.name = "casadi_model" if not isinstance(model_name, (str)) else model.name
        
        elif isinstance(model,(onnx.onnx_ml_pb2.ModelProto)):
            self.onnx_model = model
            self.name = "casadi_model" if not isinstance(model_name, (str)) else model_name
        
        else:
            raise Exception("Please pass a keras or onnx model as input")
            
        
        
        # From the ONNX model the graph and the nodes and the initializers are directly inherited
        self.graph = self.onnx_model.graph
        self.nodes = list(self.graph.node)
        onnx_initializers = list(self.graph.initializer)
        
        # The initialized tensors are converted into the numpy readable format  before assignment
        self.initialized_tensors = {}
        for initializer in onnx_initializers:
            self.initialized_tensors[initializer.name] = onnx.numpy_helper.to_array(initializer)
        
            
        # Determining the input shape 
        self.inputshape = {}
        for inpn in self.graph.input:
            if inpn.name not in self.initialized_tensors.keys():
                self.inputshape[inpn.name] = tuple([shape_dim.dim_value for shape_dim in inpn.type.tensor_type.shape.dim])
        
         
        # Determining output layer names
        self.output_layers = [out.name for out in self.graph.output]
        self.layers = [n.name for n in list(self.graph.input)] + [n.output[0] for n in self.nodes]
        

        # Create instance of operations class
        self.operations = _Operations()
        

    def __repr__(self):
        """ Prints information about the converter.
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
        
        

    def determine_shape(self,raw_shape):
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
            if dimension != None:
                shape.append(dimension)
        return tuple(shape)
    
    
            
    
    def convert(self, verbose=False, **kwargs):
        """ Convert ONNX model to CasADi model.
        
        Args:
            verbose: if True (default value), a message will be printed after
                each successful conversion of a ONNX graph operation.
            symvar_type: Chose between the casadi-specific "SX" or "MX" symbolic
                variables 
            kwargs: Input values as CasADi variables with specified input 
                variable name
                
        Returns:
            casadi_model: A CasADi symbolic function which is mathematically 
                equivalent to the model operation.
            output_values: The symbolic expression of the model output with respect
                to the symbolic input variables x, y and z.
            self.values: A dictionary with the names of all included ONNX-specific
                computation operations as keys and their CasADi symbolic expressions
                with respect to the symbolic input variables x,y and z. 
        """
        
        
        # Rename for shorther notation
        graph = self.graph
        nodes = self.nodes
        init_tensors = self.initialized_tensors
        inputshape = self.inputshape
        
            
        # Sanity check: Are the dimensions of kwargs correct. Do we have the right names and number of inputs?
        self.input = {}
        if len(kwargs) == len(inputshape):
            if not all( layer_name in list(kwargs.keys()) for layer_name in list(inputshape.keys()) ):
                raise Exception("False input layer names.\n The input layers are {}".format(list(inputshape.keys())))
                
            for input_name, shape in inputshape.items():
                if isinstance(kwargs[input_name],(np.ndarray)):
                    if kwargs[input_name].shape != inputshape[input_name]:
                        raise Exception("The shape of the input '{}' should be {}".format(input_name,inputshape[input_name]))
                
                elif isinstance(kwargs[input_name],(casadi.SX,casadi.MX)):
                    if ( (len(inputshape[input_name]) == 1) and (
                            inputshape[input_name][0] != kwargs[input_name].shape[0])) and (
                                inputshape[input_name] != kwargs[input_name].shape):
                        raise Exception("The shape of the input '{}' should be {}".format(input_name,inputshape[input_name]))
                    
                else:
                    raise Exception("Input should be of the datatype numpy.ndarray, casadi.casadi.MX or casadi.casadi.SX")
                
                self.input[input_name] = kwargs[input_name]
            
        else:
            raise Exception("The model takes {} inputs for the layers {}".format(len(inputshape.keys()),list(inputshape.keys())))
            

        # Sanity check: Are all the input variables of the same CasADi symbolic 
        for inp in self.input.values():
            if not isinstance(inp,(casadi.SX,casadi.MX,np.ndarray)):
                raise Exception("Input should be of the datatype numpy.ndarray, casadi.casadi.MX or casadi.casadi.SX")


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
                    all_values[input_layer_name] = casadi.DM(np.atleast_2d(all_values[input_layer_name]))

                ins.append(all_values[input_layer_name]) # critical ! input_layer_name should be already contained in all_values => Assumption: ONNX graph representation is correctly arranged
                
            
            # Determination of the operation type and subsequent computation
            out = getattr(self.operations, n.op_type)(*ins, attribute=n.attribute)
            

            all_values[n.output[0]] = out
            node_values[n.output[0]] = out
        
        # Assignment of all node output values to the class object 
        self.node_values = node_values
            
        

    def __getitem__(self, key):
        """ Enables the output of the CasADi expression of a specific layer or 
        graph operation node.
        """
        node_values = self.node_values

        if key in node_values.keys():
            out = node_values[key]
        else:
            raise Exception("The node '{}' is not contained in the ONNX graph.".format(key))

        return out



class _Operations:
    """ Class for the definition of the CasADi operations, which are used in the
    ONNX2CasADi class.

    Method names are the same as the ONNX operation names.
    """
    def __init__(self):
        pass

    def Tanh(self,x, attribute = None):
        return casadi.tanh(x)

    def Sigmoid(self,x, attribute = None):
        return casadi.sigmoid(x)

    def Relu(self,x, attribute = None):
        return casadi.fmax(0,x)

    def Elu(self,x, attribute = None):
        return casadi.fmax(0,x) + casadi.fmin(0,casadi.exp(x)-1)

    def MatMul(self,*args, attribute = None):
        return casadi.mtimes(*args)

    def Add(self,*args, attribute = None):
        return  casadi.sum(args)

    def Sum(self,*args, attribute = None):
        return  self.Add(*args)

    def Concat(self,*args, attribute = None):
        if attribute[0].i == 0:
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


    

