import casadi
import onnx
import tf2onnx
import numpy as np
import tensorflow as tf


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
    
    def __init__(self, model, model_name=-1):
        """ Initializes the converted model.
        Pass either a keras or ONNX model.
        """
        
        # In case of a keras model as input, convert it to an ONNX model
        if isinstance(model,(tf.keras.Model)):
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
        inputshape = {}
        for inpn in self.graph.input:
            if inpn.name not in self.initialized_tensors.keys():
                inputshape[inpn.name] = tuple([shape_dim.dim_value for shape_dim in inpn.type.tensor_type.shape.dim])
        
        self._inputshape = inputshape
        
         
        # Determining output layer names
        
        self.output_layers = [out.name for out in self.graph.output]
        all_layers = [n.name for n in list(self.graph.input)] + [n.output[0] for n in self.nodes]
        self.layers = all_layers
        
        # Initializing relevant layers
        self.relevant_layers = []
        self.relevant_values = {}

        

    @property
    def inputshape(self):
        return self._inputshape
    
    @inputshape.setter
    def inputshape(self,inpshape):
        raise Exception("The inputshape property can not be changed. It will be automatically generated while passing an input model")
        
    
    

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
        """ This method generates the actual casadi conversion.
        
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
        
        
             
        # For the sake of better code readability, the variables are first renamed
        graph = self.graph
        nodes = self.nodes
        init_tensors = self.initialized_tensors
        inputshape = self._inputshape
        # all_input_layers = self.all_input_layers
        output_layers = self.output_layers
        all_layers = self.layers 
        
            
        # Sanity check: Are the dimensions of kwargs correct. Do we have the right names and number of inputs?
        
        self.input = {}
        if len(kwargs) == len(inputshape):
            if not all( layer_name in list(kwargs.keys()) for layer_name in list(inputshape.keys()) ):
                raise Exception("False input layer names.\nThe input layers are {}".format(list(inputshape.keys())))
                
            for input_name, shape in inputshape.items():
                if isinstance(kwargs[input_name],(np.ndarray)):
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
            

        # Sanity check: Are all the input variables of the same CasADi symbolic 
        # type: SX or MX:
        global input_types
        input_types = [type(inp) for inp in self.input.values()]

        if (casadi.casadi.SX in input_types) and (casadi.casadi.MX in input_types):
            raise Exception("Please use only one casadi symbolic type: SX or MX for all inputs")
        elif casadi.casadi.SX in input_types:
            symvar_type = casadi.SX
            self.symvar_type = casadi.casadi.SX
        elif casadi.casadi.MX in input_types:
            symvar_type = casadi.MX
            self.symvar_type = casadi.casadi.MX
            
            
        
            
            
            
        # Defining activation functions (should be extended, if a required
        # activation function is not available here !)
        var = self.symvar_type.sym("var",1)
        act_func_dict = {"Tanh":np.tanh,"Sigmoid":casadi.Function("Sigmoid",[var],[1/(1+np.exp(-var))]),
                     "Relu":casadi.Function("relu",[var],[(var>=0)*var]),
                     "Elu":casadi.Function("elu",[var],[var*(var>=0)+(np.exp(var)-1)*(var<=0)])}



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
                
                # Conversion into CasADi and shape correction in case of (1,) as input shape
                if isinstance(all_values[input_layer_name],(np.ndarray)):
                    if len(all_values[input_layer_name].shape) == 1:
                        all_values[input_layer_name] = np.array([all_values[input_layer_name]])
                    all_values[input_layer_name] = symvar_type(all_values[input_layer_name])
                ins.append(all_values[input_layer_name]) # critical ! input_layer_name should be already contained in all_values => Assumption: ONNX graph representation is correctly arranged
                
            
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
                
            elif n.op_type == "Unsqueeze":
                out = ins[0]
                
            # Node result concatenation (equivalent to layer concatenation)
            elif n.op_type == "Concat":
                if n.attribute[0].i == 0:
                    concat_f = casadi.vertcat
                else:
                    concat_f = casadi.horzcat
                out = concat_f(ins[0],ins[1])
            
            # The model in its actual state is capable of the conversion of other
            # ONNX node types and operations. This design should be sufficient
            # for the deep learning regression applications with at maximum
            # 2-dimensional arrays as input (no convolutional neural networks or
            # models with 3D and higher input arrays !)
            else:
                raise Exception("The layer type: {} is not yet included in the conversion tool".format(n.op_type))
            if isinstance(out,(np.ndarray)):
                out = symvar_type(out)
            
            # Assignment of the computation result to the variables "all_values"
            # and "node_values" as initialization for the next node operation
            all_values[n.output[0]] = out
            node_values[n.output[0]] = out
        
        # Assignment of all node output values to the class object 
        self.values = node_values
            
          
            
        # computation values of relevant nodes only
        # Relevant nodes are nodes, whose output results correspond to the
        # layer output results: node names of bias addition and weight 
        # multiplication operations are eliminated. Hence, as layer results are
        # only results from concatenation nodes or activation function nodes considered
        self.relevant_layers = []
        self.relevant_values = {}
        for layer, value in node_values.items():
            operation_type = layer[layer.rfind("/")+1:].lower() # Example: "MatMul:0" from the full node 
                                                                # name "model_1/1st_output/MatMul:0"
            if not any(irrelevant in operation_type for irrelevant in ["matmul","biasadd","expanddims"]):
                relevant_layer_name = layer
                if layer.count("/")>1:
                    # Example: "model_1" from the full layer name "model_1/xy/Tanh:0"
                    relevant_layer_name = layer[layer.find("/")+1:layer.rfind("/")]
                
                # Assignment of the list of the relevant layer names and of their
                # respective values to the ONNX2CasADi object
                self.relevant_layers.append(relevant_layer_name)                
                self.relevant_values[relevant_layer_name] = self.values[layer]
                
     
        # Generation of the equivalent CasADi model (of the Outputs)
        self.output_values = {}
        for l in output_layers:
            self.output_values[l] = all_values[l]
        

        self.casadi_function = casadi.Function(self.name,list(self.input.values()),
                    list(self.output_values.values()),list(self.input.keys()),output_layers)
         
        


    def __getitem__(self, key):
        """ Enables the output of the CasADi expression of a specific layer or 
        graph operation node.
        """
        values = self.values
        values.update(self.relevant_values)
        out = None
        for layer_name in values:
            if key == layer_name:                
                out = values[layer_name]
        
        if out == None:
            raise Exception("The layer '{}' is unknown.\nIt should be either a name of onnx computational node (see the property: self.layers) or possibly a name of one of the original Keras model layers (see the property: self.relevant_layers)".format(key))
        return out