# Imports
# import json
# from dataclasses import dataclass, asdict
# from typing import Tuple
import torch
# import numpy as np
from pathlib import Path
# import pandas as pd
# import matplotlib.pyplot as plt

# Feedforward NN
class FeedforwardNN(torch.nn.Module):
    """Feedforward Neural Network model.

    Args:
        n_in (int): Number of input features.
        n_out (int): Number of output features.
        n_hidden_layers (int): Number of hidden layers.
        n_neurons (int): Number of neurons in each hidden layer.
        act_fn (str): Activation function.
        output_act_fn (str): Output activation function.
    """
    def __init__(self, n_in, n_out, n_hidden_layers=2, n_neurons=500, act_fn='relu', output_act_fn='linear'):
        super().__init__()
        assert n_hidden_layers >= 0, "Number of hidden layers must be >= 0."
        self.n_in = n_in
        self.n_out = n_out
        self.n_layers = n_hidden_layers+1
        self.n_neurons = n_neurons
        self.act_fn = act_fn
        self.output_act_fn = output_act_fn
        self.layers = torch.nn.ModuleList()
        for i in range(self.n_layers):
            if i == 0:
                self.layers.append(torch.nn.Linear(n_in, n_neurons))
                self.layers.append(self._get_activation_layer(act_fn))
            elif i == self.n_layers - 1:
                self.layers.append(torch.nn.Linear(n_neurons, n_out))
                if output_act_fn != 'linear':
                    self.layers.append(self._get_activation_layer(output_act_fn))
            else:
                self.layers.append(torch.nn.Linear(n_neurons, n_neurons))
                self.layers.append(self._get_activation_layer(act_fn))

    def _get_activation_layer(self,act_fn):
        if act_fn == 'relu':
            return torch.nn.ReLU()
        elif act_fn == 'tanh':
            return torch.nn.Tanh()
        elif act_fn == 'leaky_relu':
            return torch.nn.LeakyReLU()
        elif act_fn == 'sigmoid':
            return torch.nn.Sigmoid()
        else:
            raise ValueError("Activation function not implemented.")

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x
    
    @torch.no_grad()
    def count_params(self):
        n_params = sum(param.numel() for param in self.parameters())
        return n_params


# Approximate MPC
class ApproxMPC(torch.nn.Module):
    """Approximate MPC class with Neural Network embedded.

    Args:
        net (torch.nn.Module): Neural Network.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.torch_data_type = torch.float32
        self.step_return_type = "numpy" # "torch" or "numpy"
        self.lb_u = None # lower bound of control actions
        self.ub_u = None # upper bound of control actions

        self.x_shift = torch.tensor(0.0) # shift of input data (min-max or standard scaling)
        self.x_range = torch.tensor(1.0) # range of input data (min-max or standard scaling)
        self.y_shift = torch.tensor(0.0) # shift of output data (min-max or standard scaling)
        self.y_range = torch.tensor(1.0) # range of output data (min-max or standard scaling)

        self._set_device()

        print("----------------------------------")
        print(self)
        print("----------------------------------")

    def _set_device(self,device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.net.to(device)

    def forward(self, x):
        """Forward pass of the neural network with input scaling and output rescaling.
        """
        x_scaled = self.scale_inputs(x)
        y_scaled = self.net(x_scaled)
        y = self.rescale_outputs(y_scaled)
        return y
    
    def scale_inputs(self,x):
        """Scale inputs.

        Args:
            x (torch.Tensor): Inputs.

        Returns:
            x_scaled (torch.Tensor): Scaled inputs.
        """
        x_scaled = (x-self.x_shift)/self.x_range
        return x_scaled
    
    def rescale_outputs(self,y_scaled):
        """Rescale outputs.

        Args:
            y_scaled (torch.Tensor): Scaled outputs.

        Returns:
            y (torch.Tensor): Rescaled outputs.
        """
        y = y_scaled*self.y_range+self.y_shift
        return y
    
    def clip_control_actions(self,y):
        """Clip (rescaled) outputs of net to satisfy input (control) constraints.

        Args:
            y (torch.Tensor): Outputs.

        Returns:
            y (torch.Tensor): Clipped outputs.
        """
        if self.lb_u is not None:
            y = torch.max(y,self.lb_u)
        if self.ub_u is not None:
            y = torch.min(y,self.ub_u)
        if self.lb_u is None and self.ub_u is None:
            raise ValueError("No output constraints defined. Clipping not possible.")
        return y

    # Predict (Batch)
    @torch.no_grad()
    def predict(self,x_batch):
        y_batch = self.net(x_batch)
        return y_batch
    
    # approximate MPC step method for use in closed loop
    def make_step(self,x,clip_to_bounds=True):
        """Make one step with the approximate MPC.
        Args:
            x (torch.tensor): Input tensor of shape (n_in,).
            clip_to_bounds (bool, optional): Whether to clip the control actions. Defaults to True.
        Returns:
            np.array: Array of shape (n_out,).
        """

        # Check if inputs are tensors
        if not isinstance(x,torch.Tensor):
            x = torch.tensor(x,dtype=self.torch_data_type)

        with torch.no_grad():
            y = self.net(x)
    
        # Clip outputs to satisfy input constraints of MPC
        if clip_to_bounds:
            y = self.clip_control_actions(y)
    
        if self.step_return_type == "numpy":
            y = y.cpu().numpy()
        elif self.step_return_type == "torch":
            y = y
        else:
            raise ValueError("step_return_type must be either 'numpy' or 'torch'.")
        return y



# approx. mpc class
class ApproxMPC():
    """
    Class for the implementation of an approximate MPC. This includes the architecture of the neural networks, the storage of model weights and scaling factors as well as model parameters and the loading of these models based on the stored data. Also functionalities for the use of these approx. MPC in a closed-loop like the "make_step" function are included.

    """
    def __init__(self,settings):
        self.settings = settings

        self.torch_data_type = torch.float32

        # initialize neural network
        self.init_ann()

        # initialize scaling factors from bounds
        if self.settings.scaling_mode == "bounds":
            self.set_scaling_from_bounds()

        # initialize device
        self.set_device()

    def init_ann(self):
        """Initialize the neural network architecture.
        """
        n_layers = self.settings.n_layers
        n_neurons = self.settings.n_neurons
        n_in = self.settings.n_in
        n_out = self.settings.n_out
        act_fn = self.settings.act_fn
        self.ann = self.generate_ffNN(n_layers,n_neurons,n_in,n_out,act_fn)
        
        if hasattr(self,"torch_data_type"):
            self.ann.to(self.torch_data_type)

        print("Neural network initialized with architecture: ", self.ann)
        # print number of nn parameters
        n_params = sum(p.numel() for p in self.ann.parameters() if p.requires_grad)
        print("---------------------------------")
        print("Number of trainable parameters: ", n_params)
        print("---------------------------------")
        self.n_params = n_params
    
    def set_device(self,device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.ann.to(device)
        # all scaling params to device
        self.x_range = self.x_range.to(device)
        self.x_shift = self.x_shift.to(device)
        self.y_range = self.y_range.to(device)
        self.y_shift = self.y_shift.to(device)

    @staticmethod
    def generate_ffNN(n_layers,n_neurons,n_in,n_out,act_fn='relu'):
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(torch.nn.Linear(n_in,n_neurons))
            else:
                layers.append(torch.nn.Linear(n_neurons,n_neurons))
            if act_fn == 'relu':
                layers.append(torch.nn.ReLU())
            elif act_fn == 'tanh':
                layers.append(torch.nn.Tanh())
            else:
                raise ValueError('act_fn must be either relu or tanh')
        ann = torch.nn.Sequential(*layers,torch.nn.Linear(n_neurons,n_out))
        return ann

    def set_scaling_from_bounds(self):
        x_min = torch.tensor([self.settings.lb_x1,self.settings.lb_x2,self.settings.lb_u])
        x_max = torch.tensor([self.settings.ub_x1,self.settings.ub_x2,self.settings.ub_u])
        x_range = x_max-x_min
        x_shift = x_min

        y_min = torch.tensor([self.settings.lb_u])
        y_max = torch.tensor([self.settings.ub_u])
        y_range = y_max-y_min
        y_shift = y_min

        self.x_range = x_range.to(self.torch_data_type)
        self.x_shift = x_shift.to(self.torch_data_type)
        self.y_range = y_range.to(self.torch_data_type)
        self.y_shift = y_shift.to(self.torch_data_type)

        print("Scaling factors set from data with mode: ", "bounds")
        print("x_range: ", self.x_range)
        print("x_shift: ", self.x_shift)
        print("y_range: ", self.y_range)
        print("y_shift: ", self.y_shift)

        self.scaling_mode = "bounds"

    def scale_inputs(self,x_data):
        assert self.scaling_mode == "bounds"
        x_scaled = (x_data-self.x_shift)/self.x_range
        return x_scaled

    def scale_outputs(self,y_data):
        assert self.scaling_mode == "bounds"
        y_scaled = (y_data-self.y_shift)/self.y_range
        return y_scaled

    def rescale_inputs(self,x_scaled):
        assert self.scaling_mode == "bounds"
        x_data = x_scaled*self.x_range+self.x_shift
        return x_data

    def rescale_outputs(self,y_scaled):
        assert self.scaling_mode == "bounds"
        y_data = y_scaled*self.y_range+self.y_shift
        return y_data

    def scale_dataset(self,x_data,y_data):
        x_scaled = self.scale_inputs(x_data)
        y_scaled = self.scale_outputs(y_data)
        return x_scaled, y_scaled

    ### Loading and saving
    def save_model(self,folder_path=None,file_name="approx_MPC_state_dict"):
        if folder_path is None:
            save_pth = Path(file_name+".pt")
        else:
            save_pth = Path(folder_path,file_name+".pt")
        torch.save(self.ann.state_dict(),save_pth)
        print("model saved to: ", save_pth)
    
    def save_model_settings(self,folder_path=None,file_name="approx_MPC_settings"):
        self.settings.save_settings(folder_path,file_name)

    # Use this method to load model parameters
    def load_state_dict(self,folder_pth=None,file_name="approx_MPC_state_dict"):
        if folder_pth is None:
            load_pth = Path(file_name+".pt")
        else:
            load_pth = Path(folder_pth,file_name+".pt")
        self.ann.load_state_dict(torch.load(load_pth))
        print("model loaded from: ", load_pth)

    ### Application as approx. MPC
    def make_step(self,x,scale_inputs=True,rescale_outputs=True, clip_outputs=True):
        """Make one step with the approximate MPC.
        Args:
            x (torch.tensor): Input tensor of shape (n_in,).
            scale_inputs (bool, optional): Whether to scale the inputs. Defaults to True.
            rescale_outputs (bool, optional): Whether to rescale the outputs. Defaults to True.
        Returns:
            np.array: Array of shape (n_out,).
        """

        # Check if inputs are tensors
        if not isinstance(x,torch.Tensor):
            x = torch.tensor(x,dtype=self.torch_data_type)

        if scale_inputs:
            x_scaled = self.scale_inputs(x)
        else:
            x_scaled = x
        with torch.no_grad():
            y_scaled = self.ann(x_scaled)
        if rescale_outputs:
            y = self.rescale_outputs(y_scaled)
        else:
            y = y_scaled        
        # Clip outputs to satisfy input constraints of MPC
        if clip_outputs:
            y = torch.clamp(y,self.settings.lb_u,self.settings.ub_u)
        return y.cpu().numpy()

    ### From here on, code quite independent on approximate MPC: consider moving to separate module, e.g. "Trainer"
    ## Training
    def train_step(self,optim,x,y):
        optim.zero_grad()            
        y_pred = self.ann(x)            
        loss = torch.nn.functional.mse_loss(y_pred,y)            
        loss.backward()            
        optim.step()            
        return loss.item()
    
    def train_epoch(self,optim,train_loader):
        train_loss = 0.0
        # Training Steps
        for idx_train_batch, batch in enumerate(train_loader):
            x, y = batch
            loss = self.train_step(optim,x,y)         
            train_loss += loss
        n_train_steps = idx_train_batch+1
        train_loss = train_loss/n_train_steps
        return train_loss

    def validation_step(self,x,y):
        with torch.no_grad():
            y_pred = self.ann(x)
            loss = torch.nn.functional.mse_loss(y_pred,y)
        return loss.item()
    
    def validation_epoch(self,val_loader):
        val_loss = 0.0
        for idx_val_batch, batch in enumerate(val_loader):
            x_val, y_val = batch
            loss = self.validation_step(x_val,y_val)
            val_loss += loss
        n_val_steps = idx_val_batch+1
        val_loss = val_loss/n_val_steps
        return val_loss

    def train(self,N_epochs,optim,train_loader,val_loader=None,history=None,verbose=True):
        if history is None:
            history = {"epochs": [], "train_loss": [], "val_loss": []}
        for epoch in range(N_epochs):
            # Training
            train_loss = self.train_epoch(optim,train_loader)
            history["epochs"].append(epoch)
            history["train_loss"].append(train_loss)
            if verbose:
                print("Epoch: ",history["epochs"][-1])
                print("Train loss: ",history["train_loss"][-1])
            
            # Validation
            if val_loader is not None:
                val_loss = self.validation_epoch(val_loader)
                history["val_loss"].append(val_loss)
                if verbose:
                    print("Val loss: ",history["val_loss"][-1])
        return history
    


# Main - for test driven development
if __name__ == "__main__":
    print("test")


# todo list
# TODO: Implement scaling
# TODO: Decide on reasonable input and output names 