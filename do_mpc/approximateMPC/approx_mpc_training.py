"""
Date: 2023-09-27
Author: Lukas Lüken

Pytorch script to load some data and train a feedforward neural network to approximate the MPC controller.
"""
# %% Imports
import torch
from pathlib import Path
from approx_MPC import ApproxMPC, ApproxMPCSettings, plot_history
import json

# %%
# Setup
seed = 0
torch.manual_seed(seed)
# np.random.seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch_data_type = torch.float64
torch.set_default_dtype(torch_data_type)   
file_pth = Path(__file__).parent.resolve()
print("Filepath: ",file_pth)

# %% 
# Config
#########################################################
# Data
train_data_file_name = "dataset_train_10000.pt"
train_data_folder = file_pth.joinpath('datasets')
val_data_file_name = "dataset_val_1000.pt"
val_data_folder = file_pth.joinpath('datasets')

# NN
n_layers = 6 #(L = n+1)
n_neurons = 200
n_in = 3
n_out = 1

# NN Training
N_epochs = 1000
batch_size = 1024
lrs = [1e-2,1e-3,1e-4,1e-5]
overfit_batch = False # Default: False; Used to determine wether NN size is large enough and code is working w.o. bugs

#########################################################

# %% 
# Data loading

# Load data
print("Loading data...")
data_train = torch.load(train_data_folder.joinpath(train_data_file_name),map_location=device)
data_val   = torch.load(val_data_folder.joinpath(val_data_file_name),map_location=device)
print("Data loaded.")

# Train data
X_train,Y_train,_,_ = data_train.tensors
Y_train = Y_train[:,None]
if overfit_batch:
    X_train = X_train[:batch_size,:]
    Y_train = Y_train[:batch_size,:]

# Validation data
X_val,Y_val,_,_ = data_val.tensors
Y_val = Y_val[:,None]

n_train = X_train.shape[0]
N_steps = int(torch.ceil(torch.tensor(n_train/batch_size)))

# Print statistics: n_train, N_steps, batch_size
print("----------------------------------")
print("n_train: ",n_train)
print("N_steps: ",N_steps)
print("batch_size: ",batch_size)
print("----------------------------------")

# %% 
# Setup approx. MPC
print("Setting up approx. MPC...")
settings = ApproxMPCSettings(n_in=n_in,n_out=n_out,n_layers=n_layers,n_neurons=n_neurons)
approx_mpc = ApproxMPC(settings=settings)
# approx_mpc.set_device(device)
print("Approx. MPC setup complete.")

# %% 
# Scale data and setup data loaders
print("Scaling data...")
X_train_scaled, Y_train_scaled = approx_mpc.scale_dataset(X_train,Y_train)
X_val_scaled, Y_val_scaled = approx_mpc.scale_dataset(X_val,Y_val)
print("Data scaled.")

# torch data loader
train_dataset = torch.utils.data.TensorDataset(X_train_scaled, Y_train_scaled)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_dataset = torch.utils.data.TensorDataset(X_val_scaled, Y_val_scaled)
val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=X_val_scaled.shape[0],shuffle=False)


# %%
# Training
print("Training...")
history_pt = {"epochs": [], "train_loss": [], "val_loss": []}

for idx_lr, lr in enumerate(lrs):
        optim = torch.optim.AdamW(approx_mpc.ann.parameters(),lr=lr)
        # optim = torch.optim.Adam(approx_mpc.ann.parameters(),lr=lr)
        # train
        history_pt = approx_mpc.train(N_epochs,optim,train_loader,val_loader,history_pt,verbose=True)
    # plot history

# plot history
fig, ax = plot_history(history_pt)


# %%
# Save model
print("Saving run...")
# approx_mpc.save_model_settings(file_name="approx_MPC_settings")
# approx_mpc.save_model(file_name="approx_MPC_state_dict")

run_hparams = {"n_layers": n_layers, "n_neurons": n_neurons, "n_in": n_in, 
               "n_out": n_out, "N_epochs": N_epochs, "batch_size": batch_size, 
               "lrs": lrs, "overfit_batch": overfit_batch,
               "train_data_file_name": train_data_file_name,
               "val_data_file_name": val_data_file_name,
               "n_train": n_train, "N_steps": N_steps,
               "optimizer": str(optim.__class__.__name__),
               "train_loss": history_pt["train_loss"][-1],
               "val_loss": history_pt["val_loss"][-1]}

# check if folder "run_i" exists, count up if it does
for i in range(100):
    run_folder = file_pth.joinpath("approx_mpc_models",f"run_{i}")
    if run_folder.exists():
        continue
    else:
        run_folder.mkdir()
        approx_mpc.save_model_settings(folder_path=run_folder,file_name="approx_MPC_settings")
        approx_mpc.save_model(folder_path=run_folder,file_name="approx_MPC_state_dict")
        fig.savefig(run_folder.joinpath("history.png"))
        # save run_hparams as json
        with open(run_folder.joinpath("run_hparams.json"), 'w') as fp:
            json.dump(run_hparams, fp)
        break



# %%
# Load model
# print("Loading model...")
# approx_mpc_settings_loaded = ApproxMPCSettings.from_json("approx_MPC_settings")
# approx_mpc_loaded = ApproxMPC(approx_mpc_settings_loaded)
# approx_mpc_loaded.load_state_dict(file_name="approx_MPC_state_dict")
