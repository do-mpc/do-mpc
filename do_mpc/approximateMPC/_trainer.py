
#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.


# imports
import warnings
import torch
import torch.optim as optim
from torch.utils.data import DataLoader,random_split,TensorDataset
from pathlib import Path
import pickle as pkl
import matplotlib.pyplot as plt
import json
from ._ampcsettings import TrainerSettings
from ._ampcsettings import TrainerSchedulerSettings

#plt.ion()

class Trainer():
    def __init__(self, approx_mpc):

        # storage
        self.approx_mpc = approx_mpc

        # settings
        self._settings = TrainerSettings()
        self._sc_settings = TrainerSchedulerSettings()

        # flags
        self.flags = {
            'setup': False,
        }

        return None

    def setup(self):
        assert self.flags['setup'] is False, "Setup can only be once."
        self.flags.update({
            'setup': True,
        })
        #checks for mandatory variables
        self._settings.check_for_mandatory_settings()

        # sets device
        torch.set_default_device(self.approx_mpc.torch_device)

        # for random_split
        self.generator = torch.Generator(device=self.approx_mpc.torch_device)

        # logging
        self.history = {"epoch": []}

        return None

    @property
    def settings(self):
        return self._settings

    @settings.setter
    def settings(self, val):
        warnings.warn('Cannot change the settings attribute')

    @property
    def scheduler_settings(self):
        return self._sc_settings

    @scheduler_settings.setter
    def scheduler_settings(self, val):
        warnings.warn('Cannot change the scheduler_settings attribute')

    def scale_dataset(self, x, u0):
        x_shift = self.approx_mpc.x_shift  # torch.tensor(lbx)
        x_range = self.approx_mpc.x_range  # torch.tensor(ubx - lbx)
        y_shift = self.approx_mpc.y_shift  # torch.tensor(lbu)
        y_range = self.approx_mpc.y_range  # torch.tensor(ubu - lbu)
        x_scaled = (x - x_shift) / x_range
        # u_prev_scaled = (u_prev - y_shift.T) / y_range.T
        u0_scaled = (u0 - y_shift) / y_range
        x_scaled = x_scaled.type(self.approx_mpc.torch_data_type)
        u0_scaled = u0_scaled.type(self.approx_mpc.torch_data_type)
        return x_scaled, u0_scaled

    # def load_data(self,data_dir,n_samples, val=0.2,batch_size=1000,shuffle=True,learning_rate=1e-3):
    def load_data(self):

        data_dir = self.settings.data_dir
        n_samples = self.settings.n_samples
        val = self.settings.val
        batch_size = self.settings.batch_size
        shuffle = self.settings.shuffle
        learning_rate = self.settings.learning_rate

        #self.dir = Path()
        self.hyperparameters = {}
        self.hyperparameters['data_dir'] = self.settings.data_dir
        self.hyperparameters['n_samples'] = self.settings.n_samples
        self.hyperparameters['scheduler_flag'] = self.settings.scheduler_flag
        self.hyperparameters['lr_reduce_factor'] = self.scheduler_settings.factor
        self.hyperparameters['lr_scheduler_patience'] = self.scheduler_settings.patience
        self.hyperparameters['lr_scheduler_cooldown'] = self.scheduler_settings.cooldown
        self.hyperparameters['min_lr'] = self.scheduler_settings.min_lr
        self.hyperparameters['val'] = val
        self.hyperparameters['batch_size'] = batch_size
        self.hyperparameters['shuffle'] = shuffle
        self.hyperparameters['learning_rate'] = learning_rate

        # saving hyperparameters as a json

        json_dir = Path(data_dir)
        json_dir = json_dir.joinpath('hyperparameters' + str(n_samples) + '.json')

        with open(json_dir, 'w') as f:
            json.dump(self.hyperparameters, f, indent=4)


        data_dir = Path(data_dir)
        data_dir = data_dir.joinpath('data_n' + str(n_samples) + '_opt.pkl')
        with open(data_dir, 'rb') as f:
            dataset = pkl.load(f)




        # take n_data from
        # n_data=len(dataset['x0']) * 10 # fixed for 10 trajectory len
        # n_data = len(dataset['x0'])
        # x0= torch.tensor(dataset['x0'],dtype=self.approx_mpc.torch_data_type).reshape(n_data, -1)
        # u0=torch.tensor(dataset['u0'],dtype=self.approx_mpc.torch_data_type).reshape(n_data, -1)

        x0 = torch.tensor(dataset['x0'], dtype=self.approx_mpc.torch_data_type).reshape(-1,
                                                                                        self.approx_mpc.mpc.model.n_x)
        u0 = torch.tensor(dataset['u0'], dtype=self.approx_mpc.torch_data_type).reshape(-1,
                                                                                        self.approx_mpc.mpc.model.n_u)

        if self.approx_mpc.mpc.flags['set_rterm']:
            # u_prev = torch.tensor(dataset['u_prev'], dtype=self.approx_mpc.torch_data_type).reshape(n_data, -1)
            u_prev = torch.tensor(dataset['u_prev'], dtype=self.approx_mpc.torch_data_type).reshape(-1,
                                                                                                    self.approx_mpc.mpc.model.n_u)
            x = torch.cat((x0, u_prev), dim=1)
        else:
            x = x0

        x_scaled, u0_scaled = self.scale_dataset(x, u0)
        data = TensorDataset(x_scaled, u0_scaled)
        training_data, val_data = random_split(data, [1 - val, val], generator=self.generator)
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle, generator=self.generator)
        test_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=shuffle, generator=self.generator)
        optimizer = optim.Adam(self.approx_mpc.net.parameters(), lr=learning_rate)

        if self.settings.scheduler_flag:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=self.scheduler_settings.mode,
                                                                           factor=self.scheduler_settings.factor,
                                                                           patience=self.scheduler_settings.patience,
                                                                           threshold=self.scheduler_settings.threshold,
                                                                           threshold_mode=self.scheduler_settings.threshold_mode,
                                                                           cooldown=self.scheduler_settings.cooldown,
                                                                           min_lr=self.scheduler_settings.min_lr,
                                                                           eps=self.scheduler_settings.eps)
        return train_dataloader, test_dataloader, optimizer


    def log_value(self, val, key):
        if torch.is_tensor(val):
            val = val.detach().cpu().item()
        assert isinstance(val, (int, float)), "Value must be a scalar."
        if not key in self.history.keys():
            self.history[key] = []
        self.history[key].append(val)

    def print_last_entry(self, keys=["epoch,train_loss"]):
        assert isinstance(keys, list), "Keys must be a list."
        for key in keys:
            # check wether keys are in history
            assert key in self.history.keys(), "Key not in history."
            print(key, ": ", self.history[key][-1])

    def visualize_history(self, keys):

        # setting up plot
        fig, ax = plt.subplots(len(keys), figsize=(8, 3 * len(keys)))
        fig.suptitle('Training History')

        # If there's only one key, ax will not be a list
        if len(keys) == 1:
            ax = [ax]

        for i, key in enumerate(keys):
            #fig, ax = plt.subplots()
            ax[i].plot(self.history[key], label=key)
            
            ax[i].set_ylabel(key)
            #ax.set_title(key)
            if self.settings.log_scaling:
                ax[i].set_yscale('log')
            #ax.legend()
            
        fig.legend()
        ax[-1].set_xlabel('epoch')


        if self.settings.show_fig:
            #fig.show()
            plt.show()
            #pass
            
        if self.settings.save_fig:
            assert self.settings.data_dir is not None, "exp_pth must be provided."
            fig.savefig(Path(self.settings.data_dir).joinpath("training_history.png"))

        return None

    def train_step(self, optim, x, y):
        optim.zero_grad()
        y_pred = self.approx_mpc(x)
        loss = torch.nn.functional.mse_loss(y_pred, y)
        loss.backward()
        optim.step()
        return loss.item()

    def train_epoch(self, optim, train_loader):
        train_loss = 0.0
        # Training Steps
        for idx_train_batch, batch in enumerate(train_loader):
            x, y = batch
            loss = self.train_step(optim, x, y)
            train_loss += loss
        n_train_steps = idx_train_batch + 1
        train_loss = train_loss / n_train_steps
        return train_loss

    def validation_step(self, x, y):
        with torch.no_grad():
            y_pred = self.approx_mpc(x)
            loss = torch.nn.functional.mse_loss(y_pred, y)
        return loss.item()

    def validation_epoch(self, val_loader):
        val_loss = 0.0
        for idx_val_batch, batch in enumerate(val_loader):
            x_val, y_val = batch
            loss = self.validation_step(x_val, y_val)
            val_loss += loss
        n_val_steps = idx_val_batch + 1
        val_loss = val_loss / n_val_steps
        return val_loss

    def default_training(self, print_frequency=10 ):

        n_epochs = self.settings.n_epochs
        train_dataloader, test_dataloader, optimizer = self.load_data()


        for epoch in range(n_epochs):
            # Training
            train_loss = self.train_epoch(optimizer, train_dataloader)

            # Logging
            self.log_value(epoch, "epoch")
            self.log_value(train_loss, "train_loss")
            self.log_value(optimizer.param_groups[0]["lr"], "lr")
            print_keys = ["epoch", "train_loss"]

            # Validation
            if test_dataloader is not None:
                val_loss = self.validation_epoch(test_dataloader)
                self.log_value(val_loss, "val_loss")
                print_keys.append("val_loss")

            # Print
            if (epoch + 1) % print_frequency == 0:
                self.print_last_entry(keys=print_keys)
                print("-------------------------------")

            # scheduler
            if self.settings.scheduler_flag:
                self.lr_scheduler.step(train_loss)

                # break if training min learning rate is reached
                if optimizer.param_groups[0]["lr"] <= 1e-8:
                    break

        # put visualise history
        if self.settings.show_fig or self.settings.save_fig:
            self.visualize_history(keys=list(self.history.keys()))

        # end
        return self.history