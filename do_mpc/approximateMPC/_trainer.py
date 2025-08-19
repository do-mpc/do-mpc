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
from torch.utils.data import DataLoader, random_split, TensorDataset
from pathlib import Path
import pickle as pkl
import matplotlib.pyplot as plt
import json
from ._ampcsettings import TrainerSettings
from ._ampcsettings import TrainerSchedulerSettings
import pathlib


class Trainer:
    """Trainer class for training the ApproxMPC.

    .. versionadded:: >v4.6.0

    The Trainer class is used to train the ApproxMPC. The training data is loaded from the data directory and the
    training is done using the training data.

    **Configuration and setup:**

    Configuring and setting up the Trainer controller involves the following steps:

    1. Configure the Trainer controller with :py:class:`TrainerSettings` and :py:class:`TrainerSchedulerSettings`. The ApproxMPC instance has the attribute ``settings`` and ``scheduler_settings`` which is an instance of :py:class:`TrainerSettings` and :py:class:`TrainerSchedulerSettings` respectively.

    2. To finalize the class configuration :py:meth:`setup` may be called. This method sets up the Trainer.

    **Usage:**

    The Trainer can be used to sample the data by calling the :py:meth:`default_training` method. This method starts training the ApproxMPC with the provided configuration.

    **Example:**

    ::

        # Create an MPC instance
        mpc = do_mpc.controller.MPC(...)
        mpc.setup()

        # trainer
        trainer = Trainer(approx_mpc)
        trainer.settings.n_samples = 1000
        trainer.settings.scheduler_flag = True
        trainer.scheduler_settings.cooldown = 0
        trainer.scheduler_settings.patience = 10
        trainer.setup()
        trainer.default_training()

    Args:
        approx_mpc (do_mpc.approximateMPC.ApproxMPC): The ApproxMPC object to be trained.

    """

    def __init__(self, approx_mpc):
        # storage
        self.approx_mpc = approx_mpc

        # settings
        self._settings = TrainerSettings()
        self._sc_settings = TrainerSchedulerSettings()

        # flags
        self.flags = {
            "setup": False,
        }

        return None

    def setup(self):
        """Sets up the Trainer.

        This method sets up the Trainer. This method must be called before training the ApproxMPC.

        Returns:
            None: None
        """
        assert self.flags["setup"] is False, "Setup can only be once."
        self.flags.update(
            {
                "setup": True,
            }
        )
        # checks for mandatory variables
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
        """The settings attribute.

        The settings attribute is an instance of :py:class:`TrainerSettings` and is used to configure the Trainer.

        Example to change settings:

        ::

            trainer.settings.n_samples = 1000
        
        Note:     
            Settings cannot be updated after calling :py:meth:`do_mpc.approximateMPC.Trainer.setup`.

        Returns:
            TrainerSettings: Contains the configurable settings of the trainer.
        """
        return self._settings

    @settings.setter
    def settings(self, val):
        warnings.warn("Cannot change the settings attribute")

    @property
    def scheduler_settings(self):
        """The settings attribute.

        The settings attribute is an instance of :py:class:`TrainerSchedulerSettings` and is used to configure the Scheduler for the Trainer.

        Example to change settings:

        ::

            trainer.scheduler_settings.cooldown = 0
        
        Note:     
            Settings cannot be updated after calling :py:meth:`do_mpc.approximateMPC.Trainer.setup`.

        Returns:
            TrainerSettings: Contains the configurable settings of the scheduler.
        """
        return self._sc_settings

    @scheduler_settings.setter
    def scheduler_settings(self, val):
        warnings.warn("Cannot change the scheduler_settings attribute")

    def scale_dataset(self, x, u0):
        """Scales the dataset.

        Args:
            x (torch.tensor): States for the system.
            u0 (torch.tensor): Inputs of the system.

        Returns:
            (torch.tensor, torch.tensor): _description_
        """
        # Check setup.
        assert self.flags['setup'] == True, 'MPC was not setup yet. Please call Trainer.setup().'
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


    def load_data(self):
        """This function loads the data from the data directory and returns the relevant data loaders and the optimizer.

        Returns:
            (torch.utils.data.dataloader.DataLoader, torch.utils.data.dataloader.DataLoader, torch.optim.adam.Adam): Returns the relevant data loaders and the optimizer.
        """
        # Check setup.
        assert self.flags['setup'] == True, 'MPC was not setup yet. Please call Trainer.setup().'

        data_dir = self.settings.data_dir
        dataset_name = self.settings.dataset_name

        val = self.settings.val
        batch_size = self.settings.batch_size
        shuffle = self.settings.shuffle
        learning_rate = self.settings.learning_rate

        # self.dir = Path()
        self.hyperparameters = {}
        self.hyperparameters["data_dir"] = self.settings.data_dir
        self.hyperparameters["dataset_name"] = self.settings.dataset_name
        self.hyperparameters["scheduler_flag"] = self.settings.scheduler_flag
        self.hyperparameters["lr_reduce_factor"] = self.scheduler_settings.factor
        self.hyperparameters["lr_scheduler_patience"] = self.scheduler_settings.patience
        self.hyperparameters["lr_scheduler_cooldown"] = self.scheduler_settings.cooldown
        self.hyperparameters["min_lr"] = self.scheduler_settings.min_lr
        self.hyperparameters["val"] = val
        self.hyperparameters["batch_size"] = batch_size
        self.hyperparameters["shuffle"] = shuffle
        self.hyperparameters["learning_rate"] = learning_rate

        # saving hyperparameters as a json

        json_dir = Path(self.settings.results_dir).joinpath(
            "results_" + dataset_name, "hyperparameters.json"
        )

        # Ensure the directory exists, if not create it
        Path(self.settings.results_dir).joinpath("results_" + dataset_name).mkdir(parents=True, exist_ok=True)

        with open(json_dir, "w") as f:
            json.dump(self.hyperparameters, f, indent=4)

        data_dir = Path(data_dir)
        data_dir = data_dir.joinpath(dataset_name)
        data_dir = data_dir.joinpath("data_" + dataset_name + "_opt.pkl")

        print(f"Path from trainer to sampled files\n {data_dir}")
        with open(data_dir, "rb") as f:
            dataset = pkl.load(f)

        x0 = torch.tensor(dataset["x0"], dtype=self.approx_mpc.torch_data_type).reshape(
            -1, self.approx_mpc.mpc.model.n_x
        )
        u0 = torch.tensor(dataset["u0"], dtype=self.approx_mpc.torch_data_type).reshape(
            -1, self.approx_mpc.mpc.model.n_u
        )

        if self.approx_mpc.mpc.flags["set_rterm"]:
            # u_prev = torch.tensor(dataset['u_prev'], dtype=self.approx_mpc.torch_data_type).reshape(n_data, -1)
            u_prev = torch.tensor(
                dataset["u_prev"], dtype=self.approx_mpc.torch_data_type
            ).reshape(-1, self.approx_mpc.mpc.model.n_u)
            x = torch.cat((x0, u_prev), dim=1)
        else:
            x = x0
        if self.approx_mpc.settings.scaling:
            x_scaled, u0_scaled = self.scale_dataset(x, u0)
        else:
            x_scaled = x
            u0_scaled = u0
        data = TensorDataset(x_scaled, u0_scaled)
        training_data, val_data = random_split(
            data, [1 - val, val], generator=self.generator
        )
        train_dataloader = DataLoader(
            training_data,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=self.generator,
        )
        test_dataloader = DataLoader(
            val_data, batch_size=batch_size, shuffle=shuffle, generator=self.generator
        )
        optimizer = optim.Adam(self.approx_mpc.net.parameters(), lr=learning_rate)

        if self.settings.scheduler_flag:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.scheduler_settings.mode,
                factor=self.scheduler_settings.factor,
                patience=self.scheduler_settings.patience,
                threshold=self.scheduler_settings.threshold,
                threshold_mode=self.scheduler_settings.threshold_mode,
                cooldown=self.scheduler_settings.cooldown,
                min_lr=self.scheduler_settings.min_lr*0.1,
                eps=self.scheduler_settings.eps,
            )
        return train_dataloader, test_dataloader, optimizer

    def log_value(self, val, key):
        """This method logs the value of the key in the history.
        The key cold be either of 'epoch', 'train_loss', 'val_loss', 'lr'.

        Args:
            val (int): Contains the value to be logged.
            key (str): Contains the key to which the value belongs.
        """
        if torch.is_tensor(val):
            val = val.detach().cpu().item()
        assert isinstance(val, (int, float)), "Value must be a scalar."
        if not key in self.history.keys():
            self.history[key] = []
        self.history[key].append(val)

    def print_last_entry(self, keys=["epoch,train_loss"]):
        """This method prints the last entry of the history.
        The keys to be printed can be provided as a list. Possible entries can be 'epoch', 'train_loss', 'val_loss', 'lr'.

        Args:
            keys (list, optional): Store the keys. Possible entries are 'epoch', 'train_loss', 'val_loss', 'lr'.. Defaults to ["epoch,train_loss"].
        """
        assert isinstance(keys, list), "Keys must be a list."
        for key in keys:
            # check wether keys are in history
            assert key in self.history.keys(), "Key not in history."
            print(key, ": ", self.history[key][-1])

    def visualize_and_store_history(self):
        """This method visualizes and stores the history of the training.

        Returns:
            None: None
        """
        pathlib.Path(self.settings.results_dir).joinpath(
            "results_" + self.settings.dataset_name
        ).mkdir(parents=True, exist_ok=True)
        if self.settings.save_history:
            assert self.settings.data_dir is not None, "exp_pth must be provided."

            file_path_hist = Path(self.settings.results_dir).joinpath(
                "results_" + self.settings.dataset_name, "training_history.json"
            )
            file_path_app = Path(self.settings.results_dir).joinpath(
                "results_" + self.settings.dataset_name, "approx_mpc.pth"
            )
            self.approx_mpc.save_to_state_dict(file_path_app)
            # Save to a JSON file
            with open(file_path_hist, "w") as json_file:
                json.dump(self.history, json_file, indent=4)
        # setting up plot
        if self.settings.save_fig or self.settings.show_fig:
            fig, ax = plt.subplots(2, figsize=(8, 3 * 2))
            fig.suptitle("Training History")

            # plotting learning rate
            ax[0].plot(self.history["epoch"], self.history["lr"], label="Learning Rate")
            ax[0].set_yscale("log")
            ax[0].set_ylabel("Learning Rate")

            # plotting losses
            ax[1].plot(
                self.history["epoch"], self.history["train_loss"], label="Training Loss"
            )
            ax[1].plot(
                self.history["epoch"], self.history["val_loss"], label="Validation Loss"
            )
            ax[1].set_ylabel("Losses")
            ax[1].set_yscale("log")
            ax[1].legend()

            # adding x axis label
            ax[-1].set_xlabel("epoch")

            if self.settings.show_fig:
                plt.show()

            if self.settings.save_fig:
                assert self.settings.data_dir is not None, "exp_pth must be provided."
                fig.savefig(
                    Path(self.settings.results_dir).joinpath(
                        "results_" + self.settings.dataset_name,
                        "training_history.png",
                    )
                )
            plt.close(fig)

        return None

    def train_step(self, optim, x, y):
        """This method computes the forward pass and the backward pass of the class and returns the loss.

        Args:
            optim (torch.optim.adam.Adam): Contains the optimizer chosen for the backward pass.
            x (torch.Tensor): This contains the states of the system.
            y (torch.Tensor): This contains the evaluated input of the system. When properly trained, this should be same a the output of the mpc class.

        Returns:
            float: Scalar value containing the mean summed squared error of the predicted output from the actual output.
        """
        optim.zero_grad()
        y_pred = self.approx_mpc(x)
        loss = torch.nn.functional.mse_loss(y_pred, y)
        loss.backward()
        optim.step()
        return loss.item()

    def train_epoch(self, optim, train_loader):
        """This method calculates the training loss averaged over all epoch, with the training data.

        Args:
            optim (torch.optim.adam.Adam): Contains the optimizer chosen for the backward pass.
            train_loader (torch.utils.data.dataloader.DataLoader): This DataLoader contains the training data.

        Returns:
            float: Scalar value containing the mean summed squared error of the predicted output from the actual output, averaged over all the epochs.
        """
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
        """This method calculates the loss with validation data.

        Args:
            x (torch.Tensor): This contains the states of the system.
            y (torch.Tensor): This contains the evaluated input of the system. When properly trained, this should be same a the output of the mpc class.

        Returns:
            float: _description_
        """
        with torch.no_grad():
            y_pred = self.approx_mpc(x)
            loss = torch.nn.functional.mse_loss(y_pred, y)
        return loss.item()

    def validation_epoch(self, val_loader):
        """This method calculates the loss with validation data averaged over all epochs.

        Args:
            val_loader (torch.utils.data.dataloader.DataLoader): This DataLoader contains the validation data.

        Returns:
            float: Scalar value containing the mean summed squared error of the predicted output from the actual output, averaged over all the epochs.
        """
        val_loss = 0.0
        for idx_val_batch, batch in enumerate(val_loader):
            x_val, y_val = batch
            loss = self.validation_step(x_val, y_val)
            val_loss += loss
        n_val_steps = idx_val_batch + 1
        val_loss = val_loss / n_val_steps
        return val_loss

    def default_training(self):
        """This method handles the training of the ApproxMPC class.
        Data loaded from the data directory is used to train and validate the ApproxMPC class. 
        Training performance is stored and subsequently plotted and saves depending on the settings chosen during the setup.

        Returns:
            None: None
        """
        # Check setup.
        assert self.flags['setup'] == True, 'MPC was not setup yet. Please call Trainer.setup().'

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
            if (epoch + 1) % self.settings.print_frequency == 0:
                self.print_last_entry(keys=print_keys)
                print("-------------------------------")

            # scheduler
            if self.settings.scheduler_flag:
                self.lr_scheduler.step(val_loss)
                # break if training min learning rate is reached
                if optimizer.param_groups[0]["lr"] < self.scheduler_settings.min_lr:
                    break

        # put visualize history
        if (
            self.settings.show_fig
            or self.settings.save_fig
            or self.settings.save_history
        ):
            self.visualize_and_store_history()

        # end
        return None
