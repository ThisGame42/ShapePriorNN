import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F


from torch.utils.data import DataLoader


from Lib.CollectionUtils.JList import JList
from Lib.TrainingUtils.UtilsFn import save_weights, plot_loss
from NN.DeformationNet import DeformationNet
from Data.Dataset.MuscleDataset import MuscleDataset


class Trainer(object):
    def __init__(self, config):
        self.config = config

    def run(self):
        start_training(self.config)


def start_training(config):
    num_epochs = config.epochs
    img_size = config.img_size
    ctl_pts_shape = config.ctl_pts_shape
    model = DeformationNet(img_size, ctrlpt_shape=ctl_pts_shape)

    optimiser = optim.AdamW(lr=config.lr,
                            weight_decay=config.w_deacy,
                            params=model.parameters())

    train_ds = MuscleDataset(config, mode="train")
    val_ds = MuscleDataset(config, mode="val")

    train_loader = DataLoader(dataset=train_ds,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_ds,
                            shuffle=True)

    running_loss_t = np.zeros(num_epochs, dtype=np.float32)
    running_loss_v = np.zeros(num_epochs, dtype=np.float32)
    for epoch in range(num_epochs):
        loss_t_epoch = JList()
        model.train()
        for i, batch in enumerate(train_loader):
            optimiser.zero_grad()
            x, y, prior = batch
            y_hat = model(x, prior)
            loss = F.mse_loss(y_hat, y)
            loss.backward()
            loss_t_epoch.push_back(loss.item())
            optimiser.step()
        running_loss_t[epoch] = loss_t_epoch.mean_f()
        loss_t_epoch.reset()
        print(f"The mean training loss of epoch {epoch+1} is {running_loss_t[epoch]}.")

        with torch.no_grad():
            loss_v_epoch = JList()
            model.eval()
            for i, batch in enumerate(val_loader):
                x, y, prior = batch
                y_hat = model(y, prior)
                loss = F.mse_loss(y_hat, y)
                loss_v_epoch.push_back(loss.item())
            running_loss_v[epoch] = loss_v_epoch.mean_f()
            loss_v_epoch.reset()
            print(f"The mean validation loss of epoch {epoch+1} is {running_loss_v[epoch]}")

    save_weights(model, config.path_model)

    plot_loss(loss_t=running_loss_t,
              loss_v=running_loss_v,
              path=config.path_plot)