import os
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from Lib.CollectionUtils.JList import JList


from Lib.FileUtils.UtilsFn import FileHelper


def to_onehot(labels: torch.Tensor, num_classes: int = 3, is_3d: bool = False) -> torch.Tensor:
    """
    Convert a slice of image into one-hot encoding. Expects the input data in the shape of
    [C, H, W]. The returned one-hot encoded tensor will be in the shape of [C, H, W] where C = num_classes.

    Note: Only works on a single instance, be it a single slice of the image of a sub-volume of the scan.
    DOES NOT work with mini-batches.
    """
    labels = torch.squeeze(labels)
    target = F.one_hot(labels.to(torch.int64), num_classes=num_classes)
    to_dim = (3, 0, 1, 2) if is_3d else (2, 0, 1)
    one_hot = target.permute(to_dim)
    return one_hot


def split_vol(x, y, shape_prior, slices_used, img, label, img_affine, label_affine):
    # Split the volume into sub-volumes
    # This happens when a 3D volume is too big to be processed all in once
    # e.g. (288, 96, 192). In this case we split it into smaller sub-volumes
    assert x.shape[
               0] % slices_used == 0, f"The number of slices in the volume must be divisble by the number of slices " \
                                      f"in the sub-volumes."
    assert x.shape == y.shape == shape_prior.shape
    num_sub_vol = x.shape[0] // slices_used
    ret_val = JList()
    for i in range(num_sub_vol):
        start = i * slices_used
        end = (i + 1) * slices_used
        x_i = x[start:end]
        y_i = y[start:end]
        prior_i = shape_prior[start:end]
        ret_val.push_back((x_i, y_i, prior_i, img, label, img_affine, label_affine))

    return ret_val


def save_weights(model, path) -> None:
    torch.save(model.state_dict(), path)


def plot_loss(loss_t, loss_v, path) -> None:
    num_epochs = len(loss_t)
    # Output Running Loss
    X = np.arange(1, num_epochs, 1)
    Y = loss_t
    Y1 = loss_v
    plt.title("Running Loss")
    plt.plot(X, Y, label="Training Loss")
    plt.plot(X, Y1, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    dir, fname = os.path.split(path)
    FileHelper.make_dir_if_none(dir)
    plt.savefig(path)
    plt.cla()