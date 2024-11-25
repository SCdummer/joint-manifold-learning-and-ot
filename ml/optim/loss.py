import torch
import monai.losses as monai_losses
import segmentation_models_pytorch as smp


def get_loss(loss_name, loss_params):
    if hasattr(torch.nn, loss_name):
        return getattr(torch.nn, loss_name)(**loss_params)
    elif hasattr(smp.losses, loss_name):
        return getattr(smp.losses, loss_name)(**loss_params)
    elif hasattr(monai_losses, loss_name):
        return getattr(monai_losses, loss_name)(**loss_params)
