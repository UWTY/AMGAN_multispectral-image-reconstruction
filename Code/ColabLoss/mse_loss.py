# losses/mse_loss.py
import torch
import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        # alpha: main MSE loss weight
        self.alpha = alpha
        self.mse   = nn.MSELoss(reduction="none")

    def forward(self, outputs, targets, mask_s30):
        """
        outputs: Tensor[B, C, H, W]
        targets: Tensor[B, C, H, W]
        mask_s30:Tensor[B, 1, H, W], mask=1 indicates valid pixels
        """
        # Calculate per-pixel MSE
        loss_map = self.mse(outputs, targets)  # [B, C, H, W]
        # Expand mask to channel dimension
        mask_expand = mask_s30.expand_as(loss_map)  # [B, C, H, W]
        valid_pixels = torch.sum(mask_expand)  # Number of valid elements
        if valid_pixels > 0:
            main_loss = torch.sum(loss_map * mask_expand) / valid_pixels
        else:
            main_loss = torch.tensor(0.0, device=outputs.device)

        return self.alpha * main_loss 

LOSS_CLASS = MSELoss
