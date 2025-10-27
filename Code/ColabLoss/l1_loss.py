# losses/l1_loss.py
import torch
import torch.nn as nn

class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def masked_l1_loss(self, outputs, targets, mask):
        """
        Custom masked L1 loss calculation
        outputs: Tensor[B, C, H, W]
        targets: Tensor[B, C, H, W]
        mask: Tensor[B, C, H, W], mask=1 indicates valid pixels
        """
        # Calculate L1 distance: |outputs - targets|
        l1_distance = torch.abs(outputs - targets)
        # Calculate loss only in valid regions
        masked_loss = l1_distance * mask
        # Calculate average loss
        valid_pixels = torch.sum(mask)
        if valid_pixels > 0:
            return torch.sum(masked_loss) / valid_pixels
        else:
            return torch.tensor(0.0, device=outputs.device)

    def forward(self, outputs, targets, mask_s30):
        """
        outputs: Tensor[B, C, H, W]
        targets: Tensor[B, C, H, W]
        mask_s30:Tensor[B, 1, H, W], mask=1 indicates valid pixels
        """
        # Expand mask to channel dimension
        mask_expand = mask_s30.expand_as(outputs)  # [B, C, H, W]
        
        # Use custom masked L1 loss
        main_loss = self.masked_l1_loss(outputs, targets, mask_expand)

        return main_loss

LOSS_CLASS = L1Loss