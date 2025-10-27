import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    
    def forward(self, x, mask):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            mask: Mask tensor of shape (B, 1, H, W) - Note: Must be image mask, not meta data!
        """
        # Check input dimensions
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B, C, H, W), got {x.dim()}D with shape {x.shape}")
        
        if mask.dim() != 4:
            raise ValueError(f"Expected 4D mask (B, 1, H, W) or (B, C, H, W), got {mask.dim()}D with shape {mask.shape}")
        
        # Ensure mask shape is correct
        B, C, H, W = x.shape
        if mask.shape[0] != B:
            raise ValueError(f"Batch size mismatch: x has {B}, mask has {mask.shape[0]}")
            
        if mask.shape[1] == 1:
            # (B, 1, H, W) -> expand to (B, C, H, W)
            mask = mask.expand(B, C, H, W)
        elif mask.shape[1] != C:
            raise ValueError(f"Channel mismatch: x has {C} channels, mask has {mask.shape[1]}")
        
        if mask.shape[2:] != (H, W):
            raise ValueError(f"Spatial size mismatch: x is {H}x{W}, mask is {mask.shape[2]}x{mask.shape[3]}")
        
        mask = mask.float()
        
        if self.training:
            return self._forward_training(x, mask)
        else:
            return self._forward_inference(x, mask)
    
    def _forward_training(self, x, mask):
        B, C, H, W = x.shape
        
        # Calculate statistics for each sample separately, then average across batch dimension
        batch_means = []
        batch_vars = []
        
        for b in range(B):
            # Current sample's data and mask
            x_b = x[b]  # (C, H, W)
            mask_b = mask[b]  # (C, H, W)
            
            sample_mean = torch.zeros(C, device=x.device, dtype=x.dtype)
            sample_var = torch.zeros(C, device=x.device, dtype=x.dtype)
            
            for c in range(C):
                # Valid pixels in current channel
                valid_mask = mask_b[c] > 0.5  # (H, W)
                valid_pixels = x_b[c][valid_mask]  # (N_valid,)
                
                if valid_pixels.numel() > 1:  # Need at least 2 pixels to calculate variance
                    sample_mean[c] = valid_pixels.mean()
                    sample_var[c] = valid_pixels.var(unbiased=False)
                elif valid_pixels.numel() == 1:
                    sample_mean[c] = valid_pixels[0]
                    sample_var[c] = 1.0  # Set variance to 1 for single pixel
                else:
                    sample_mean[c] = 0.0
                    sample_var[c] = 1.0
            
            batch_means.append(sample_mean)
            batch_vars.append(sample_var)
        
        # Average across batch dimension
        final_mean = torch.stack(batch_means).mean(dim=0)  # (C,)
        final_var = torch.stack(batch_vars).mean(dim=0)    # (C,)
        
        # Update running statistics
        if self.track_running_stats:
            with torch.no_grad():
                if self.num_batches_tracked == 0:
                    self.running_mean.copy_(final_mean)
                    self.running_var.copy_(final_var)
                else:
                    self.running_mean.mul_(1 - self.momentum).add_(final_mean, alpha=self.momentum)
                    self.running_var.mul_(1 - self.momentum).add_(final_var, alpha=self.momentum)
                self.num_batches_tracked += 1
        
        return self._normalize(x, mask, final_mean, final_var)
    
    def _forward_inference(self, x, mask):
        if self.track_running_stats and self.running_mean is not None:
            return self._normalize(x, mask, self.running_mean, self.running_var)
        else:
            return self._forward_training(x, mask)
    
    def _normalize(self, x, mask, mean, var):

        mean = mean.view(1, -1, 1, 1)
        var = var.view(1, -1, 1, 1)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply affine transformation
        if self.affine:
            weight = self.weight.view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)
            x_norm = x_norm * weight + bias
        
        # Apply normalization only to valid regions, keep original values in masked regions
        output = torch.where(mask > 0.5, x_norm, x)
        
        return output

# # Test function
# def test_masked_bn():
#     """Test the Masked BatchNorm implementation"""
#     print("Testing Masked BatchNorm...")
    
#     # Create test data
#     B, C, H, W = 2, 4, 8, 8
#     x = torch.randn(B, C, H, W)
    
#     # Create mask with some zeros (masked regions)
#     mask = torch.ones(B, 1, H, W)
#     mask[:, :, :2, :] = 0  # Mask out first 2 rows
#     mask[:, :, :, :2] = 0  # Mask out first 2 columns
    
#     # Test both implementations
#     mbn1 = MaskedBatchNorm2d(C)
#     mbn2 = MaskedBatchNorm2dEfficient(C)
    
#     # Training mode
#     mbn1.train()
#     mbn2.train()
    
#     out1 = mbn1(x, mask)
#     out2 = mbn2(x, mask)
    
#     print(f"Input shape: {x.shape}")
#     print(f"Mask shape: {mask.shape}")
#     print(f"Output1 shape: {out1.shape}")
#     print(f"Output2 shape: {out2.shape}")
    
#     # Check that masked regions remain unchanged
#     masked_regions = mask.expand_as(x) < 0.5
#     print(f"Original values preserved in masked regions: {torch.allclose(x[masked_regions], out1[masked_regions])}")
#     print(f"Efficient version matches: {torch.allclose(out1, out2, atol=1e-5)}")
    
#     # Test inference mode
#     mbn1.eval()
#     mbn2.eval()
    
#     with torch.no_grad():
#         out1_eval = mbn1(x, mask)
#         out2_eval = mbn2(x, mask)
    
#     print(f"Inference mode works: {out1_eval.shape == x.shape}")
#     print("Test completed!")

# if __name__ == "__main__":
#     test_masked_bn()