"""
Multimodal Masked U-Net with three separate encoder branches.
Each branch input: image + mask + meta, finally fuse meta at Bottleneck, and combine skip connections from each branch in Decoder.
Enhanced version: Add optional Masked Self-Attention module at each decoder stage.

Input：
  - l30_img    (B, C_l30, H, W)
  - mask_l30   (B, 1, H, W)
  - l30_meta   (B, META_DIM)
  - s1_img     (B, C_s1, H, W)
  - mask_s1    (B, 1, H, W)
  - s1_meta    (B, META_DIM)
  - planet_img (B, C_planet, H, W)
  - mask_planet(B, 1, H, W)
  - planet_meta(B, META_DIM)

Output：
  - reconstructed s30_img (B, C_s30, H, W)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def GenerateMask(x,dim=1):
    mask = (x > 1e-6).float().sum(dim=dim)
    mask = mask.unsqueeze(1)
    return mask
class MaskedBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, 
                 track_running_stats=True, ignore_zeros=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.ignore_zeros = ignore_zeros
        
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
            mask: Mask tensor of shape (B, 1, H, W)
        """
        B, C, H, W = x.shape
        
        # —— Expand channels & binarize —— 
        if mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask.expand(B, C, H, W)
        elif mask.dim() == 4 and mask.shape[1] != C:
            raise ValueError(f"mask channel count {mask.shape[1]} is inconsistent with x's {C}!")
        
        # Key: treat all positions in mask > 0.5 as valid
        mask = (mask > 0.5).float()
        
        # —— If need to ignore zero values, create zero value mask —— 
        if self.ignore_zeros:
            non_zero_mask = (x != 0).float()
            effective_mask = mask * non_zero_mask
        else:
            effective_mask = mask
        
        if self.training:
            return self._forward_training(x, effective_mask, mask)
        else:
            return self._forward_inference(x, effective_mask, mask)
    
    def _forward_training(self, x, effective_mask, original_mask):
        B, C, H, W = x.shape
        
        # —— Calculate total valid pixels, Sum and Sum of Squares for each channel —— 
        # effective_mask is the actual calculation mask considering zero values
        valid_count = effective_mask.sum(dim=(0,2,3)).clamp(min=1.0) 
        
        sum_x  = (x * effective_mask).sum(dim=(0,2,3))       
        sum_x2 = (x * effective_mask * x).sum(dim=(0,2,3))  
        
        mean = sum_x / valid_count
        var  = sum_x2 / valid_count - mean * mean
        var  = var.clamp(min=0.0)
        
        # —— update running statistics —— 
        if self.track_running_stats:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
                self.num_batches_tracked += 1
        
        return self._normalize(x, effective_mask, original_mask, mean, var)
    
    def _forward_inference(self, x, effective_mask, original_mask):
        if self.track_running_stats:
            return self._normalize(x, effective_mask, original_mask, self.running_mean, self.running_var)
        else:
            return self._forward_training(x, effective_mask, original_mask)
    
    def _normalize(self, x, effective_mask, original_mask, mean, var):
        mean = mean.view(1, -1, 1, 1)
        var = var.view(1, -1, 1, 1)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply affine transformation
        if self.affine:
            weight = self.weight.view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)
            x_norm = x_norm * weight + bias
        
        # Apply normalization only to valid regions
        # Use effective_mask to ensure zero value positions remain zero
        output = x_norm * effective_mask
        
        return output

class _MaskedConvBlock(nn.Module):
    """
    Convolution block with MaskedBatchNorm: Conv2d -> MaskedBatchNorm2d -> ReLU.
    If stride>1, downsample mask; otherwise mask remains unchanged.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = MaskedBatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, mask):
        """
        x:    Tensor[B, in_ch, h, w]
        mask: Tensor[B, 1, h, w]
        returns：
          out:      Tensor[B, out_ch, h/stride, w/stride]
          out_mask: Tensor[B, 1, h/stride, w/stride]
        """
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        out = self.conv(x)
        if self.stride > 1:
            out_mask = F.interpolate(mask.float(), scale_factor=1.0 / self.stride, mode="nearest")
        else:
            out_mask = mask
        out = self.bn(out, out_mask)
        out = self.act(out)
        return out, out_mask

class _MaskedUpConvBlock(nn.Module):
    """
    Upsampling block with MaskedBatchNorm：Upsample -> Conv2d -> MaskedBatchNorm2d -> ReLU。
    """
    def __init__(self, in_ch, out_ch, mode='bilinear'):
        super().__init__()
        self.up_mode = mode
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = MaskedBatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        """
        x:    Tensor[B, in_ch, h, w]
        mask: Tensor[B, 1, h, w]
        returns：
          out:      Tensor[B, out_ch, h*2, w*2]
          out_mask: Tensor[B, 1, h*2, w*2]
        """
        out = F.interpolate(x, scale_factor=2.0, mode=self.up_mode, align_corners=False)
        out_mask = F.interpolate(mask.float(), scale_factor=2.0, mode="nearest")
        out = self.conv(out)
        out = self.bn(out, out_mask)
        out = self.act(out)
        return out, out_mask

class MaskedSelfAttention(nn.Module):
    """
    Self-Attention with mask: only attention between positions where mask=1.
    Input:
      x:     (B, C, H, W)
      mask:  (B, 1, H, W) binary mask (1 indicates valid, 0 indicates invalid)
    Output:
      out:   (B, C, H, W), with residual connection added
    """
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels,       1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = 1.0 / math.sqrt(in_channels // 8)

    def forward(self, x, mask):
        mask = (mask > 0.5).float()
        B, C, H, W = x.shape
        N = H * W
        
        # Calculate Q, K, V
        q = self.query(x).view(B, -1, N).permute(0, 2, 1)  # (B, N, C_q)
        k = self.key(x).view(B, -1, N)                      # (B, C_q, N)
        v = self.value(x).view(B, -1, N)                    # (B, C, N)
        
        # Build mask matrix
        mask_flat = mask.view(B, N).bool()                  # (B, N)
        
        # Calculate attention scores
        attn_logits = torch.bmm(q, k) * self.scale         # (B, N, N)
        
        # Handle invalid positions
        valid_positions = mask_flat.unsqueeze(2)            # (B, N, 1)
        # attn_logits = attn_logits.masked_fill(~mask_mat, float("-inf"))
        attn_logits = attn_logits.masked_fill(~valid_positions, 0)
        
        attn = F.softmax(attn_logits, dim=-1)              # (B, N, N)
        
        # Apply attention
        out = torch.bmm(v, attn.permute(0, 2, 1))          # (B, C, N)
        out = out.view(B, C, H, W)                          # (B, C, H, W)
        
        # Apply residual connection only at valid positions
        mask_ext = mask.expand(-1, C, -1, -1)
        out = self.gamma * out * mask_ext + x
        
        return out

class AdaptiveSkipFusion(nn.Module):
    """
    Adaptive weighted skip connection fusion module
    Fuses features from different sources by learning importance weights for each modality
    """
    def __init__(self, skip_dim, up_dim, out_dim, use_spatial_attention=True):
        super().__init__()
        self.use_spatial_attention = use_spatial_attention
        self.skip_dim = skip_dim
        # Channel attention weight generation network
        self.channel_attention = nn.ModuleDict({
            'l30': self._make_channel_attention(skip_dim),
            's1': self._make_channel_attention(skip_dim),
            'planet': self._make_channel_attention(skip_dim)
        })
        
        # Spatial attention - optional
        if use_spatial_attention:
            self.spatial_attention = nn.ModuleDict({
                'l30': self._make_spatial_attention(skip_dim),
                's1': self._make_spatial_attention(skip_dim),
                'planet': self._make_spatial_attention(skip_dim)
            })
            
        # Feature transformation convolution
        self.transform = nn.ModuleDict({
            'l30': self._make_transform(skip_dim),
            's1': self._make_transform(skip_dim),
            'planet': self._make_transform(skip_dim)
        })
        
        # Fusion convolution
        self.fusion = nn.Sequential(
            # nn.Conv2d(skip_dim*3 + up_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(skip_dim + up_dim, out_dim, kernel_size=3, padding=1, bias=False),
            MaskedBatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        
        # Weighted fusion metadata
        self.meta_weights = nn.Parameter(torch.ones(3) / 3)
    
    def _make_channel_attention(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )
    
    def _make_spatial_attention(self, dim):
        return nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=3, padding=1),
            MaskedBatchNorm2d(dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def _make_transform(self, dim):
        return nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            MaskedBatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
    
    def _apply_channel_attention(self, x, attention, mask):
        """Apply channel attention, considering mask"""
        B, C = x.shape[0], x.shape[1]
        
        # Calculate channel weights considering only valid regions
        valid_pixels = mask.sum(dim=(2, 3)).clamp(min=1.0)
        masked_x = x * mask
        avg_feat = masked_x.sum(dim=(2, 3)) / valid_pixels
        
        # Pass computed features directly to channel attention module
        channel_weights = attention(avg_feat).view(B, C, 1, 1)
        
        return x * channel_weights
    
    def _apply_spatial_attention(self, x, attention, mask):
        """Apply spatial attention, considering mask"""
        # Always use single-channel mask as base
        if mask.shape[1] > 1:
            mask_single = mask[:, :1]  # If multi-channel mask, take only first channel
        else:
            mask_single = mask
        
        # First convolutional layer (dim -> dim//4)
        first_conv = attention[0]
        out = first_conv(x)
        
        # Prepare mask with appropriate channel count for BatchNorm
        bn_mask = mask_single.expand(-1, out.shape[1], -1, -1)
        
        # BatchNorm
        bn_layer = attention[1]
        out = bn_layer(out, bn_mask)
        
        # Remaining layers (ReLU -> Conv -> Sigmoid)
        for i in range(2, len(attention)):
            out = attention[i](out)
        
        # Ensure output follows mask
        spatial_weights = out * mask_single
        
        return x * spatial_weights
    
    def forward(self, skip_l30, skip_mask_l30, skip_s1, skip_mask_s1, skip_planet, skip_mask_planet, 
                up_features, up_mask):

        # Transform features for each modality
        l30_feat = self._process_modality(skip_l30, skip_mask_l30, 'l30')
        s1_feat = self._process_modality(skip_s1, skip_mask_s1, 's1')
        planet_feat = self._process_modality(skip_planet, skip_mask_planet, 'planet')
        
        # Use learnable meta weights for modality weighting
        softmax_weights = F.softmax(self.meta_weights, dim=0)
        
        # Weighted fusion of skip connection features
        combined_skip = (
            l30_feat * softmax_weights[0] + 
            s1_feat * softmax_weights[1] + 
            planet_feat * softmax_weights[2]
        )
        # combined_skip = torch.cat([l30_feat * softmax_weights[0], 
        #                            s1_feat * softmax_weights[1], 
        #                            planet_feat * softmax_weights[2]], dim=1)
        
        # Concatenate weighted fused skip connections and upsampled features
        fusion_input = torch.cat([combined_skip, up_features], dim=1)
        fusion_mask = ((skip_mask_l30 + skip_mask_s1 + skip_mask_planet + up_mask) > 1e-6).float()
        # skip_mask_l30 = skip_mask_l30.expand(-1, l30_feat.size(1), -1, -1)
        # skip_mask_s1 = skip_mask_s1.expand(-1, s1_feat.size(1), -1, -1)
        # skip_mask_planet = skip_mask_planet.expand(-1, planet_feat.size(1), -1, -1)
        # if up_mask.size(1)!= up_features.size(1):
        #     up_mask = up_mask.expand(-1, up_features.size(1), -1, -1)
        # fusion_mask = torch.cat([skip_mask_l30, skip_mask_s1, skip_mask_planet,up_mask], dim=1)
        
        # Final fusion
        out = self.fusion[0](fusion_input)
        out = self.fusion[1](out, fusion_mask)
        out = self.fusion[2](out)
        
        return out, fusion_mask
    
    def _process_modality(self, features, mask, modality):
        """Process features for a single modality"""
        # Ensure mask dimensions are correct - note here we no longer expand to all channels
        # Each method will adjust mask channels as needed internally
        if mask.shape[1] != 1:
            mask = mask[:, :1]  #Ensure mask is single-channel
        
        # Apply channel attention
        weighted_feat = self._apply_channel_attention(
            features, self.channel_attention[modality], mask.expand_as(features))
        
        # Apply spatial attention (optional)
        if self.use_spatial_attention:
            weighted_feat = self._apply_spatial_attention(
                weighted_feat, self.spatial_attention[modality], mask)
        
        # Apply feature transformation - needs multi-channel mask
        out = weighted_feat
        transform_mask = mask.expand_as(features)
        
        for i, layer in enumerate(self.transform[modality]):
            if isinstance(layer, MaskedBatchNorm2d):
                out = layer(out, transform_mask)
            else:
                out = layer(out)
        
        # Ensure output conforms to mask
        out = out * transform_mask
        
        return out

class Unet(nn.Module):
    """
    Three-branch Encoder Multimodal Masked U-Net.
    Each Encoder branch processes l30, s1, planet inputs separately, then fuses at Bottleneck.
    Decoder uses Skip-first attention strategy: apply self-attention to skip connections first, then fuse with upsampled features.
    """
    def __init__(self,
                 in_channels_l30=11,
                 in_channels_s1=3,
                 in_channels_planet=7,
                 meta_dim=11,
                 out_channels_s30=12,
                 base_ch=64,
                 use_meta=True,
                 use_selfattention=True,
                 use_spatial_attention=True):
        """
        Args:
          in_channels_l30:   L30 image channel count (e.g., 11)
          in_channels_s1:    S1 image channel count (e.g., 3)
          in_channels_planet:Planet image channel count (e.g., 7)
          meta_dim:          Single modality meta dimension (e.g., 11)
          out_channels_s30:  Output S30 image channel count (e.g., 12)
          base_ch:           U-Net base feature channel count (e.g., 64)
          use_selfattention: Whether to use self-attention module
        """
        super().__init__()
        self.use_meta = use_meta
        self.use_selfattention = use_selfattention

        ##### ===== Build three branch Encoders ===== #####

        # --- Branch L30 ---
        self.enc1_l = _MaskedConvBlock(in_channels_l30, base_ch, stride=1, padding=1)    # (B, base_ch, H, W)
        self.down1_l = _MaskedConvBlock(base_ch, base_ch * 2, stride=2, padding=1)       # (B, base_ch*2, H/2, W/2)
        self.down2_l = _MaskedConvBlock(base_ch * 2, base_ch * 4, stride=2, padding=1)   # (B, base_ch*4, H/4, W/4)
        self.down3_l = _MaskedConvBlock(base_ch * 4, base_ch * 8, stride=2, padding=1)   # (B, base_ch*8, H/8, W/8)
        self.down4_l = _MaskedConvBlock(base_ch * 8, base_ch * 8, stride=2, padding=1)   # (B, base_ch*8, H/16, W/16)

        # --- Branch S1 ---
        self.enc1_s = _MaskedConvBlock(in_channels_s1, base_ch, stride=1, padding=1)
        self.down1_s = _MaskedConvBlock(base_ch, base_ch * 2, stride=2, padding=1)
        self.down2_s = _MaskedConvBlock(base_ch * 2, base_ch * 4, stride=2, padding=1)
        self.down3_s = _MaskedConvBlock(base_ch * 4, base_ch * 8, stride=2, padding=1)
        self.down4_s = _MaskedConvBlock(base_ch * 8, base_ch * 8, stride=2, padding=1)

        # --- Branch Planet ---
        self.enc1_p = _MaskedConvBlock(in_channels_planet, base_ch, stride=1, padding=1)
        self.down1_p = _MaskedConvBlock(base_ch, base_ch * 2, stride=2, padding=1)
        self.down2_p = _MaskedConvBlock(base_ch * 2, base_ch * 4, stride=2, padding=1)
        self.down3_p = _MaskedConvBlock(base_ch * 4, base_ch * 8, stride=2, padding=1)
        self.down4_p = _MaskedConvBlock(base_ch * 8, base_ch * 8, stride=2, padding=1)

        ##### ===== Bottleneck fusion and Meta ===== #####

        # For each branch's down4 output, do one more convolution + MaskedBN + ReLU
        self.bottle_conv_l = nn.Conv2d(base_ch * 8, base_ch * 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bottle_bn_l = MaskedBatchNorm2d(base_ch * 16)
        self.bottle_act_l = nn.ReLU(inplace=True)

        self.bottle_conv_s = nn.Conv2d(base_ch * 8, base_ch * 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bottle_bn_s = MaskedBatchNorm2d(base_ch * 16)
        self.bottle_act_s = nn.ReLU(inplace=True)

        self.bottle_conv_p = nn.Conv2d(base_ch * 8, base_ch * 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bottle_bn_p = MaskedBatchNorm2d(base_ch * 16)
        self.bottle_act_p = nn.ReLU(inplace=True)

        # If using meta, concatenate three meta vectors then project to base_ch*16 dimension
        if self.use_meta:
            self.meta_mlp_l = nn.Sequential(
                nn.Linear(meta_dim, base_ch * 16), nn.ReLU(inplace=True)
            )
            self.meta_mlp_s = nn.Sequential(
                nn.Linear(meta_dim, base_ch * 16), nn.ReLU(inplace=True)
            )
            self.meta_mlp_p = nn.Sequential(
                nn.Linear(meta_dim, base_ch * 16), nn.ReLU(inplace=True)
            )
        
        # Three branches' Bottleneck features concatenated have channels = base_ch*16 * 3
        self.fuse_conv = nn.Conv2d(base_ch * 16 * 3, base_ch * 16, kernel_size=1, bias=False)
        self.fuse_bn = MaskedBatchNorm2d(base_ch * 16)
        self.fuse_act = nn.ReLU(inplace=True)
        
        # Bottleneck self-attention (optional)
        if self.use_selfattention:
            self.bottleneck_attention = MaskedSelfAttention(base_ch * 16)

        ##### ===== Decoder with Skip-first Attention ===== #####

        # Level 4: Upsample + Skip-first attention fusion
        self.up4 = _MaskedUpConvBlock(base_ch * 16, base_ch * 8)
        self.dec4_fusion = AdaptiveSkipFusion(
            skip_dim=base_ch * 8,
            up_dim=base_ch * 8,
            out_dim=base_ch * 8,
            use_spatial_attention=use_spatial_attention
        )
        # Level 3: Upsample + Skip-first attention fusion
        self.up3 = _MaskedUpConvBlock(base_ch * 8, base_ch * 4)
        self.dec3_fusion = AdaptiveSkipFusion(
            skip_dim=base_ch * 4,
            up_dim=base_ch * 4,
            out_dim=base_ch * 4,
            use_spatial_attention=use_spatial_attention
        )
        # Level 2: Upsample + Skip-first attention fusion
        self.up2 = _MaskedUpConvBlock(base_ch * 4, base_ch * 2)
        self.dec2_fusion = AdaptiveSkipFusion(
            skip_dim=base_ch * 2,
            up_dim=base_ch * 2,
            out_dim=base_ch * 2,
            use_spatial_attention=use_spatial_attention
        )
        # Level 1: Upsample + Skip-first attention fusion
        self.up1 = _MaskedUpConvBlock(base_ch * 2, base_ch)
        self.dec1_fusion = AdaptiveSkipFusion(
            skip_dim=base_ch,
            up_dim=base_ch,
            out_dim=base_ch,
            use_spatial_attention=use_spatial_attention
        )
        # Final output layer
        self.final_conv = nn.Conv2d(base_ch, out_channels_s30, kernel_size=1, stride=1, padding=0)
        self.output_activation = nn.Sigmoid()

    def forward(self,
                l30_img, l30_meta,
                s1_img, s1_meta,
                planet_img, planet_meta):
        """
        Args:
          l30_img, mask_l30, l30_meta,
          s1_img,  mask_s1,  s1_meta,
          planet_img, mask_planet, planet_meta
        Returns:
          Tensor[B, C_s30, H, W]
        """
        # -------------------- Encoder branches --------------------
        mask_l30 = GenerateMask(l30_img,1)
        mask_s1 = GenerateMask(s1_img,1)
        mask_planet = GenerateMask(planet_img,1)
        # ==== Branch L30 ====
        e1_l, m1_l = self.enc1_l(l30_img, mask_l30)    # (B, base_ch, H, W), (B,1,H,W)
        d1_l, m2_l = self.down1_l(e1_l, m1_l)           # (B, base_ch*2, H/2, W/2), (B,1,H/2,W/2)
        d2_l, m3_l = self.down2_l(d1_l, m2_l)           # (B, base_ch*4, H/4, W/4), (B,1,H/4,W/4)
        d3_l, m4_l = self.down3_l(d2_l, m3_l)           # (B, base_ch*8, H/8, W/8), (B,1,H/8,W/8)
        d4_l, m5_l = self.down4_l(d3_l, m4_l)           # (B, base_ch*8, H/16, W/16), (B,1,H/16,W/16)

        # ==== Branch S1 ====
        e1_s, m1_s = self.enc1_s(s1_img, mask_s1)
        d1_s, m2_s = self.down1_s(e1_s, m1_s)
        d2_s, m3_s = self.down2_s(d1_s, m2_s)
        d3_s, m4_s = self.down3_s(d2_s, m3_s)
        d4_s, m5_s = self.down4_s(d3_s, m4_s)

        # ==== Branch Planet ====
        e1_p, m1_p = self.enc1_p(planet_img, mask_planet)
        d1_p, m2_p = self.down1_p(e1_p, m1_p)
        d2_p, m3_p = self.down2_p(d1_p, m2_p)
        d3_p, m4_p = self.down3_p(d2_p, m3_p)
        d4_p, m5_p = self.down4_p(d3_p, m4_p)

        # -------------------- Bottleneck + Meta fusion --------------------

        # Branch L30 Bottleneck
        b_l = self.bottle_conv_l(d4_l)         # (B, base_ch*16, H/16, W/16)
        b_l = self.bottle_bn_l(b_l, m5_l)
        b_l = self.bottle_act_l(b_l)

        # Branch S1 Bottleneck
        b_s = self.bottle_conv_s(d4_s)
        b_s = self.bottle_bn_s(b_s, m5_s)
        b_s = self.bottle_act_s(b_s)

        # Branch Planet Bottleneck
        b_p = self.bottle_conv_p(d4_p)
        b_p = self.bottle_bn_p(b_p, m5_p)
        b_p = self.bottle_act_p(b_p)

        # Fuse Bottleneck features from each branch
        b_cat = torch.cat([b_l, b_s, b_p], dim=1)  # (B, base_ch*16*3, H/16, W/16)

        # After fusing Bottleneck, do 1x1 conv + MaskedBN + ReLU
        # Merge mask: valid as long as any of the three branches' Bottleneck is valid
        m_b = ((m5_l + m5_s + m5_p) > 0.5).float()  # (B,1,H/16,W/16)
        b_fuse = self.fuse_conv(b_cat)            # (B, base_ch*16, H/16, W/16)
        b_fuse = self.fuse_bn(b_fuse, m_b)
        b_fuse = self.fuse_act(b_fuse)

        # If using meta, project each branch's meta with respective MLP then add
        if self.use_meta:
            # Project meta vectors with respective MLPs
            m_feat_l = self.meta_mlp_l(l30_meta).view(-1, b_fuse.shape[1], 1, 1)
            m_feat_s = self.meta_mlp_s(s1_meta).view(-1, b_fuse.shape[1], 1, 1)
            m_feat_p = self.meta_mlp_p(planet_meta).view(-1, b_fuse.shape[1], 1, 1)
            # Broadcast to spatial size H/16, W/16
            m_feat_l = m_feat_l.expand_as(b_fuse)
            m_feat_s = m_feat_s.expand_as(b_fuse)
            m_feat_p = m_feat_p.expand_as(b_fuse)
            b_fuse = b_fuse + m_feat_l + m_feat_s + m_feat_p  # (B, base_ch*16, H/16, W/16)
        
        # Bottleneck self-attention (optional)
        if self.use_selfattention:
            b_fuse = self.bottleneck_attention(b_fuse, m_b)
        
        # -------------------- Decoder with Skip-first Attention --------------------

        # -- Level 4 --
        u4, mu4 = self.up4(b_fuse, m_b)  
        c4, mc4 = self.dec4_fusion(
            d3_l, m4_l,       # L30 skip connection
            d3_s, m4_s,       # S1 skip connection
            d3_p, m4_p,       # Planet skip connection
            u4, mu4           # Upsampled features
        )
        # -- Level 3 --
        u3, mu3 = self.up3(c4, mc4)
        c3, mc3 = self.dec3_fusion(
            d2_l, m3_l,
            d2_s, m3_s,
            d2_p, m3_p,
            u3, mu3
        )
        # -- Level 2 --
        u2, mu2 = self.up2(c3, mc3)                              
        c2, mc2 = self.dec2_fusion(
            d1_l, m2_l,
            d1_s, m2_s,
            d1_p, m2_p,
            u2, mu2
        )
        # -- Level 1 --
        u1, mu1 = self.up1(c2, mc2)                              
        c1, mc1 = self.dec1_fusion(
            e1_l, m1_l,
            e1_s, m1_s,
            e1_p, m1_p,
            u1, mu1
        )
        # -- Final output --
        out = self.final_conv(c1)                                # (B, C_s30, H, W)
        out = self.output_activation(out)
        
        return out

MODEL_CLASS = Unet