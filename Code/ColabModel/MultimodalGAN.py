import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

import os, sys
from pathlib import Path
# 1. Find the path where the current file is located
here = Path(__file__).resolve().parent  

# 2. Project root is one level up
project_root = here.parent

# 3. Insert project root into sys.path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def GenerateMask(x,dim=1):
    mask = (x != 0).float().sum(dim=dim)
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
            mask: Mask tensor of shape (B, 1, H, W) - Note: Must be image mask, not meta data!
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
            # Create non-zero value mask: positions where x != 0 are 1, x == 0 are 0
            non_zero_mask = (x != 0).float()
            # Combine original mask with non-zero mask: only positions where original mask is 1 and value is non-zero participate in calculation
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
        
        # —— Update running statistics —— 
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

    def forward(self, x, mask):
        """
        Args:
          x:    Tensor, shape (B, C, H, W)
          mask: Tensor, shape (B, 1, H, W), 0/1
        Returns:
          out:  Tensor, shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        N = H * W

        # 1×1 convolution + flatten
        q = self.query(x).view(B, -1, N).permute(0, 2, 1)  # (B, N, C_q)
        k = self.key(x).view(B, -1, N)                    # (B, C_q, N)
        v = self.value(x).view(B, -1, N)                  # (B, C,   N)

        # Flatten mask to (B, N), bool type
        mask_flat = mask.view(B, N).bool()                # (B, N)

        # Calculate attention logits: (B, N, N)
        attn_logits = torch.bmm(q, k)                     # (B, N, N)

        # Construct (B, N, N) mask_mat: True only when both i,j are valid pixels
        # First expand (B, N) to (B, N, 1) and (B, 1, N), then do element-wise AND
        mask_i = mask_flat.unsqueeze(2)   # (B, N, 1)
        mask_j = mask_flat.unsqueeze(1)   # (B, 1, N)
        mask_mat = mask_i & mask_j        # (B, N, N)；True only when both (i,j) are valid

        # For positions where mask_mat == False, set logits to -inf (or -1e9), so softmax weights become 0
        attn_logits = attn_logits.masked_fill(~mask_mat, float("-1e9"))

        # Apply softmax to get attention weights
        attn = F.softmax(attn_logits, dim=-1)  # (B, N, N)

        # Weight value with attention weights
        out = torch.bmm(v, attn.permute(0, 2, 1))  # (B, C, N)
        out = out.view(B, C, H, W)                # (B, C, H, W)

        # Residual connection
        out = self.gamma * out + x
        return out

class _MaskedConvBlock(nn.Module):
    """
    Convolution block with MaskedBatchNorm: Conv2d -> MaskedBatchNorm2d -> LeakyReLU.
    Properly handles mask downsampling.
    """
    def __init__(self, in_ch, out_ch, kernel_size=4, stride=2, padding=1, use_spectral_norm=False):
        super().__init__()
        conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.conv = spectral_norm(conv) if use_spectral_norm else conv
        self.bn = MaskedBatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.stride = stride

    def forward(self, x, mask):
        """
        Args:
            x: Tensor[B, in_ch, H, W]
            mask: Tensor[B, 1, H, W] (binary mask, 1 indicates valid pixels)
        Returns:
            out: Tensor[B, out_ch, H', W']
            out_mask: Tensor[B, 1, H', W'] where H'=H/stride, W'=W/stride
        """
        out = self.conv(x)
        
        # Adjust mask to match output spatial dimensions
        if out.shape[2:] != mask.shape[2:]:
            # Use adaptive pooling to match exact output size
            out_mask = F.adaptive_avg_pool2d(mask, output_size=out.shape[2:])
            # Binarize the mask after pooling
            out_mask = (out_mask > 0.5).float()
        elif self.stride > 1:
            out_mask = F.max_pool2d(mask, kernel_size=self.stride, stride=self.stride)
        else:
            out_mask = mask
            
        out = self.bn(out, out_mask)
        out = self.act(out)
        return out, out_mask


class _MaskedDeconvBlock(nn.Module):
    """
    Deconvolution block with MaskedBatchNorm: ConvTranspose2d -> MaskedBatchNorm2d -> ReLU.
    Includes optional dropout for regularization.
    """
    def __init__(self, in_ch, out_ch, kernel_size=4, stride=2, padding=1, dropout_rate=0.0):
        super().__init__()
        # self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        # bilinear interpolation
        self.upsample = nn.Upsample(scale_factor=stride,mode='bilinear',align_corners=False)
        # use 3×3 convolution for feature extraction
        self.post_conv = nn.Conv2d(in_ch, out_ch,kernel_size=3,stride=1,padding=1,bias=False)
        
        self.bn = MaskedBatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        self.stride = stride

    def forward(self, x, mask):
        """
        Args:
            x: Tensor[B, in_ch, h, w]
            mask: Tensor[B, 1, h, w]
        Returns:
            out: Tensor[B, out_ch, h*stride, w*stride]
            out_mask: Tensor[B, 1, h*stride, w*stride]
        """
        # out = self.deconv(x)
        # 1) Interpolation upsampling
        out = self.upsample(x)         # (B, in_ch, h*stride, w*stride)
        # 2) 3×3 convolution
        out = self.post_conv(out)      # (B, out_ch, h*stride, w*stride)
        
        # Upsample mask using nearest neighbor to maintain binary nature
        if mask is not None:
            out_mask = F.interpolate(mask, scale_factor=self.stride, mode='nearest')
            out = self.bn(out, out_mask)
        else:
            # If no mask provided, use ones
            out_mask = torch.ones((out.shape[0], 1, out.shape[2], out.shape[3]),
                                dtype=out.dtype, device=out.device)
            out = self.bn(out, out_mask)
            
        out = self.act(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
            
        return out, out_mask


class Generator(nn.Module):
    """
    Enhanced Multimodal U-Net Generator with attention and improved mask handling.
    """
    def __init__(self,
                 in_channels_l30=11,
                 in_channels_s1=3,
                 in_channels_planet=7,
                 meta_dim=11,
                 out_channels_s30=12,
                 ngf=64,
                 use_attention=True,
                 dropout_rate=0.2,
                 output_activation='sigmoid'):
        super().__init__()
        
        # Input validation
        assert output_activation in ['sigmoid', 'tanh', 'none'], \
            "output_activation must be 'sigmoid', 'tanh', or 'none'"
        
        self.concat_channels = in_channels_l30 + in_channels_s1 + in_channels_planet
        self.use_attention = use_attention
        
        # Meta information processing
        self.use_meta = (meta_dim > 0)
        if self.use_meta:
            self.meta_mlp = nn.Sequential(
                nn.Linear(3 * meta_dim, ngf * 8),
                nn.ReLU(inplace=True),
                # nn.Dropout(0.2),
                nn.Linear(ngf * 8, ngf * 8),
                nn.ReLU(inplace=True)
            )
        
        # Encoder
        self.enc1 = _MaskedConvBlock(self.concat_channels, ngf)
        self.enc2 = _MaskedConvBlock(ngf, ngf * 2)
        self.enc3 = _MaskedConvBlock(ngf * 2, ngf * 4)
        self.enc4 = _MaskedConvBlock(ngf * 4, ngf * 8)
        
        # Bottleneck with optional attention
        self.bottleneck_conv = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bottleneck_bn = MaskedBatchNorm2d(ngf * 8)
        self.bottleneck_act = nn.ReLU(inplace=True)
        
        if self.use_attention:
            self.attention = MaskedSelfAttention(ngf * 8)
        
        # Decoder with dropout in first few layers
        self.dec4 = _MaskedDeconvBlock(ngf * 8, ngf * 8, dropout_rate=dropout_rate)
        self.dec3 = _MaskedDeconvBlock(ngf * 8 * 2, ngf * 4, dropout_rate=dropout_rate)
        self.dec2 = _MaskedDeconvBlock(ngf * 4 * 2, ngf * 2, dropout_rate=dropout_rate)
        self.dec1 = _MaskedDeconvBlock(ngf * 2 * 2, ngf)
        
        ## Final convolution
        # self.final_conv = nn.ConvTranspose2d(
        #     ngf * 2, out_channels_s30, kernel_size=4, stride=2, padding=1, bias=True
        # )
        # 1) do bilinear interpolation upsampling ×2
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # 2) use a regular 3×3 convolution to restore channel count (keep bias=True)
        self.final_conv = nn.Conv2d(
            ngf * 2, out_channels_s30,
            kernel_size=3, stride=1, padding=1, bias=True
        )
        # Output activation
        if output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = nn.Identity()
    
    def forward(self,
                l30_img, mask_l30, l30_meta,
                s1_img, mask_s1, s1_meta,
                planet_img, mask_planet, planet_meta):
        """
        Forward pass with proper mask handling.
        
        Args:
            l30_img: Tensor[B, C_l30, H, W]
            mask_l30: Tensor[B, 1, H, W]
            l30_meta: Tensor[B, meta_dim]
            s1_img: Tensor[B, C_s1, H, W]
            mask_s1: Tensor[B, 1, H, W]
            s1_meta: Tensor[B, meta_dim]
            planet_img: Tensor[B, C_planet, H, W]
            mask_planet: Tensor[B, 1, H, W]
            planet_meta: Tensor[B, meta_dim]
            
        Returns:
            Tensor[B, C_s30, H, W]: Reconstructed S30 image
        """
        # Input validation
        B, _, H, W = l30_img.shape
        assert mask_l30.shape == (B, 1, H, W), f"mask_l30 shape mismatch: {mask_l30.shape}"
        assert s1_img.shape[2:] == (H, W), "All images must have same spatial dimensions"
        assert planet_img.shape[2:] == (H, W), "All images must have same spatial dimensions"
        
        # Concatenate modalities
        x = torch.cat([l30_img, s1_img, planet_img], dim=1)
        
        # Combine masks using element-wise maximum (logical OR for binary masks)
        mask_all = torch.max(torch.max(mask_l30, mask_s1), mask_planet)
        
        # Encoder
        e1, m1 = self.enc1(x, mask_all)
        e2, m2 = self.enc2(e1, m1)
        e3, m3 = self.enc3(e2, m2)
        e4, m4 = self.enc4(e3, m3)
        
        # Bottleneck
        b = self.bottleneck_conv(e4)
        m_b = F.max_pool2d(m4, kernel_size=2, stride=2)
        b = self.bottleneck_bn(b, m_b)
        b = self.bottleneck_act(b)
        
        # Apply attention if enabled
        if self.use_attention:
            b = self.attention(b, m_b)
        
        # Fuse meta information at bottleneck
        if self.use_meta:
            meta_all = torch.cat([l30_meta, s1_meta, planet_meta], dim=1)
            meta_feat = self.meta_mlp(meta_all)
            meta_feat = meta_feat.view(B, -1, 1, 1)
            b = b + meta_feat.expand_as(b)
        
        # Decoder with skip connections
        d4, md4 = self.dec4(b, m_b)
        d4 = torch.cat([d4, e4], dim=1)
        
        # Use maximum of upsampled mask and encoder mask
        m3_combined = torch.max(md4, m4)
        d3, md3 = self.dec3(d4, m3_combined)
        d3 = torch.cat([d3, e3], dim=1)
        
        m2_combined = torch.max(md3, m3)
        d2, md2 = self.dec2(d3, m2_combined)
        d2 = torch.cat([d2, e2], dim=1)
        
        m1_combined = torch.max(md2, m2)
        d1, md1 = self.dec1(d2, m1_combined)
        d1 = torch.cat([d1, e1], dim=1)
        
        # Final convolution and activation
        # out = self.final_conv(d1)
        out = self.upsample(d1)
        out = self.final_conv(out)
        out = self.output_activation(out)
        # out = out * mask_s30 
        return out


class Discriminator(nn.Module):
    """
    Enhanced PatchGAN Discriminator with spectral normalization and proper mask handling.
    """
    def __init__(self,
                 in_channels_l30=11,
                 in_channels_s1=3,
                 in_channels_planet=7,
                 in_channels_s30=12,
                 ndf=64,
                 dropout_rate=0.4,
                 use_spectral_norm=True):
        super().__init__()
        
        self.input_channels = in_channels_l30 + in_channels_s1 + in_channels_planet + in_channels_s30
        self.dropout1 = nn.Dropout(dropout_rate) 
        # First layer without batch norm
        conv1 = nn.Conv2d(self.input_channels, ndf, kernel_size=4, stride=2, padding=1, bias=True)
        self.layer1 = nn.Sequential(
            spectral_norm(conv1) if use_spectral_norm else conv1,
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Subsequent layers with masked batch norm
        self.layer2 = _MaskedConvBlock(ndf, ndf * 2, use_spectral_norm=use_spectral_norm)
        self.layer3 = _MaskedConvBlock(ndf * 2, ndf * 4, use_spectral_norm=use_spectral_norm)
        # Note: stride=1 with kernel=4, padding=1 reduces spatial size by 2
        self.layer4 = _MaskedConvBlock(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=1, use_spectral_norm=use_spectral_norm)
        
        # Final classification layer
        conv_final = nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.final_conv = spectral_norm(conv_final) if use_spectral_norm else conv_final
    
    def forward(self,
                l30_img, mask_l30,
                s1_img, mask_s1,
                planet_img, mask_planet,
                s30_img, mask_s30):
        """
        Forward pass returning patch-wise discrimination scores.
        
        Args:
            l30_img: Tensor[B, C_l30, H, W]
            mask_l30: Tensor[B, 1, H, W]
            s1_img: Tensor[B, C_s1, H, W]
            mask_s1: Tensor[B, 1, H, W]
            planet_img: Tensor[B, C_planet, H, W]
            mask_planet: Tensor[B, 1, H, W]
            s30_img: Tensor[B, C_s30, H, W] (real or generated)
            mask_s30: Tensor[B, 1, H, W]
            
        Returns:
            Tensor[B, 1, H', W']: Patch discrimination scores (logits)
        """
        # Concatenate all modalities
        x = torch.cat([l30_img, s1_img, planet_img, s30_img], dim=1)
        
        # Combine all masks
        # mask_all = torch.max(
        #     torch.max(torch.max(mask_l30, mask_s1), mask_planet),
        #     mask_s30
        # )
        mask_all = torch.max(torch.max(mask_l30, mask_s1), mask_planet)
        # Forward through layers
        x = self.layer1(x)
        x = self.dropout1(x)
        m1 = F.max_pool2d(mask_all, kernel_size=2, stride=2)
        
        x, m2 = self.layer2(x, m1)
        x = self.dropout1(x)
        x, m3 = self.layer3(x, m2)
        x = self.dropout1(x)
        x, m4 = self.layer4(x, m3)
        
        # Final classification
        out = self.final_conv(x)
        out = out * m4
        return out


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for improved training stability and quality.
    Uses multiple discriminators at different scales.
    """
    def __init__(self,
                 in_channels_l30=11,
                 in_channels_s1=3,
                 in_channels_planet=7,
                 in_channels_s30=12,
                 ndf=64,
                 num_scales=3,
                 dropout_rate=0.4,
                 use_spectral_norm=True):
        super().__init__()
        
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList()
        
        for i in range(num_scales):
            self.discriminators.append(
                Discriminator(
                    in_channels_l30=in_channels_l30,
                    in_channels_s1=in_channels_s1,
                    in_channels_planet=in_channels_planet,
                    in_channels_s30=in_channels_s30,
                    ndf=ndf,
                    dropout_rate=dropout_rate,
                    use_spectral_norm=use_spectral_norm
                )
            )
        
        # Downsampling layers for creating image pyramid
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self,
                l30_img,
                s1_img,
                planet_img,
                s30_img, mask_s30):
        """
        Forward pass through all scales.
        
        Returns:
            List[Tensor]: List of discrimination scores from each scale
        """
        outputs = []
        m_l30 = GenerateMask(l30_img)
        m_s1 = GenerateMask(s1_img)
        m_planet = GenerateMask(planet_img)
        m_s30=mask_s30

        # Current scale inputs
        l30 = l30_img
        s1 = s1_img
        planet = planet_img
        s30 = s30_img
        
        for i, disc in enumerate(self.discriminators):
            # Get discrimination scores at current scale
            out = disc(l30, m_l30, s1, m_s1, planet, m_planet, s30, m_s30)
            outputs.append(out)
            
            # Downsample for next scale (except for last discriminator)
            if i < self.num_scales - 1:
                l30 = self.downsample(l30)
                # m_l30 = self.downsample(m_l30)
                m_l30 = F.interpolate(m_l30, scale_factor=0.5, mode="nearest")
                
                s1 = self.downsample(s1)
                # m_s1 = self.downsample(m_s1)
                m_s1 = F.interpolate(m_s1, scale_factor=0.5, mode="nearest")
                
                planet = self.downsample(planet)
                # m_planet = self.downsample(m_planet)
                m_planet = F.interpolate(m_planet, scale_factor=0.5, mode="nearest")
                
                s30 = self.downsample(s30)
                # m_s30 = self.downsample(m_s30)
                m_s30 = F.interpolate(m_s30, scale_factor=0.5, mode="nearest")
        
        return outputs



# ------------------------------------------------------------------------------
# Wrapper class so that train.py can import MODEL_CLASS
# ------------------------------------------------------------------------------
class MultimodalGAN(nn.Module):
    """
    Wrapper for Generator and Discriminator.
    Exposes .generator and .discriminator attributes.
    """
    def __init__(self,
                 in_channels_l30=11,
                 in_channels_s1=3,
                 in_channels_planet=7,
                 meta_dim=11,
                 out_channels_s30=12,
                 ngf=64,
                 ndf=64,
                 use_attention=True,
                 dropout_rate=0.2,
                 use_spectral_norm=True,
                 num_scales=3):
        super().__init__()
        # Generator
        self.generator = Generator(
            in_channels_l30=in_channels_l30,
            in_channels_s1=in_channels_s1,
            in_channels_planet=in_channels_planet,
            meta_dim=meta_dim,
            out_channels_s30=out_channels_s30,
            ngf=ngf,
            use_attention=use_attention,
            dropout_rate=dropout_rate,
            output_activation='sigmoid'
        )
        # Discriminator (multi-scale)
        self.discriminator = MultiScaleDiscriminator(
            in_channels_l30=in_channels_l30,
            in_channels_s1=in_channels_s1,
            in_channels_planet=in_channels_planet,
            in_channels_s30=out_channels_s30,
            ndf=ndf,
            num_scales=num_scales,
            dropout_rate=dropout_rate,
            use_spectral_norm=use_spectral_norm
        )

    def forward(self, *args, **kwargs):
        """
        Not used directly. Use .generator(...) or .discriminator(...).
        """
        raise NotImplementedError("Use .generator() and .discriminator() separately.")

MODEL_CLASS=MultimodalGAN