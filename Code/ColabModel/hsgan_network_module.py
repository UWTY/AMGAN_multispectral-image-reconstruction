import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter


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
    
    def forward(self, x):
        B, C, H, W = x.shape
        mask = (x != 0).float() 
        if mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask.expand(B, C, H, W)
        elif mask.dim() == 4 and mask.shape[1] != C:
            raise ValueError(f"mask channel count {mask.shape[1]} is inconsistent with x's {C}!")
        # mask = (mask > 0.5).float()
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
        valid_count = effective_mask.sum(dim=(0,2,3)).clamp(min=1.0)
        sum_x  = (x * effective_mask).sum(dim=(0,2,3))
        sum_x2 = (x * effective_mask * x).sum(dim=(0,2,3))
        mean = sum_x / valid_count
        var  = sum_x2 / valid_count - mean * mean
        var  = var.clamp(min=0.0)
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
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            weight = self.weight.view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)
            x_norm = x_norm * weight + bias
        output = x_norm * effective_mask
        return output

class MaskedLayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(MaskedLayerNorm2d, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(1, normalized_shape[0], 1, 1))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W]
        mask = (x != 0).float()  # Mask zero pixels
        valid = mask.sum(dim=(2, 3), keepdim=True).clamp(min=1)
        mean = (x * mask).sum(dim=(2, 3), keepdim=True) / valid
        var = ((x * mask - mean) ** 2).sum(dim=(2, 3), keepdim=True) / valid
        x_norm = x / torch.sqrt(var + self.eps)
        x_norm = x_norm * self.weight
        x_norm = x_norm * mask  # Keep only valid regions
        return x_norm

class MaskedInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        """
        Custom InstanceNorm2d that automatically identifies valid regions
        :param num_features: Number of input channels
        :param eps: Avoid division by zero
        :param affine: Whether to use learnable affine transformation
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input):
        """
        Forward propagation
        :param input: Tensor [B, C, H, W]
        :return: Tensor [B, C, H, W]
        """
        B, C, H, W = input.size()

        # Automatically identify invalid regions: pixels where all channels are zero at a spatial location are invalid
        spatial_mask = (input.abs().sum(dim=1, keepdim=True) != 0).float()  # [B, 1, H, W]
        mask = spatial_mask.expand(B, C, H, W)  # [B, C, H, W]

        # Initialize output
        output = torch.zeros_like(input)

        for b in range(B):
            for c in range(C):
                # Current channel's features and mask
                channel_data = input[b, c]  # [H, W]
                channel_mask = mask[b, c]   # [H, W]

                # Consider only valid regions
                valid_pixels = channel_data[channel_mask == 1]

                if valid_pixels.numel() > 1:  # Need at least 2 valid pixels to calculate variance
                    # Calculate mean and variance
                    mean = torch.mean(valid_pixels)
                    var = torch.var(valid_pixels, unbiased=False)

                    # Normalize
                    normalized = (channel_data - mean) / torch.sqrt(var + self.eps)

                    # Update valid pixel locations
                    output[b, c] = torch.where(channel_mask == 1, normalized, channel_data)
                else:
                    # If insufficient valid pixels, keep original values
                    output[b, c] = channel_data

        # Apply affine transformation
        if self.affine:
            weight = self.weight.view(1, -1, 1, 1)  # [1, C, 1, 1]
            bias = self.bias.view(1, -1, 1, 1)      # [1, C, 1, 1]
            output = output * weight + bias

        return output

    def extra_repr(self):
        return f"{self.num_features}, eps={self.eps}, affine={self.affine}"
# ----------------------------------------
#               Conv2d Block
# ----------------------------------------
class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            # self.norm = nn.BatchNorm2d(out_channels)
            self.norm = MaskedBatchNorm2d(out_channels)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(out_channels)
            self.norm = MaskedInstanceNorm2d(out_channels)
        elif norm == 'ln':
            # self.norm = LayerNorm(out_channels)
            self.norm = MaskedLayerNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
    
    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
    
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, in_channels, latent_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False):
        super(ResidualDenseBlock_5C, self).__init__()
        # dense convolutions
        self.conv1 = Conv2dLayer(in_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
        self.conv2 = Conv2dLayer(in_channels + latent_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
        self.conv3 = Conv2dLayer(in_channels + latent_channels * 2, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
        self.conv4 = Conv2dLayer(in_channels + latent_channels * 3, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
        self.conv5 = Conv2dLayer(in_channels + latent_channels * 4, in_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)

    def forward(self, x):
        residual = x
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x5 = 0.1 * x5 + residual
        return x5

# ----------------------------------------
#           Spectral Norm Block
# ----------------------------------------
def l2normalize(v, eps = 1e-6):
    norm = torch.sqrt(torch.sum(v * v) + eps)
    return v / norm
# def l2normalize(v, eps = 1e-12):
#     return v / (v.norm() + eps)
class SpectralNorm(nn.Module):
    def __init__(self, module, name = 'weight', power_iterations = 1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

# ----------------------------------------
#              PixelShuffle
# ----------------------------------------
class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        if len(x.size()) != 4:
            raise ValueError("input tensor shape {} is not supported.".format(x.size()))
        N, C, H, W = x.size()
        c = C // (self.upscale_factor ** 2)
        h, w = H * self.upscale_factor, W * self.upscale_factor
        # (N, C, H, W) => (N, c, r, r, H, W)
        x = x.reshape(-1, c, self.upscale_factor,
                        self.upscale_factor, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(-1, c, h, w)
        return x

class PixelUnShuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        if len(x.size()) != 4:
            raise ValueError("input tensor shape {} is not supported.".format(x.size()))
        N, C, H, W = x.size()
        c = int(C * (self.downscale_factor ** 2))
        h, w = H // self.downscale_factor, W // self.downscale_factor
        x = x.reshape(-1, C, h, self.downscale_factor, w, self.downscale_factor)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(-1, c, h, w)
        return x