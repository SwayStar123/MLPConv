import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple

class MLPConv2d(nn.Module):
    """
    A convolutional layer where each convolution operation is performed by an MLP.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        num_layers (int, optional): Number of layers in the MLP. Default: 2
        hidden_size (int, optional): Number of neurons in the hidden layers of the MLP. Default: out_channels
        activation (callable, optional): Activation function. Default: nn.ReLU
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 num_layers: int =2, 
                 hidden_size: Union[int, None] = None,
                 activation=nn.ReLU, 
                 bias: bool = True):
        super(MLPConv2d, self).__init__()
        
        # Ensure kernel_size, stride, padding, dilation are tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        if hidden_size is None:
            hidden_size = out_channels
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # (kh, kw)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Input feature size per patch
        self.input_size = in_channels * kernel_size[0] * kernel_size[1]
        
        # Build MLP
        layers = []
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if num_layers == 1:
            # Single linear layer
            layers.append(nn.Linear(self.input_size, out_channels, bias=bias))
        else:
            # Multiple layers: input -> hidden -> ... -> output
            layers.append(nn.Linear(self.input_size, hidden_size, bias=True))
            layers.append(activation())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_size, hidden_size, bias=True))
                layers.append(activation())
            layers.append(nn.Linear(hidden_size, out_channels, bias=bias))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass of the MLPConv2d.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, H_out, W_out)
        """
        batch_size, in_channels, H, W = x.shape
        if in_channels != self.in_channels:
            raise ValueError(f"Expected input with {self.in_channels} channels, got {in_channels}")
        
        # Use unfold to extract sliding local blocks
        # Output shape: (batch_size, in_channels * kernel_height * kernel_width, L)
        patches = F.unfold(x, 
                           kernel_size=self.kernel_size, 
                           dilation=self.dilation, 
                           padding=self.padding, 
                           stride=self.stride)  # shape: (B, C*K*K, L)
        
        # Transpose to (batch_size, L, C*K*K)
        patches = patches.transpose(1, 2)  # shape: (B, L, C*K*K)
        
        # Reshape to (B * L, C*K*K)
        patches = patches.reshape(-1, self.input_size)  # shape: (B*L, C*K*K)
        
        # Pass through MLP
        outputs = self.mlp(patches)  # shape: (B*L, out_channels)
        
        # Reshape to (B, L, out_channels)
        outputs = outputs.reshape(batch_size, -1, self.out_channels)  # shape: (B, L, out_channels)
        
        # Compute output spatial dimensions
        H_out = (H + 2 * self.padding[0] - self.dilation[0]*(self.kernel_size[0]-1) -1) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - self.dilation[1]*(self.kernel_size[1]-1) -1) // self.stride[1] + 1
        
        # Transpose to (B, out_channels, H_out, W_out)
        outputs = outputs.transpose(1, 2).contiguous().view(batch_size, self.out_channels, H_out, W_out)
        
        return outputs

# Test Case to Verify MLPConv2d behaves like Conv2d when num_layers=1
def test_mlpconv2d():
    batch_size = 2
    in_channels = 3
    out_channels = 4
    H, W = 8, 8
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    bias = True

    # Create random input
    x = torch.randn(batch_size, in_channels, H, W)

    # Initialize Conv2d
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    # Initialize MLPConv2d with num_layers=1
    mlp_conv = MLPConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, num_layers=1, bias=bias)

    # Copy Conv2d weights and bias to MLPConv2d
    with torch.no_grad():
        # MLPConv2d's first layer is the linear layer
        mlp_conv.mlp[0].weight.copy_(conv.weight.view(out_channels, -1))
        if bias:
            mlp_conv.mlp[0].bias.copy_(conv.bias)

    # Forward pass
    out_conv = conv(x)
    out_mlp_conv = mlp_conv(x)

    # Check if outputs are the same
    if torch.allclose(out_conv, out_mlp_conv, atol=1e-6):
        print("Test Passed: MLPConv2d with num_layers=1 matches Conv2d.")
    else:
        print("Test Failed: Outputs do not match.")
        print("Difference:", torch.abs(out_conv - out_mlp_conv).max())

# Example Usage
if __name__ == "__main__":
    # Run test case
    test_mlpconv2d()

    # Example of using MLPConv2d with num_layers > 1
    batch_size = 1
    in_channels = 3
    out_channels = 16
    H, W = 32, 32
    kernel_size = 3
    stride = 1
    padding = 1

    x = torch.randn(batch_size, in_channels, H, W)

    mlp_conv = MLPConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, num_layers=3, hidden_size=64)
    output = mlp_conv(x)
    print(f"Output shape with MLPConv2d: {output.shape}")  # Expected: (1, 16, 32, 32)
