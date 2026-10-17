import cv2
import numpy as np

def pro_rand_conv(image: np.ndarray, num_layers: int = 3, kernel_size: int = 3) -> np.ndarray:
    """
    Applies Progressive Random Convolutions (Pro-RandConv).
    Inspired by Qualcomm CVPR 2023.
    Recursively stacks small random convolutions to generate extreme style diversity
    while preserving object semantics. Used for Single Domain Generalization.
    """
    # Guard: only process 3-channel BGR images
    if image.ndim != 3 or image.shape[2] != 3:
        return image

    # Normalize image to float32 [0, 1]
    x = image.astype(np.float32) / 255.0
    
    for layer_i in range(num_layers):
        # 1. Random Convolution Weights (Gaussian distribution with He initialization scaling)
        # Using variance = 1.0 / (kernel_size^2 * channels)
        variance = 1.0 / (kernel_size**2 * 3)
        kernel = np.random.normal(0, np.sqrt(variance), (kernel_size, kernel_size))
        
        # 2. Apply convolution — mix with the identity (skip connection) to preserve
        # structural content.  alpha blends between the original and the random conv.
        conv_out = cv2.filter2D(x, -1, kernel)
        alpha = 0.5                  # 50 % identity, 50 % random conv
        x = alpha * x + (1 - alpha) * conv_out
        
        # 3. Contrast Diversification (Affine Transform)
        # gamma is a SCALE factor — must be centered at 1.0, not 0.0, otherwise it
        # destroys image content (50 % chance of color inversion, E[gamma]=0 collapses
        # all features to grey/black noise).  beta is a SHIFT factor centered at 0.
        gamma = np.random.normal(1.0, 0.3)
        beta  = np.random.normal(0.0, 0.1)
        
        # Fast spatial mean and std per channel using OpenCV (MUCH faster than numpy on CPU)
        mean, std = cv2.meanStdDev(x)
        mean = mean.reshape(1, 1, 3)
        std = std.reshape(1, 1, 3) + 1e-6
        
        # Apply affine transformation
        x = gamma * ((x - mean) / std) + beta
        
        # 4. Clip to valid range instead of tanh — tanh with the original
        # N(0,0.5) parameters was mapping everything into [-1,1] centered
        # at 0 (grey), destroying all intensity information after multiple layers.
        x = np.clip(x, -3.0, 3.0)
        
    # Scale back to uint8 [0, 255]
    # Re-normalize from standardized space back to [0, 255]
    mean_out, std_out = cv2.meanStdDev(x)
    mean_out = mean_out.reshape(1, 1, 3)
    std_out = std_out.reshape(1, 1, 3) + 1e-6
    x = (x - mean_out) / std_out          # zero-mean, unit-var
    x = x * 0.225 + 0.45                  # ~ ImageNet-like range
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return x
