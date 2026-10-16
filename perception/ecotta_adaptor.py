import torch
import torch.nn as nn
import torch.optim as optim

class EcoTTAMetaNetwork(nn.Module):
    """
    Lightweight Meta-Network attached to the frozen base model.
    Contains only a BatchNorm and a 1x1 Convolution (inspired by EcoTTA, CVPR 2023).
    """
    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels, affine=True)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        # Initialize as an identity mapping to start from the source domain logic
        nn.init.dirac_(self.conv.weight)

    def forward(self, x):
        return self.conv(self.bn(x))

class EcoTTAWrapper(nn.Module):
    """
    Wraps a PyTorch segmentation model for memory-efficient Test-Time Adaptation.

    Inherits from nn.Module so it can be called like a normal model:
        output = wrapper(input_tensor)      # inference without adaptation
        output = wrapper.adapt_step(input)  # inference WITH one adaptation step
    """
    def __init__(self, base_model, channels=21, learning_rate=0.005):
        super().__init__()
        self.base_model = base_model
        
        # Freeze base model to save memory and prevent catastrophic forgetting
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # The meta network is the only part that trains during inference
        self.meta_net = EcoTTAMetaNetwork(channels)
        self.meta_net.eval()
        
        # Device placement
        device = next(self.base_model.parameters()).device
        self.meta_net = self.meta_net.to(device)
        
        self._lr = learning_rate
        self.optimizer = optim.SGD(self.meta_net.parameters(), lr=learning_rate)
        
        # Store initial state for reset on video loop
        self._initial_meta_state = {k: v.clone() for k, v in self.meta_net.state_dict().items()}

    def forward(self, input_tensor):
        """Standard forward pass — frozen base model + meta-network, no adaptation."""
        with torch.no_grad():
            base_outputs = self.base_model(input_tensor)
        if isinstance(base_outputs, dict):
            base_features = base_outputs['out']
        else:
            base_features = base_outputs
        adapted = self.meta_net(base_features)
        if isinstance(base_outputs, dict):
            return {'out': adapted}
        return adapted

    def adapt_step(self, input_tensor):
        """
        Perform one step of unsupervised entropy minimization on the meta-network
        and return the adapted predictions.
        """
        self.meta_net.train()
        self.optimizer.zero_grad()
        
        # Forward pass through frozen base model (no gradient tracking needed here)
        with torch.no_grad():
            base_outputs = self.base_model(input_tensor)
            
        # Handle dict outputs from torchvision models (like LRASPP)
        if isinstance(base_outputs, dict):
            base_features = base_outputs['out'].detach().requires_grad_(True)
        else:
            base_features = base_outputs.detach().requires_grad_(True)
            
        adapted_features = self.meta_net(base_features)
        
        # Unsupervised Entropy Minimization (EcoTTA main loss)
        probs = torch.softmax(adapted_features, dim=1)
        entropy_loss = -(probs * torch.log(probs + 1e-6)).sum(dim=1).mean()
        
        # Confidence gate: only adapt when the model is uncertain (entropy > 0.3).
        # This prevents the trivial solution collapse (all pixels → one class).
        if entropy_loss.item() > 0.3:
            entropy_loss.backward()
            # Gradient clipping to prevent divergence
            torch.nn.utils.clip_grad_norm_(self.meta_net.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        self.meta_net.eval()
        
        # Return dict if base model did, to maintain API compatibility
        if isinstance(base_outputs, dict):
            return {'out': adapted_features.detach()}
        return adapted_features.detach()

    def reset(self):
        """Reset meta-network weights + optimizer — call on video loop restart."""
        self.meta_net.load_state_dict(self._initial_meta_state)
        self.optimizer = optim.SGD(self.meta_net.parameters(), lr=self._lr)
