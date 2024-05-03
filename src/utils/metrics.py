import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=3.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        
        if torch.cuda.is_available():
            self.alpha = self.alpha.to(inputs.device)
        
        # Calculate Log Softmax
        log_softmax = F.log_softmax(inputs, dim=1)
        
        #print(targets.unsqueeze(1))
        # Gather the log probabilities by the actual class indices
        targets = torch.where(targets == -100, torch.tensor(0, device=targets.device), targets)
        gather = torch.gather(log_softmax, 1, targets.unsqueeze(1))

        
        # Calculate the focal loss
        focal_loss = -(1 - torch.exp(gather)) ** self.gamma * gather

        # Apply the alpha weighting
        if self.alpha is not None:
            at = self.alpha.gather(0, targets)  # This gathers the correct alpha per target
            at = at.unsqueeze(1)  # Ensure it has the shape [batch_size, 1] if not already
            focal_loss *= at
        
        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        
        return focal_loss
