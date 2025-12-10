import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        
        # --- FIX 1: Prevent crash when indexing ignore_index (255) ---
        # We clamp 255 to 0 temporarily so one_hot doesn't crash.
        # We mask these out immediately after, so the value 0 doesn't affect the loss.
        targets_clamped = targets.clone()
        targets_clamped[targets == self.ignore_index] = 0
        
        true_1_hot = torch.eye(num_classes, device=logits.device)[targets_clamped]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        
        # Apply the mask to remove the ignored pixels we just clamped
        mask = (targets != self.ignore_index).unsqueeze(1)
        probs = probs * mask
        true_1_hot = true_1_hot * mask

        dims = (0, 2, 3)
        intersection = torch.sum(probs * true_1_hot, dims)
        cardinality = torch.sum(probs + true_1_hot, dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - torch.mean(dice_score)

class OHEMLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255, keep_ratio=0.25):
        super().__init__()
        self.ignore_index = ignore_index
        self.keep_ratio = keep_ratio
        # 'none' reduction allows us to sort pixels
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, labels):
        # Calculate loss per pixel (Shape: [Batch, H, W])
        pixel_losses = self.ce(logits, labels)
        
        # --- FIX 2: Filter out ignore_index BEFORE sorting ---
        # Otherwise, 0-loss pixels (ignored) might fill up the "keep_ratio" quota
        valid_mask = (labels != self.ignore_index)
        valid_losses = pixel_losses[valid_mask].view(-1)
        
        if valid_losses.numel() == 0:
            return pixel_losses.sum() * 0.0 # Return 0 with grad connection
            
        # Sort errors High -> Low
        sorted_losses, _ = torch.sort(valid_losses, descending=True)
        
        # Keep top 25% hardest valid pixels
        num_keep = int(valid_losses.size(0) * self.keep_ratio)
        num_keep = max(1, num_keep) # Ensure at least 1 pixel kept
        
        return sorted_losses[:num_keep].mean()

class CompoundLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255):
        super().__init__()
        self.ohem = OHEMLoss(num_classes, ignore_index, keep_ratio=0.25)
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits, labels):
        return 0.5 * self.ohem(logits, labels) + 0.5 * self.dice(logits, labels)