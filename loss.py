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
        true_1_hot = torch.eye(num_classes, device=logits.device)[targets]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        
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
        # Calculate loss per pixel
        pixel_losses = self.ce(logits, labels).view(-1)
        # Sort errors High -> Low
        sorted_losses, _ = torch.sort(pixel_losses, descending=True)
        # Keep top 25% hardest pixels
        num_keep = int(pixel_losses.size(0) * self.keep_ratio)
        return sorted_losses[:num_keep].mean() if num_keep > 0 else sorted_losses.mean()

class CompoundLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255):
        super().__init__()
        # OHEM (Hard Examples) + Dice (Shapes)
        self.ohem = OHEMLoss(num_classes, ignore_index, keep_ratio=0.25)
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits, labels):
        return 0.5 * self.ohem(logits, labels) + 0.5 * self.dice(logits, labels)