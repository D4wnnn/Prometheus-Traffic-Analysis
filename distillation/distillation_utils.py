import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss Function
    Loss = alpha * CrossEntropy(Student, Truth) + (1-alpha) * KL_Div(Student, Teacher)
    """
    def __init__(self, alpha=0.5, temperature=3.0):
        """
        Args:
            alpha: Weight for Hard Label (Cross Entropy) loss.
            temperature: Distillation temperature; higher temperature smooths the softmax, 
                         focusing more on the distribution of negative classes.
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        """
        Args:
            student_logits: Output from Light Model.
            teacher_logits: Output from Heavy Model.
            labels: Ground truth labels.
        """
        # 1. Hard Label Loss (Cross Entropy)
        loss_ce = self.ce_loss(student_logits, labels)
        
        # 2. Soft Label Loss (KL Divergence)
        # Note: KLDivLoss expects input as LogSoftmax and target as Softmax (probability distribution).
        student_log_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        loss_kd = self.kl_div_loss(student_log_soft, teacher_soft) * (self.temperature ** 2)
        
        # 3. Combination
        total_loss = self.alpha * loss_ce + (1 - self.alpha) * loss_kd
        
        return total_loss, loss_ce.item(), loss_kd.item()