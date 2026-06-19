# d:/CAPSTONE/Navigation/neurogait/communication/transfer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeTransfer:
    """
    Phase 3: Knowledge Transfer.
    Defines tools for policy distillation and domain adaptation transfers
    from source robot models (R1) to target robot architectures (R2).
    """
    def __init__(self, temperature=2.0, device="cuda:0"):
        self.temperature = temperature
        self.device = device
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def compute_distillation_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Computes soft-target distillation loss to transfer high-level navigation
        know-how between models.
        """
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        return loss

    def polyak_update(self, source_model: nn.Module, target_model: nn.Module, tau=0.005):
        """
        Performs soft parameter averaging (Polyak update) to transfer weights gradually.
        """
        with torch.no_grad():
            for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
                target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
                
        print(f"Executed Polyak updates on network weights (tau={tau})")
        return target_model
