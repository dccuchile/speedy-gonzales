import logging
from abc import ABC, abstractmethod

import torch

logger = logging.getLogger(__name__)


class KnowledgeDistilationLoss(torch.nn.Module):
    def __init__(self, loss_type: str, alpha: float, temperature: float) -> None:
        super(KnowledgeDistilationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.gt_component = torch.nn.CrossEntropyLoss()
        self.kd_component: _KDComponentWithSpecificLoss

        if loss_type == "ce":
            self.kd_component = _KDComponentWithCE(self.temperature)
        elif loss_type == "kldiv":
            self.kd_component = _KDComponentWithKLDiv(self.temperature)
        elif loss_type == "mse":
            self.kd_component = _KDComponentWithMSE()
        else:
            raise ValueError("loss_type should be ce, kldiv or mse")

        logger.info(f"Loss type: {loss_type}")

    def forward(
        self,
        student_logits: torch.Tensor,
        real_labels: torch.Tensor,
        teacher_logits: torch.Tensor | None = None,
    ):
        if teacher_logits is None or self.alpha == 1:
            return self.gt_component(student_logits, real_labels)
        elif self.alpha == 0:
            return self.kd_component(student_logits, teacher_logits)
        else:
            return self.alpha * self.gt_component(student_logits, real_labels) + (
                1 - self.alpha
            ) * self.kd_component(
                student_logits,
                teacher_logits,
            )


class _KDComponentWithSpecificLoss(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor):
        pass


class _KDComponentWithCE(_KDComponentWithSpecificLoss):
    def __init__(self, temperature: float) -> None:
        super().__init__()
        self.temperature = temperature
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ):

        return (
            self.ce_loss(
                student_logits / self.temperature,
                torch.nn.functional.softmax(teacher_logits / self.temperature, dim=1),
            )
            * self.temperature
            * self.temperature
        )


class _KDComponentWithKLDiv(_KDComponentWithSpecificLoss):
    def __init__(self, temperature: float) -> None:
        super().__init__()
        self.temperature = temperature
        self.kldiv_loss = torch.nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ):

        return (
            self.kldiv_loss(
                torch.nn.functional.log_softmax(
                    student_logits / self.temperature, dim=1
                ),
                torch.nn.functional.softmax(teacher_logits / self.temperature, dim=1),
            )
            * self.temperature
            * self.temperature
        )


class _KDComponentWithMSE(_KDComponentWithSpecificLoss):
    def __init__(self) -> None:
        super().__init__()
        self.mse_loss = torch.nn.MSELoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ):

        return self.mse_loss(
            student_logits,
            teacher_logits,
        )
