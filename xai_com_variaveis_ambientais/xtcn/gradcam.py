"""Grad-CAM acoplado a uma camada convolucional do X-TCN."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = target_layer.register_forward_hook(self._save_activations)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        self.activations = output

    def _save_gradients(
        self,
        _module: nn.Module,
        _grad_input: tuple,
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0]

    def __call__(
        self,
        inputs: torch.Tensor,
        class_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        logits = self.model(inputs)
        if class_indices is None:
            class_indices = logits.argmax(dim=1)
        scores = logits.gather(1, class_indices.view(-1, 1)).sum()
        self.model.zero_grad(set_to_none=True)
        scores.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Os hooks do Grad-CAM nao capturaram tensores.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
        minimum = cam.amin(dim=(2, 3), keepdim=True)
        maximum = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - minimum) / (maximum - minimum + 1e-8)
        return cam.detach(), logits.detach()

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

