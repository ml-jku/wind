from math import ceil
from typing import List

import torch
from torch import Tensor


def expand_like(a: Tensor, b: Tensor, dim_insert_pos: int = -1) -> Tensor:
    """Function to reshape a to broadcastable dimension of b.

    Args:
      a (Tensor): [batch_dim, ...]
      b (Tensor): [batch_dim, ...]
      dim_insert_pos (int): Position where to insert new dimensions.
                           -1 (default) means at the end,
                           positive values insert at that position.

    Returns:
      Tensor: Reshaped tensor.
    """
    num_new_dims = len(b.size()) - len(a.size())
    if num_new_dims <= 0:
        return a

    # Handle negative indexing properly
    a_ndim = len(a.size())
    if dim_insert_pos < 0:
        dim_insert_pos = a_ndim + dim_insert_pos + 1

    # Create new shape by inserting 1s at the specified position
    new_shape = list(a.shape)
    for _ in range(num_new_dims):
        new_shape.insert(dim_insert_pos, 1)

    a = a.view(*new_shape)
    return a


def replace_with_mask(a: Tensor, b: Tensor, mask: Tensor) -> Tensor:
    """Replace a with b where mask is True."""
    return torch.where(expand_like(mask, a), a, b)


def context_from_mask(x: Tensor, context_mask: Tensor) -> Tensor:
    """Create new context from the mask and the condition."""
    return replace_with_mask(x, torch.zeros_like(x), context_mask)


def compute_n_forecasts(window_size: int, n_steps: int, n_cond_steps: int) -> int:
    if n_cond_steps >= window_size:
        raise ValueError(
            "n_cond_steps cannot be greater than or equal to window_size, "
            "because there is no prediction"
        )
    if n_steps <= window_size:
        return 1
    else:
        # We get n_cond_steps for free -> substract them in the first part
        # For the remaining steps we always get window_size - n_cond_steps per
        # forecast window
        return ceil((n_steps - n_cond_steps) / (window_size - n_cond_steps))


def compose_forecast(forecast_results: List[Tensor], target_length: int) -> Tensor:
    """Concatenate all forecast results and trim to target length."""
    concatenated = torch.cat(forecast_results, dim=1)
    return concatenated[:, :target_length]


def extract_frames(
    prediction: Tensor,
    is_first: bool = False,
    overlap: int = 0,
) -> List[Tensor]:
    """Add a forecast result, handling first vs subsequent forecast windows differently."""
    if is_first:
        return prediction.clone().detach()
    else:
        return prediction[:, overlap:].clone().detach()
