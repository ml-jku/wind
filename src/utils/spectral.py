import torch
import torch.fft


def rapsd_torch(field):
    """
    Calculate radially averaged power spectral density for PyTorch tensors.

    Parameters
    ----------
    field : torch.Tensor
        Input tensor of shape (b, c, h, w) or (b, t, c, h, w)

    Returns
    -------
    rapsd : torch.Tensor
        RAPSD per sample:
        - If input is 4D: shape (b, c, n_bins)
        - If input is 5D: shape (b, t, c, n_bins)
    freq : torch.Tensor
        Frequency bins
    """
    match field.dim():
        case 5:
            is_5d = True
            b, t, c, h, w = field.shape
        case 4:
            is_5d = False
            b, c, h, w = field.shape
        case _:
            raise ValueError(f"Unsupported field dimension: {field.dim()}")

    if field.dtype != torch.float32:
        field = field.float()

    fft = torch.fft.fft2(field)
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    power = torch.abs(fft_shifted) ** 2 / (h * w)

    y = torch.arange(h, device=field.device) - h // 2
    x = torch.arange(w, device=field.device) - w // 2
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    r = torch.sqrt(xx**2 + yy**2).round().long()

    max_r = max(h, w) // 2
    if max_r % 2 == 1:
        max_r += 1

    if is_5d:
        rapsd = torch.zeros(b, t, c, max_r, device=field.device)
    else:
        rapsd = torch.zeros(b, c, max_r, device=field.device)

    for i in range(max_r):
        mask = r == i
        if mask.any():
            if is_5d:
                rapsd[:, :, :, i] = power[..., mask].mean(dim=-1)
            else:
                rapsd[:, :, i] = power[..., mask].mean(dim=-1)

    freq = torch.arange(max_r, device=field.device)
    return rapsd, freq
