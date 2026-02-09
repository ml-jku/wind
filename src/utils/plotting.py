import os
from typing import Literal, Optional, Sequence, Tuple, Union
from warnings import warn

import cartopy.crs as ccrs
import matplotlib as mlp
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import plotly.graph_objects as go
import torch
from cartopy.util import add_cyclic_point
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Colormap
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
from scipy import fft
from torch import Tensor
from torchfsm.plot import plot_2D_field, sym_colormap


# Copied from torchfsm.plot v0.0.3
def _find_min_max(
    traj: np.ndarray,
    vmin: Union[float, Sequence[Optional[float]]],
    vmax: Union[float, Sequence[Optional[float]]],
):
    axis = tuple([0, 1] + [i + 3 for i in range(len(traj.shape) - 3)])
    vmins = np.min(traj, axis=axis)
    vmaxs = np.max(traj, axis=axis)
    if vmin is not None:
        if isinstance(vmin, float) or isinstance(vmin, int):
            vmin = [vmin] * len(vmins)
        elif len(vmin) != len(vmins):
            raise ValueError(
                "The number of vmin values should be equal to the number of channels in the input trajectory."
            )
        vmins = np.asarray(
            [vmin[i] if vmin[i] is not None else vmins[i] for i in range(len(vmins))]
        )
    if vmax is not None:
        if isinstance(vmax, float) or isinstance(vmax, int):
            vmax = [vmax] * len(vmaxs)
        elif len(vmax) != len(vmaxs):
            raise ValueError(
                "The number of vmax values should be equal to the number of channels in the input trajectory."
            )
        vmaxs = np.asarray(
            [vmax[i] if vmax[i] is not None else vmaxs[i] for i in range(len(vmaxs))]
        )
    return vmins, vmaxs


def rapsd(field, d=1.0):
    """
    Pure NumPy/SciPy replacement for pysteps.utils.spectral.rapsd
    Computes Radially Averaged Power Spectral Density.
    """
    h, w = field.shape
    # 1. Compute 2D FFT and shift the zero-frequency component to the center
    f_coeff = fft.fftshift(fft.fft2(field))

    # 2. Power Spectrum
    psd2d = np.abs(f_coeff) ** 2 / (h * w)

    # 3. Create a coordinate system of radial distances from the center
    y, x = np.indices(psd2d.shape)
    center = np.array([h // 2, w // 2])
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r = r.astype(int)

    # 4. Bin the power based on radial distance
    tbin = np.bincount(r.ravel(), psd2d.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / nr

    # 5. Get frequencies (optional, matches pysteps return)
    freqs = fft.fftfreq(max(h, w), d)[: len(radial_profile)]

    return radial_profile


def plot_raspsd(
    raspsd_pred: torch.Tensor,
    raspsd_target: torch.Tensor,
    channel: int,
    title: Optional[str] = None,
    pred_color: str = "blue",
    target_color: str = "red",
    line_width: int = 2,
) -> go.Figure:
    """Create an interactive plot of RAPSD (Radially Averaged Power Spectral Density)
    over time.

    Args:
        raspsd_pred: Predicted RAPSD values of shape (time_steps, channels, frequencies)
        raspsd_target: Target RAPSD values of shape (time_steps, channels, frequencies)
        channel: Channel index to plot
        title: Optional title for the plot. If None, will use default title with
            channel number
        pred_color: Color for predicted values
        target_color: Color for target values
        line_width: Width of the plotted lines

    Returns:
        plotly.graph_objects.Figure: Interactive plot with time slider
    """
    if isinstance(raspsd_pred, Tensor):
        raspsd_pred = raspsd_pred.cpu().numpy()
    if isinstance(raspsd_target, Tensor):
        raspsd_target = raspsd_target.cpu().numpy()

    time_steps = raspsd_pred.shape[0]
    d = raspsd_pred.shape[2]

    y_min = min(raspsd_pred[:, channel].min(), raspsd_target[:, channel].min())
    y_max = max(raspsd_pred[:, channel].max(), raspsd_target[:, channel].max())

    if y_min > 0 and y_max > 0:
        log_range = np.log10(y_max) - np.log10(y_min)
        y_min = 10 ** (np.log10(y_min) - 0.1 * log_range)
        y_max = 10 ** (np.log10(y_max) + 0.1 * log_range)

    fig = go.Figure()

    for t in range(time_steps):
        fig.add_trace(
            go.Scatter(
                x=list(range(d)),
                y=raspsd_pred[t, channel],
                name=f"predicted t={t}",
                visible=(t == 0),
                mode="lines",
                line=dict(width=line_width, color=pred_color),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(d)),
                y=raspsd_target[t, channel],
                name=f"target t={t}",
                visible=(t == 0),
                mode="lines",
                line=dict(width=line_width, color=target_color),
            )
        )

    steps = []
    for t in range(time_steps):
        step = dict(
            method="update",
            args=[{"visible": [False] * (2 * time_steps)}],
            label=f"t={t}",
        )
        step["args"][0]["visible"][2 * t] = True
        step["args"][0]["visible"][2 * t + 1] = True
        steps.append(step)

    # Update layout
    fig.update_layout(
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "Timestep: "},
                pad={"t": 50},
                steps=steps,
            )
        ],
        title=title or f"RAPSD over time - Channel {channel}",
        xaxis_title="frequency",
        yaxis_title="power",
        xaxis_type="log",
        yaxis_type="log",
        xaxis=dict(
            range=[np.log10(1), np.log10(d)],  # Fixed x-axis range in log scale
            autorange=False,
        ),
        yaxis=dict(
            range=[np.log10(y_min), np.log10(y_max)],  # Fixed y-axis range in log scale
            autorange=False,
        ),
        showlegend=True,
    )

    return fig


def plot_comparison_traj(
    forecast: Union[torch.Tensor, np.ndarray],  # B T C H W
    ground_truth: Union[torch.Tensor, np.ndarray],  # B T C H W
    residual_error: Optional[Union[torch.Tensor, np.ndarray]] = None,  # B T C H W
    fourth_plot: Optional[Union[torch.Tensor, np.ndarray]] = None,  # B T C H W
    fourth_plot_title: str = "",
    channel_names: Optional[Sequence[str]] = None,
    batch_names: Optional[Sequence[str]] = None,
    vmin: Union[float, Sequence[Optional[float]]] = None,
    vmax: Union[float, Sequence[Optional[float]]] = None,
    subfig_size: float = 3.5,
    x_space: float = 0.7,
    y_space: float = 0.1,
    cbar_pad: float = 0.1,
    aspect: Literal["auto", "equal"] = "auto",
    num_colorbar_value: int = 4,
    ctick_format: Optional[str] = "%.1f",
    show_ticks: Union[Literal["auto"], bool] = "auto",
    show_time_index: bool = True,
    use_sym_colormap: bool = True,
    cmap: Union[str, Colormap] = "coolwarm",
    error_cmap: Union[str, Colormap] = "coolwarm",
    ticks_t: Tuple[Sequence[float], Sequence[str]] = None,
    ticks_x: Tuple[Sequence[float], Sequence[str]] = None,
    ticks_y: Tuple[Sequence[float], Sequence[str]] = None,
    fps: int = 30,
    show_in_notebook: bool = True,
    animation_engine: Literal["jshtml", "html5"] = "html5",
    save_name: Optional[str] = None,
    **kwargs,
):
    """
    Plot comparison trajectory with forecast, ground truth, and residual error.
    Inspired from torchfsm.plot.plot_traj.

    For each channel:
    - B rows (one per batch)
    - 3 or 4 columns: forecast, ground truth, residual error, (optional) fourth plot
    """

    # Convert to numpy if needed
    if isinstance(forecast, torch.Tensor):
        forecast = forecast.cpu().detach().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().detach().numpy()

    # Compute residual error if not provided
    if residual_error is None:
        residual_error = ground_truth - forecast
    elif isinstance(residual_error, torch.Tensor):
        residual_error = residual_error.cpu().detach().numpy()

    # Process fourth plot if provided
    if fourth_plot is not None and isinstance(fourth_plot, torch.Tensor):
        fourth_plot = fourth_plot.cpu().detach().numpy()

    # Determine number of columns based on fourth_plot
    n_cols = 4 if fourth_plot is not None else 3

    # Combine all data for consistent processing
    # Stack along a new dimension: [3, B, T, C, H, W]
    # all_data = np.stack([forecast, ground_truth, residual_error], axis=0)

    # Reshape to match plot_traj expected format: [(B*3), T, C, H, W]
    n_batch, n_frame, n_channel = (
        forecast.shape[0],
        forecast.shape[1],
        forecast.shape[2],
    )
    n_dim = len(forecast.shape) - 3

    if n_dim != 2:
        raise ValueError("This function only supports 2D data")

    # Create extended batch names for the grid
    extended_batch_names = []
    for c in range(n_channel):
        for b in range(n_batch):
            if fourth_plot is not None:
                extended_batch_names.extend(
                    [
                        batch_names[b] if batch_names else f"batch {b}",
                        "",  # No label for ground truth column
                        "",  # No label for error column
                        "",  # No label for fourth plot column
                    ]
                )
            else:
                extended_batch_names.extend(
                    [
                        batch_names[b] if batch_names else f"batch {b}",
                        "",  # No label for ground truth column
                        "",  # No label for error column
                    ]
                )

    # Default channel names
    channel_names = channel_names or [f"channel {i}" for i in range(n_channel)]

    if len(channel_names) != n_channel:
        raise ValueError(
            "The number of channel names should be equal to the number of channels"
        )

    # Find min/max for forecast and ground truth (they should share the same scale)
    combined_data = np.concatenate([forecast, ground_truth], axis=1)
    vmins, vmaxs = _find_min_max(combined_data, vmin, vmax)

    # Find min/max for error (separate scale)
    error_vmins, error_vmaxs = _find_min_max(residual_error, vmin, vmax)

    # Find min/max for fourth plot if provided
    if fourth_plot is not None:
        fourth_vmins, fourth_vmaxs = _find_min_max(fourth_plot, vmin, vmax)
        # When fourth plot is provided, use its colorscale for residual error
        error_vmins = fourth_vmins.copy()
        error_vmaxs = fourth_vmaxs.copy()
    else:
        # Make error symmetric around 0 only when fourth plot is not provided
        for i in range(n_channel):
            max_abs = max(abs(error_vmins[i]), abs(error_vmaxs[i]))
            error_vmins[i] = -max_abs
            error_vmaxs[i] = max_abs

    # Create colormaps exactly like the original plot_traj
    cmaps = [
        sym_colormap(vmins[i], vmaxs[i], cmap=cmap) if use_sym_colormap else cmap
        for i in range(n_channel)
    ]
    # Create colormaps for fourth plot if provided
    if fourth_plot is not None:
        fourth_cmaps = [
            (
                sym_colormap(fourth_vmins[i], fourth_vmaxs[i], cmap=cmap)
                if use_sym_colormap
                else cmap
            )
            for i in range(n_channel)
        ]
        # Use the same colormaps for error when fourth plot is provided
        error_cmaps = fourth_cmaps
    else:
        # For error, we usually want symmetric colormaps centered at 0
        error_cmaps = [
            sym_colormap(error_vmins[i], error_vmaxs[i], cmap=error_cmap)
            for i in range(n_channel)
        ]

    if show_ticks == "auto":
        show_ticks = False

    # Calculate subfigure dimensions
    subfig_h = subfig_size
    subfig_w = subfig_size * forecast.shape[-2] / forecast.shape[-1]

    # Create separate figure with subplots for each channel
    # Each channel gets B rows × 3 columns
    # Increase spacing to prevent label overlap
    increased_y_space = y_space * 2.5  # More space between rows
    channel_separation = 1.2  # More space between channels

    fig_height = (
        subfig_h * n_batch * n_channel
        + increased_y_space * (n_batch - 1) * n_channel
        + channel_separation * (n_channel - 1)
        + 1.5
    )
    fig_width = (
        subfig_w * n_cols + x_space * (n_cols - 1) + 1.8
    )  # Extra space for colorbars

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Store all axes and grids
    all_grids = []

    # Create a separate ImageGrid for each channel
    for c in range(n_channel):
        # Calculate position for this channel's grid with better spacing
        grid_height = n_batch * subfig_h + (n_batch - 1) * increased_y_space
        total_used = c * (grid_height + channel_separation)
        bottom = 1 - (total_used + grid_height + 0.8) / fig_height
        height = grid_height / fig_height

        grid = ImageGrid(
            fig,
            [0.08, bottom, 0.75, height],
            nrows_ncols=(n_batch, n_cols),
            axes_pad=(x_space, increased_y_space),
            share_all=True,
            cbar_location="right",
            cbar_mode="edge",
            direction="row",
            cbar_pad=cbar_pad,
            aspect=True,
        )
        all_grids.append(grid)

    def set_colorbar():
        """Set colorbars for each channel grid"""
        for c, grid in enumerate(all_grids):
            # Data colorbar (for forecast and ground truth)
            cb = grid.cbar_axes[0].colorbar(
                mlp.cm.ScalarMappable(
                    colors.Normalize(vmin=vmins[c], vmax=vmaxs[c]), cmap=cmaps[c]
                ),
                ticklocation="right",
                label=channel_names[c],
                format=ctick_format,
            )
            cb.ax.minorticks_on()
            cb.set_ticks(
                np.linspace(vmins[c], vmaxs[c], num_colorbar_value, endpoint=True)
            )

            # Error colorbar
            cb_err = grid.cbar_axes[1].colorbar(
                mlp.cm.ScalarMappable(
                    colors.Normalize(vmin=error_vmins[c], vmax=error_vmaxs[c]),
                    cmap=error_cmaps[c],
                ),
                ticklocation="right",
                label=f"{channel_names[c]} (Error)",
                format=ctick_format,
            )
            cb_err.ax.minorticks_on()
            cb_err.set_ticks(
                np.linspace(
                    error_vmins[c], error_vmaxs[c], num_colorbar_value, endpoint=True
                )
            )

            # Fourth plot colorbar if provided
            if fourth_plot is not None and len(grid.cbar_axes) > 2:
                cb_fourth = grid.cbar_axes[2].colorbar(
                    mlp.cm.ScalarMappable(
                        colors.Normalize(vmin=fourth_vmins[c], vmax=fourth_vmaxs[c]),
                        cmap=fourth_cmaps[c],
                    ),
                    ticklocation="right",
                    label=(
                        f"{channel_names[c]} ({fourth_plot_title})"
                        if fourth_plot_title
                        else channel_names[c]
                    ),
                    format=ctick_format,
                )
                cb_fourth.ax.minorticks_on()
                cb_fourth.set_ticks(
                    np.linspace(
                        fourth_vmins[c],
                        fourth_vmaxs[c],
                        num_colorbar_value,
                        endpoint=True,
                    )
                )

    def title_t(i):
        if show_time_index:
            if ticks_t is not None:
                if i in ticks_t[0]:
                    fig.suptitle(f"t={ticks_t[1][i]}", fontsize=14, y=0.98)
            else:
                fig.suptitle(f"t={i}", fontsize=14, y=0.98)

    # Add column headers
    if fourth_plot is not None:
        column_headers = [
            "Forecast",
            "Ground Truth",
            "Residual Error",
            fourth_plot_title if fourth_plot_title else "Fourth Plot",
        ]
    else:
        column_headers = ["Forecast", "Ground Truth", "Residual Error"]

    def ani_func(frame_idx):
        for c, grid in enumerate(all_grids):
            for b in range(n_batch):
                # Calculate axes indices for this batch
                base_idx = b * n_cols

                # Forecast
                ax = grid[base_idx]
                ax.clear()

                # Add channel name to first row of each channel
                if b == 0:
                    title = f"{channel_names[c]}\n{column_headers[0]}"
                else:
                    title = None

                plot_2D_field(
                    ax=ax,
                    data=forecast[b, frame_idx, c, ...],
                    show_ticks=show_ticks,
                    x_label="x" if (c == n_channel - 1 and b == n_batch - 1) else None,
                    y_label=batch_names[b] if batch_names and n_batch > 1 else "y",
                    title=title,
                    cmap=cmaps[c],
                    vmin=vmins[c],
                    vmax=vmaxs[c],
                    ticks_x=ticks_x,
                    ticks_y=ticks_y,
                    aspect=aspect,
                    **kwargs,
                )

                # Ground Truth
                ax = grid[base_idx + 1]
                ax.clear()

                title = column_headers[1] if b == 0 else None

                plot_2D_field(
                    ax=ax,
                    data=ground_truth[b, frame_idx, c, ...],
                    show_ticks=show_ticks,
                    x_label="x" if (c == n_channel - 1 and b == n_batch - 1) else None,
                    y_label=None,
                    title=title,
                    cmap=cmaps[c],
                    vmin=vmins[c],
                    vmax=vmaxs[c],
                    ticks_x=ticks_x,
                    ticks_y=ticks_y,
                    aspect=aspect,
                    **kwargs,
                )

                # Residual Error
                ax = grid[base_idx + 2]
                ax.clear()

                title = column_headers[2] if b == 0 else None

                plot_2D_field(
                    ax=ax,
                    data=residual_error[b, frame_idx, c, ...],
                    show_ticks=show_ticks,
                    x_label=(
                        "x"
                        if (
                            c == n_channel - 1
                            and b == n_batch - 1
                            and fourth_plot is None
                        )
                        else None
                    ),
                    y_label=None,
                    title=title,
                    cmap=error_cmaps[c],
                    vmin=error_vmins[c],
                    vmax=error_vmaxs[c],
                    ticks_x=ticks_x,
                    ticks_y=ticks_y,
                    aspect=aspect,
                    **kwargs,
                )

                # Fourth plot if provided
                if fourth_plot is not None:
                    ax = grid[base_idx + 3]
                    ax.clear()

                    title = column_headers[3] if b == 0 else None

                    plot_2D_field(
                        ax=ax,
                        data=fourth_plot[b, frame_idx, c, ...],
                        show_ticks=show_ticks,
                        x_label=(
                            "x" if (c == n_channel - 1 and b == n_batch - 1) else None
                        ),
                        y_label=None,
                        title=title,
                        cmap=fourth_cmaps[c],
                        vmin=fourth_vmins[c],
                        vmax=fourth_vmaxs[c],
                        ticks_x=ticks_x,
                        ticks_y=ticks_y,
                        aspect=aspect,
                        **kwargs,
                    )

        set_colorbar()
        title_t(frame_idx)

    # Create animation
    if n_frame != 1:
        ani = FuncAnimation(
            fig, ani_func, frames=n_frame, repeat=False, interval=1000 / fps
        )
        if show_in_notebook:
            plt.close()
            if animation_engine == "jshtml":
                return HTML(ani.to_jshtml())
            elif animation_engine == "html5":
                try:
                    return HTML(ani.to_html5_video())
                except Exception as e:
                    warn_msg = (
                        "Error occurs when generating html5 video, use jshtml instead."
                        + os.linesep
                    )
                    warn_msg += f"Error message: {e}" + os.linesep
                    warn_msg += (
                        "This is probably due to the `ffmpeg` is not properly "
                        "installed." + os.linesep
                    )
                    warn_msg += "Please install `ffmpeg` and try again." + os.linesep
                    warn(warn_msg)
                    return HTML(ani.to_jshtml())
            else:
                raise ValueError("The animation engine should be 'jshtml' or 'html5'.")
        else:
            return ani
    else:
        ani_func(0)
        if save_name is not None:
            plt.savefig(save_name)
        plt.show()


# --------------------------
# ERA5 related plotting functions
# --------------------------


def spatial_spectral_density_multi(
    *datasets,
    labels=None,
    resolution_deg_per_px=0.25,  # same resolution for all inputs
    channel=0,  # which channel to use if data has (T,C,Y,X)
    normalize=True,
    axis=None,
    fontsize=18,
    linewidth=None,
    legend=True,
    title=None,
    x_ax_name="Wavelength [km]",
    y_ax_name="PSD [a.u.]",
    do_savefig=False,
    xlog_base=2,
    ylog_base=10,
):
    """
    Plot mean radial PSD for an arbitrary number of datasets on the same grid.

    Each dataset is expected to be an ndarray shaped like:
        (time, channel, y, x)  or  (time, y, x)
    Only 'channel' index is used if present.

    Parameters
    ----------
    *datasets : np.ndarray
        Any number of arrays with identical spatial shape (Y, X) and same res.
    labels : list[str] | None
        Optional names for each dataset; defaults to Series 1..N.
    resolution_deg_per_px : float
        Grid spacing in degrees per pixel (same for all datasets).
    channel : int
        Channel index to select when data has 4D shape.
    normalize : bool
        Whether to normalize PSD in `rapsd`.
    axis : matplotlib.axes.Axes | None
        Axes to plot on; if None, a new one is created.
    linewidth : float | None
        Line width passed to ax.plot.
    legend : bool
        Show legend.
    title, x_ax_name, y_ax_name : str
        Text for the plot.
    do_savefig : str | False
        If str, path to save figure.

    Returns
    -------
    result : dict
        {
          "ax": Axes,
          "freq": freq array,
          "wavelength_km": wavelength array,
          "psds": [np.ndarray, ...],  # one per dataset
        }
    """
    if len(datasets) == 0:
        raise ValueError("Provide at least one dataset.")

    # Standardize shapes and basic checks
    def _select_field(arr, t):
        if arr.ndim == 4:
            return arr[t, channel, :, :]
        elif arr.ndim == 3:
            return arr[t, :, :]
        else:
            raise ValueError("Each dataset must be (T,C,Y,X) or (T,Y,X).")

    # Compute mean PSDs
    psds = []
    freq_ref = None
    for idx, data in enumerate(datasets):
        T = data.shape[0]
        acc = None
        fr = None
        for t in range(T):
            field = _select_field(data, t)
            psd, freq = rapsd(
                field, return_freq=True, normalize=normalize, fft_method=np.fft
            )
            if acc is None:
                acc = np.zeros_like(psd)
                fr = freq
            acc += psd
        mean_psd = acc / T
        psds.append(mean_psd)
        if freq_ref is None:
            freq_ref = fr
        else:
            # sanity check: same freq length (implies same grid size)
            if len(fr) != len(freq_ref):
                raise ValueError(
                    "All datasets must share the same spatial shape to compare PSDs."
                )

    # Convert frequency to wavelength in km (1 deg ≈ 111 km, /2 as in your original)
    wavelength_km = 1.0 / freq_ref * resolution_deg_per_px * 111.0 / 2.0

    # Labels
    if labels is None:
        labels = [f"Series {i+1}" for i in range(len(datasets))]
    elif len(labels) != len(datasets):
        raise ValueError("Length of 'labels' must match number of datasets.")

    # Plot (let Matplotlib use its default color cycle)
    ax = axis if axis is not None else plt.subplots()[1]
    for psd, lab in zip(psds, labels):
        ax.plot(wavelength_km, psd, label=lab, linewidth=linewidth)

    ax.set_yscale("log", base=ylog_base)
    ax.set_xscale("log", base=xlog_base)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel(x_ax_name, fontsize=fontsize)
    ax.set_ylabel(y_ax_name, fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize)
    if legend:
        ax.legend()

    if do_savefig:
        plt.savefig(do_savefig)

    return {
        "ax": ax,
        "freq": freq_ref,
        "wavelength_km": wavelength_km,
        "psds": psds,
    }


# --------------------------
# LATITUDINAL MEAN (→ function of LONGITUDE)
# --------------------------
def latitudinal_mean_multi(
    *datasets,
    labels=None,
    lon=None,  # optional 1D longitude array (nlon,)
    lon_resolution_deg=None,  # or supply resolution in degrees/pixel
    lon_start_deg=-180.0,  # used if lon is not provided
    axis=None,
    linewidth=None,
    legend=True,
    title="Latitudinal mean",
    x_ax_name="Longitude [deg]",
    y_ax_name="Mean value",
    fontsize=18,
    do_savefig=False,
):
    """
    Computes the latitudinal mean (i.e., mean over latitude), returning a
    series vs. LONGITUDE for any number of datasets.
    """
    if len(datasets) == 0:
        raise ValueError("Provide at least one dataset.")

    # Reduce to (nlon,) by averaging over time, channel, and latitude.
    def _latitudinal_mean(a):
        if a.ndim == 4:  # (T,C,Y,X)
            return a.mean(axis=(0, 1, 2))
        elif a.ndim == 3:  # (T,Y,X)
            return a.mean(axis=(0, 1))
        else:
            raise ValueError("Each dataset must be (T,C,Y,X) or (T,Y,X).")

    series = [_latitudinal_mean(a) for a in datasets]
    nlon = series[0].shape[0]
    if any(s.shape[0] != nlon for s in series):
        raise ValueError("All datasets must share the same longitude dimension length.")

    # Longitudes
    if lon is None:
        if lon_resolution_deg is not None:
            lon = lon_start_deg + lon_resolution_deg * np.arange(nlon)
        else:
            # Fallback: index along x
            lon = np.arange(nlon)
            if x_ax_name == "Longitude [deg]":  # avoid misleading units
                x_ax_name = "Longitude index"
    else:
        if lon.shape[0] != nlon:
            raise ValueError("Provided 'lon' length does not match data width.")

    # Labels
    if labels is None:
        labels = [f"Series {i+1}" for i in range(len(series))]
    elif len(labels) != len(series):
        raise ValueError("Length of 'labels' must match number of datasets.")

    # Plot
    ax = axis if axis is not None else plt.subplots()[1]
    for y, lab in zip(series, labels):
        ax.plot(lon, y, label=lab, linewidth=linewidth)

    ax.set_xlabel(x_ax_name, fontsize=fontsize)
    ax.set_ylabel(y_ax_name, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    if legend:
        ax.legend()

    if do_savefig:
        plt.savefig(do_savefig)

    return {"ax": ax, "lon": lon, "series": series}


# --------------------------
# LONGITUDINAL MEAN (→ function of LATITUDE)
# --------------------------
def longitudinal_mean_multi(
    *datasets,
    labels=None,
    lat=None,  # optional 1D latitude array (nlat,)
    lat_resolution_deg=None,  # or supply resolution in degrees/pixel
    lat_start_deg=-90.0,  # used if lat is not provided
    axis=None,
    linewidth=None,
    legend=True,
    title="Longitudinal mean",
    x_ax_name="Latitude [deg]",
    y_ax_name="Mean value",
    fontsize=18,
    do_savefig=False,
):
    """
    Computes the longitudinal mean (i.e., mean over longitude), returning a
    series vs. LATITUDE for any number of datasets.
    """
    if len(datasets) == 0:
        raise ValueError("Provide at least one dataset.")

    # Reduce to (nlat,) by averaging over time, channel, and longitude.
    def _longitudinal_mean(a):
        if a.ndim == 4:  # (T,C,Y,X)
            return a.mean(axis=(0, 1, 3))
        elif a.ndim == 3:  # (T,Y,X)
            return a.mean(axis=(0, 2))
        else:
            raise ValueError("Each dataset must be (T,C,Y,X) or (T,Y,X).")

    series = [_longitudinal_mean(a) for a in datasets]
    nlat = series[0].shape[0]
    if any(s.shape[0] != nlat for s in series):
        raise ValueError("All datasets must share the same latitude dimension length.")

    # Latitudes
    if lat is None:
        if lat_resolution_deg is not None:
            lat = lat_start_deg + lat_resolution_deg * np.arange(nlat)
        else:
            # Fallback: index along y
            lat = np.arange(nlat)
            if x_ax_name == "Latitude [deg]":
                x_ax_name = "Latitude index"
    else:
        if lat.shape[0] != nlat:
            raise ValueError("Provided 'lat' length does not match data height.")

    # Labels
    if labels is None:
        labels = [f"Series {i+1}" for i in range(len(series))]
    elif len(labels) != len(series):
        raise ValueError("Length of 'labels' must match number of datasets.")

    # Plot
    ax = axis if axis is not None else plt.subplots()[1]
    for y, lab in zip(series, labels):
        ax.plot(lat, y, label=lab, linewidth=linewidth)

    ax.set_xlabel(x_ax_name, fontsize=fontsize)
    ax.set_ylabel(y_ax_name, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    if legend:
        ax.legend()

    if do_savefig:
        plt.savefig(do_savefig)

    return {"ax": ax, "lat": lat, "series": series}


# --------------------------
# HISTOGRAMS (multi-series)
# --------------------------
def histograms_multi(
    *datasets,
    labels=None,
    bins=100,
    density=True,
    logy=True,
    alpha=1.0,
    axis=None,
    title="Value distribution",
    x_ax_name="Value",  # fixed: was "Longitude" before (wrong)
    y_ax_name="Density",  # clearer default
    fontsize=18,
    xlim=None,  # tuple (min,max) or None
    linewidth=2,
    legend=True,
    do_savefig=False,
):
    """
    Plot histograms for any number of datasets (flattened), using default colors.
    """
    if len(datasets) == 0:
        raise ValueError("Provide at least one dataset.")

    flat = [np.asarray(a).ravel() for a in datasets]

    # Labels
    if labels is None:
        labels = [f"Series {i+1}" for i in range(len(flat))]
    elif len(labels) != len(flat):
        raise ValueError("Length of 'labels' must match number of datasets.")

    # Plot
    ax = axis if axis is not None else plt.subplots()[1]
    for arr, lab in zip(flat, labels):
        _ = ax.hist(
            arr,
            bins=bins,
            histtype="step",
            log=logy,
            density=density,
            alpha=alpha,
            linewidth=linewidth,
            label=lab,
        )

    ax.set_xlabel(x_ax_name, fontsize=fontsize)
    ax.set_ylabel(y_ax_name, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5, alpha=0.6)
    if legend:
        ax.legend()

    if do_savefig:
        plt.savefig(do_savefig)

    return {"ax": ax}


def plot_four_panels(
    *datasets,
    labels=None,
    figsize=(18, 12),
    suptitle=None,
    tight_layout=True,
    # Per-panel kwargs
    psd_kwargs=None,
    lat_kwargs=None,
    lon_kwargs=None,
    hist_kwargs=None,
):
    """
    Create a 2x2 figure with colorblind-friendly styling (Okabe-Ito palette).
    """
    # --- 1. Define Colorblind-Friendly Palette (Okabe-Ito) ---
    # Order: Black, Orange, Sky Blue, Bluish Green, Yellow, Blue, Vermilion, Reddish Purple
    cb_colors = [
        "#000000",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
    ]

    # Prepare kwargs containers
    psd_kwargs = {} if psd_kwargs is None else dict(psd_kwargs)
    lat_kwargs = {} if lat_kwargs is None else dict(lat_kwargs)
    lon_kwargs = {} if lon_kwargs is None else dict(lon_kwargs)
    hist_kwargs = {} if hist_kwargs is None else dict(hist_kwargs)

    # If a panel didn't get labels, fall back to shared labels
    for d in (psd_kwargs, lat_kwargs, lon_kwargs, hist_kwargs):
        d.setdefault("labels", labels)

    # Build figure & axes
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # --- 2. Apply the Palette to All Axes ---
    # This ensures any plot commands inside helper functions use these colors
    for ax in axes.flat:
        ax.set_prop_cycle(color=cb_colors)

    # (0,0) PSD
    psd_kwargs.setdefault("title", psd_kwargs.get("title", "Mean radial PSD"))
    psd_res = spatial_spectral_density_multi(
        *datasets,
        axis=axes[0, 0],
        **psd_kwargs,
    )

    # (0,1) Latitudinal mean
    lat_kwargs.setdefault("title", lat_kwargs.get("title", "Latitudinal mean"))
    lat_res = latitudinal_mean_multi(
        *datasets,
        axis=axes[0, 1],
        **lat_kwargs,
    )

    # (1,0) Longitudinal mean
    # Note: Your original code mapped this to axes[1, 1], I kept it as is.
    lon_kwargs.setdefault("title", lon_kwargs.get("title", "Longitudinal mean"))
    lon_res = longitudinal_mean_multi(
        *datasets,
        axis=axes[1, 1],
        **lon_kwargs,
    )

    # (1,1) Histograms
    # Note: Your original code mapped this to axes[1, 0], I kept it as is.
    hist_kwargs.setdefault("title", hist_kwargs.get("title", "Value distribution"))
    hist_res = histograms_multi(
        *datasets,
        axis=axes[1, 0],
        **hist_kwargs,
    )

    if suptitle:
        fig.suptitle(suptitle)

    if tight_layout:
        try:
            fig.tight_layout()
        except Exception:
            pass

    return {
        "fig": fig,
        "axes": axes,
        "psd": psd_res,
        "lat": lat_res,
        "lon": lon_res,
        "hist": hist_res,
    }


def infer_era5_colormap(field_name: str) -> str:
    """
    Choose a sensible colormap based on an ERA5-style field name.
    Covers all fields in your list via substring checks.
    """
    f = field_name.lower()

    # Temperatures
    if "temperature" in f:
        return "coolwarm"

    # Geopotential
    if "geopotential" in f:
        return "coolwarm"

    # Specific humidity
    if "specific_humidity" in f or "humidity" in f:
        return "PuBuGn"

    # Wind components
    if "u_component_of_wind" in f or "v_component_of_wind" in f:
        return "RdBu_r"

    # Mean sea level pressure
    if "mean_sea_level_pressure" in f:
        return "cividis"

    # Precipitation
    if "precipitation" in f or "tp" in f:
        return "Blues"

    # Fallback
    return "turbo"


def plot_era5_fields(
    data_array,
    panel_labels,
    field_name,
    lats=None,
    lons=None,
    cmap="auto",
    units="",
):
    """
    Plot multiple versions of the SAME ERA5 field (e.g. target vs prediction)
    stacked vertically, each with its own colorbar.

    Parameters
    ----------
    data_array : array-like or torch.Tensor
        Shape (N, H, W) or (H, W), where N = number of panels.
    panel_labels : list of str
        Labels for each panel (e.g. ["Target", "Prediction"]).
    field_name : str
        Name of the field (e.g. "850m_temperature").
        Used for:
          - Figure title
          - Colormap selection when cmap="auto"
    lats : 1D array, optional
        Latitudes (size H). If None, assumes linspace(-90, 90, H).
    lons : 1D array, optional
        Longitudes (size W). If None, assumes regular 0..360 grid.
    cmap : str, optional
        - "auto": choose colormap based on field_name
        - any Matplotlib colormap name (applied to all panels)
    units : str, optional
        Label for each colorbar (e.g. "K").
    """
    # Torch → NumPy
    try:
        import torch

        if isinstance(data_array, torch.Tensor):
            data_array = data_array.detach().cpu().numpy()
    except ImportError:
        pass

    data_array = np.asarray(data_array)

    # Ensure shape (N, H, W)
    if data_array.ndim == 2:
        data_array = data_array[None, ...]  # (H,W) → (1,H,W)

    n_panels, nlat, nlon = data_array.shape

    if len(panel_labels) != n_panels:
        raise ValueError(f"Need {n_panels} panel_labels, got {len(panel_labels)}")

    # Coordinates
    if lats is None:
        lats = np.linspace(-90, 90, nlat)
    else:
        lats = np.asarray(lats)
        assert lats.size == nlat, "lats length must match data latitude dimension"

    if lons is None:
        lons = np.linspace(0, 360, nlon, endpoint=False)
    else:
        lons = np.asarray(lons)
        assert lons.size == nlon, "lons length must match data longitude dimension"

    # Shared color scale across all panels (so you can compare)
    vmin = np.nanmin(data_array)
    vmax = np.nanmax(data_array)

    proj = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(8, 4 * n_panels),
        subplot_kw=dict(projection=proj),
        squeeze=False,
    )
    axes = axes[:, 0]

    # Decide colormap for this field
    if cmap == "auto":
        chosen_cmap = infer_era5_colormap(field_name)
    else:
        chosen_cmap = cmap

    for i in range(n_panels):
        ax = axes[i]
        field = data_array[i]

        # Cyclic point to avoid seam
        field_cyc, lons_cyc = add_cyclic_point(field, coord=lons)
        lon2d, lat2d = np.meshgrid(lons_cyc, lats)

        ax.set_global()

        img = ax.pcolormesh(
            lon2d,
            lat2d,
            field_cyc,
            cmap=chosen_cmap,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
            transform=proj,
        )

        # Subtle coastlines / borders
        ax.coastlines(resolution="110m", linewidth=0.5, color="k", alpha=0.8)
        # ax.add_feature(
        #     cfeature.BORDERS.with_scale("110m"),
        #     linewidth=0.3,
        #     edgecolor="gray",
        #     alpha=0.5,
        # )

        # Panel label (e.g. "Target", "Prediction")
        ax.set_title(panel_labels[i])

        # Separate colorbar for this panel
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(
            "right",
            size="3%",
            pad=0.2,
            axes_class=plt.Axes,
        )
        cbar = plt.colorbar(img, cax=cax, orientation="vertical")
        if units:
            cbar.set_label(units)

    # One overall title with the field name
    fig.suptitle(field_name, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    plt.show()
