"""Fourier reconstruction and epicycle animation for a 2D silhouette."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.collections import LineCollection


def coefficients(x, y):
    """Return Fourier coefficients and their signed frequencies, largest first."""
    signal = np.asarray(x, dtype=float) + 1j * np.asarray(y, dtype=float)
    values = np.fft.fft(signal) / len(signal)
    frequencies = np.fft.fftfreq(len(signal), d=1 / len(signal))
    order = np.argsort(np.abs(values))[::-1]
    return values[order], frequencies[order]


def reconstruct(values, frequencies, percentage, sample_count):
    """Reconstruct a closed path from the requested share of strongest terms."""
    term_count = max(1, int(np.ceil(len(values) * percentage / 100)))
    times = np.arange(sample_count) / sample_count
    terms = values[:term_count, None] * np.exp(
        2j * np.pi * frequencies[:term_count, None] * times[None, :]
    )
    return np.sum(terms, axis=0)


def save_reconstruction_mosaic(x, y, output_path):
    """Save the required 20-panel reconstruction from 5% to 100% of terms."""
    values, frequencies = coefficients(x, y)
    fig, axes = plt.subplots(4, 5, figsize=(14, 11))
    sample_count = len(x)

    for ax, percentage in zip(axes.flat, range(5, 101, 5)):
        path = reconstruct(values, frequencies, percentage, sample_count)
        ax.plot(path.real, path.imag, color="navy", linewidth=0.8)
        ax.set_title(f"{percentage}% of harmonics")
        ax.set_aspect("equal")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def save_epicycle_animation(x, y, output_path, term_limit=40, frame_count=120, fps=24):
    """Save an MP4 of rotating Fourier vectors drawing the silhouette."""
    import imageio_ffmpeg

    values, frequencies = coefficients(x, y)
    term_count = min(term_limit, len(values))
    values = values[:term_count]
    frequencies = frequencies[:term_count]
    times = np.arange(frame_count) / frame_count

    # Each coefficient is a rotating vector; adding the vectors tip-to-tail
    # gives the point currently being drawn by the epicycles.
    vectors = values[:, None] * np.exp(
        2j * np.pi * frequencies[:, None] * times[None, :]
    )
    tips = np.cumsum(vectors, axis=0)[-1]
    starts = np.vstack((np.zeros(frame_count, dtype=complex), np.cumsum(vectors, axis=0)[:-1]))

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.set_aspect("equal")
    margin = 0.08 * max(np.ptp(x), np.ptp(y), 1)
    ax.set_xlim(np.min(x) - margin, np.max(x) + margin)
    ax.set_ylim(np.max(y) + margin, np.min(y) - margin)
    ax.axis("off")
    ax.set_title(f"Fourier epicycles ({term_count} harmonics)")

    unit_circle = np.exp(1j * np.linspace(0, 2 * np.pi, 48))
    circle_paths = np.zeros((term_count, len(unit_circle), 2))
    circle_collection = LineCollection(circle_paths, colors="steelblue", linewidths=0.65, alpha=0.6)
    vector_collection = LineCollection(np.zeros((term_count, 2, 2)), colors="darkorange", linewidths=1.0)
    ax.add_collection(circle_collection)
    ax.add_collection(vector_collection)
    trace, = ax.plot([], [], color="navy", linewidth=1.4)
    tip, = ax.plot([], [], "o", color="crimson", markersize=3)

    def update(frame):
        centers = starts[:, frame]
        circle_points = centers[:, None] + np.abs(values)[:, None] * unit_circle[None, :]
        circle_collection.set_segments(
            np.stack((circle_points.real, circle_points.imag), axis=-1)
        )

        endpoints = centers + vectors[:, frame]
        segments = np.stack(
            (
                np.column_stack((centers.real, centers.imag)),
                np.column_stack((endpoints.real, endpoints.imag)),
            ),
            axis=1,
        )
        vector_collection.set_segments(segments)
        trace.set_data(tips[:frame + 1].real, tips[:frame + 1].imag)
        tip.set_data([tips[frame].real], [tips[frame].imag])
        return circle_collection, vector_collection, trace, tip

    animation = FuncAnimation(fig, update, frames=frame_count, interval=1000 / fps, blit=True)
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    writer = FFMpegWriter(
        fps=fps,
        codec="mpeg4",
        bitrate=1400,
        metadata={"title": "Fourier epicycles"},
        extra_args=["-pix_fmt", "yuv420p"],
    )
    animation.save(output_path, writer=writer)
    plt.close(fig)
