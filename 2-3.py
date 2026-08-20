"""Create a high-resolution Mandelbrot zoom and a Julia set with PyTorch."""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_directory = Path(__file__).resolve().parent

print("PyTorch Version:", torch.__version__)
print("Device:", device)


def make_complex_grid(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    spacing: float,
) -> torch.Tensor:
    """Create a grid in the complex plane and move it to the device."""
    y_values, x_values = np.mgrid[y_min:y_max:spacing, x_min:x_max:spacing]
    x = torch.from_numpy(x_values.astype(np.float32)).to(device)
    y = torch.from_numpy(y_values.astype(np.float32)).to(device)
    return torch.complex(x, y)


def calculate_escape_counts(
    z: torch.Tensor,
    c: torch.Tensor,
    max_iterations: int,
) -> np.ndarray:
    """Count how many iterations each point remains inside the escape radius."""
    z = z.clone()
    counts = torch.zeros(z.shape, dtype=torch.float32, device=device)
    active = torch.ones(z.shape, dtype=torch.bool, device=device)

    for _ in range(max_iterations):
        z[active] = z[active] ** 2 + c[active]
        active &= torch.abs(z) < 4.0
        counts += active

    return counts.cpu().numpy()


def save_fractal(
    counts: np.ndarray,
    extent: tuple,
    title: str,
    filename: str,
    max_iterations: int,
    dpi: int,
) -> None:
    """Plot escape counts, colouring points that never escaped black."""
    colour_map = plt.colormaps["turbo"].copy()
    colour_map.set_bad("black")
    bounded_points = np.ma.masked_equal(counts, max_iterations)

    figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    image = axis.imshow(
        bounded_points,
        extent=extent,
        origin="lower",
        cmap=colour_map,
        interpolation="nearest",
    )
    axis.set_title(title)
    axis.set_xlabel("Real axis")
    axis.set_ylabel("Imaginary axis")
    figure.colorbar(image, ax=axis, label="Iterations before escape")
    figure.savefig(output_directory / filename, dpi=dpi)
    plt.close(figure)


# High-resolution Mandelbrot zoom: z0 is zero and c varies at every pixel.
mandelbrot_iterations = 500
mandelbrot_plane = make_complex_grid(-0.80, -0.70, 0.05, 0.15, 0.00005)
start_time = perf_counter()
mandelbrot_counts = calculate_escape_counts(
    torch.zeros_like(mandelbrot_plane),
    mandelbrot_plane,
    mandelbrot_iterations,
)
mandelbrot_time = perf_counter() - start_time
save_fractal(
    mandelbrot_counts,
    (-0.80, -0.70, 0.05, 0.15),
    "High-Resolution Mandelbrot Zoom",
    "mandelbrot_zoom.png",
    mandelbrot_iterations,
    dpi=300,
)

# Julia set: z0 varies at every pixel while c remains fixed.
julia_iterations = 300
julia_plane = make_complex_grid(-1.7, 1.7, -1.3, 1.3, 0.003)
julia_constant = torch.full_like(julia_plane, -0.8 + 0.156j)
start_time = perf_counter()
julia_counts = calculate_escape_counts(julia_plane, julia_constant, julia_iterations)
julia_time = perf_counter() - start_time
save_fractal(
    julia_counts,
    (-1.7, 1.7, -1.3, 1.3),
    "Julia Set: c = -0.8 + 0.156i",
    "julia_set.png",
    julia_iterations,
    dpi=200,
)

print(f"Mandelbrot grid: {mandelbrot_plane.shape}, computation: {mandelbrot_time:.2f} seconds")
print(f"Julia grid: {julia_plane.shape}, computation: {julia_time:.2f} seconds")
print("Saved: mandelbrot_zoom.png")
print("Saved: julia_set.png")
