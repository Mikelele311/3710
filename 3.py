"""Create a Chaos Game Fern with vectorized PyTorch CPU calculations."""

import os
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import torch


# PyTorch divides each vectorized tensor operation between these CPU threads.
cpu_threads = os.cpu_count() or 1
torch.set_num_threads(cpu_threads)
torch.manual_seed(3710)

spacing = 0.01
iterations = 1000

# Add half a step to include 4.00 despite floating-point rounding.
values = torch.arange(-4.0, 4.0 + spacing / 2, spacing, dtype=torch.float32)
y_grid, x_grid = torch.meshgrid(values, values, indexing="ij")
x = x_grid.flatten()
y = y_grid.flatten()

print(f"PyTorch version: {torch.__version__}")
print(f"CPU threads: {torch.get_num_threads()}")
print(f"Grid: {x_grid.shape[1]} x {x_grid.shape[0]} = {x.numel():,} points")

start_time = perf_counter()

# The iterations are sequential, but every grid point is calculated at once.
# The probability intervals are 0-1%, 1-86%, 86-93%, and 93-100%.
for _ in range(iterations):
    probability = torch.rand_like(x)

    first = probability < 0.01
    second = (probability >= 0.01) & (probability < 0.86)
    third = (probability >= 0.86) & (probability < 0.93)

    x_new = torch.where(
        first,
        torch.zeros_like(x),
        torch.where(
            second,
            0.85 * x + 0.04 * y,
            torch.where(third, 0.20 * x - 0.26 * y, -0.15 * x + 0.28 * y),
        ),
    )
    y_new = torch.where(
        first,
        0.16 * y,
        torch.where(
            second,
            -0.04 * x + 0.85 * y + 1.60,
            torch.where(
                third,
                0.23 * x + 0.22 * y + 1.60,
                0.26 * x + 0.24 * y + 0.44,
            ),
        ),
    )
    x, y = x_new, y_new

calculation_time = perf_counter() - start_time

# Convert only once, after all PyTorch calculations have finished.
x_plot = x.numpy()
y_plot = y.numpy()

figure, axis = plt.subplots(figsize=(6, 10), constrained_layout=True)
axis.scatter(x_plot, y_plot, s=0.08, c="#168a36", marker=".", linewidths=0)
axis.set_title("Chaos Game Fern (1,000 iterations per grid point)")
axis.set_xlabel("x")
axis.set_ylabel("y")
axis.set_aspect("equal")
axis.set_facecolor("black")

output_path = Path(__file__).resolve().parent / "chaos_game_fern.png"
figure.savefig(output_path, dpi=250, facecolor=figure.get_facecolor())
plt.close(figure)

print(f"Calculation time: {calculation_time:.2f} seconds")
print(f"Saved: {output_path}")
