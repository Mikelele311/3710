"""Compute and plot the Mandelbrot set with PyTorch."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch Version:", torch.__version__)
print("Device:", device)

# Create coordinates covering the useful part of the complex plane.
y_values, x_values = np.mgrid[-1.3:1.3:0.005, -2.0:1.0:0.005]
x = torch.from_numpy(x_values.astype(np.float32)).to(device)
y = torch.from_numpy(y_values.astype(np.float32)).to(device)

# Each pixel represents the complex number c = x + yi.
c = torch.complex(x, y)
z = torch.zeros_like(c)  # z0 = 0 for every point.
iteration_counts = torch.zeros(c.shape, dtype=torch.float32, device=device)
active = torch.ones(c.shape, dtype=torch.bool, device=device)

# Repeatedly apply z(n+1) = z(n)^2 + c.
for _ in range(200):
    z[active] = z[active] ** 2 + c[active]
    active &= torch.abs(z) < 4.0
    iteration_counts += active


def colour_fractal(counts: np.ndarray) -> np.ndarray:
    """Convert iteration counts into an RGB image."""
    colours = 2.0 * np.pi * counts / 20.0
    image = np.stack(
        (
            10 + 20 * np.cos(colours),
            30 + 50 * np.sin(colours),
            155 - 80 * np.cos(colours),
        ),
        axis=-1,
    )
    image[counts == counts.max()] = 0
    return np.uint8(np.clip(image, 0, 255))


result = iteration_counts.cpu().numpy()
figure, axis = plt.subplots(figsize=(12, 8), constrained_layout=True)
axis.imshow(colour_fractal(result), extent=(-2, 1, -1.3, 1.3), origin="lower")
axis.set_title("Mandelbrot Set")
axis.set_xlabel("Real axis")
axis.set_ylabel("Imaginary axis")

output_path = Path(__file__).resolve().parent / "mandelbrot.png"
figure.savefig(output_path, dpi=200)
print("Saved:", output_path.name)
plt.show()
