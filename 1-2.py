"""Plot a 2D sine function and a sine multiplied by a Gaussian."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_directory = Path(__file__).resolve().parent

# Create x and y coordinates directly as PyTorch tensors.
coordinates = torch.arange(-4.0, 4.0, 0.01, device=device)
y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")

# Keep the Gaussian because it is needed later in the lab.
gaussian = torch.exp(-(x**2 + y**2) / 2.0)

# The angle depends on x and y, producing diagonal stripes.
sine = torch.sin(2.0 * torch.pi * (x + y))

# Multiplication makes the stripes strongest at the centre and fades them out.
gaussian_sine = gaussian * sine

# Matplotlib requires CPU NumPy arrays.
sine_image = sine.cpu().numpy()
gaussian_sine_image = gaussian_sine.cpu().numpy()
extent = (-4, 4, -4, 4)

figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
plot = axis.imshow(sine_image, extent=extent, origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
axis.set_title("2D Sine Function")
axis.set_xlabel("x")
axis.set_ylabel("y")
figure.colorbar(plot, ax=axis, label="Value")
figure.savefig(output_directory / "sine_function.png", dpi=200)

figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
plot = axis.imshow(
    gaussian_sine_image,
    extent=extent,
    origin="lower",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
)
axis.set_title("Gaussian Multiplied by 2D Sine")
axis.set_xlabel("x")
axis.set_ylabel("y")
figure.colorbar(plot, ax=axis, label="Value")
figure.savefig(output_directory / "gaussian_sine_product.png", dpi=200)

plt.show()
