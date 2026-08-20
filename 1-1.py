"""Create and display a two-dimensional Gaussian with NumPy and PyTorch."""

import numpy as np
import torch
import matplotlib.pyplot as plt


# Device configuration: use an NVIDIA GPU when PyTorch can access one.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    #create numpy array from -4 to 4 with 0.01 internal in both x, y
    x_coordinates, y_coordinates = np.mgrid[-4.0:4.0:0.01, -4.0:4.0:0.01]

    #turn numpy into torch and input data in device
    x = torch.from_numpy(x_coordinates).to(device)
    y = torch.from_numpy(y_coordinates).to(device)

    # follow formula of e^-((-x^2+x^2)/2)
    z = torch.exp(-(x**2 + y**2) / 2.0)

    # Plot z based on previous numpy data
    plt.imshow(z.cpu().numpy(), extent=(-4, 4, -4, 4), origin="lower")
    plt.colorbar(label="Gaussian value")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
