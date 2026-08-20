# AI prompt log: NumPy and PyTorch 2D functions

## Initial NumPy request

> Generate a Python script to plot a 2D Gaussian function using NumPy and Matplotlib.

## Extra prompt for a matching plot

> Match the earlier PyTorch result exactly: create x and y grids with `np.mgrid[-4.0:4.0:0.01, -4.0:4.0:0.01]`, calculate `exp(-(x**2 + y**2) / 2.0)`, use `imshow` with `origin="lower"` and an extent of `(-4, 4, -4, 4)`.

The extra detail was required because a general Gaussian request does not specify its bounds, image resolution, standard deviation, or plot orientation. Any of those changes would make a visually valid Gaussian but not a close replica of the first plot.

## PyTorch conversion request

> Convert the NumPy Gaussian script to PyTorch. Use `torch.from_numpy(...astype(np.float32))`, select `cuda` when available and otherwise `cpu`, and convert the final Tensor with `.cpu().numpy()` only for Matplotlib.

## 2D sine request

> Add a PyTorch 2D sine plot using the same coordinate Tensors. Make the angle depend on both axes with `2 * pi * (x + y)` so the result forms diagonal stripes. Plot it with a color scale fixed from -1 to 1.

## Observations

The conversion is nearly direct: `np.exp` becomes `torch.exp`, and the expression for the function is otherwise unchanged. The main PyTorch-specific steps are choosing a device, moving input Tensors to it, and returning the final Tensor to the CPU before calling Matplotlib. The NumPy and PyTorch Gaussian outputs are expected to differ only by a small floating-point rounding amount because the Tensor computation uses `float32`.
