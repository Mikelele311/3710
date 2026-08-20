# Detailed research: high-resolution Mandelbrot and Julia sets

## 1. Aim of the experiment

This experiment extends the full Mandelbrot image from `2-1.py` in two ways:

1. It decreases the coordinate spacing and zooms into a detailed part of the Mandelbrot boundary.
2. It changes the interpretation of the coordinate grid to produce a Julia set.

Both calculations use the same quadratic recurrence:

\[
z_{n+1}=z_n^2+c.
\]

The important difference is which value changes from pixel to pixel. In the Mandelbrot calculation, `c` changes and `z0` is always zero. In the Julia calculation, `c` is fixed and `z0` changes.

The program saves the results as:

- `mandelbrot_zoom.png`
- `julia_set.png`

## 2. Complex-number calculation

A complex number contains a real and imaginary component:

\[
z=x+yi.
\]

Squaring it gives:

\[
z^2=(x+yi)^2=x^2-y^2+2xyi.
\]

Therefore, one iteration of the quadratic function is:

\[
z_{n+1}=(x_n^2-y_n^2+c_x)+(2x_ny_n+c_y)i.
\]

It would be possible to calculate those two components separately. PyTorch's native complex tensors make that unnecessary: `z ** 2 + c` performs the real and imaginary arithmetic directly. PyTorch supports `complex64` and `complex128` tensors, and `torch.abs` computes their magnitudes.

The script starts with NumPy `float32` coordinate arrays. Combining two `float32` tensors with `torch.complex` produces a `complex64` tensor. This is a useful balance for this lab because every complex value occupies 8 bytes rather than the 16 bytes required by `complex128`.

## 3. Turning pixels into complex coordinates

The function `make_complex_grid` receives minimum and maximum real and imaginary coordinates plus a spacing:

```python
y_values, x_values = np.mgrid[y_min:y_max:spacing, x_min:x_max:spacing]
```

`np.mgrid` creates dense coordinate matrices. Every output position corresponds to one image pixel. At array position `[row, column]`:

```text
x_values[row, column] = real coordinate
y_values[row, column] = imaginary coordinate
```

The two matrices are converted to tensors and combined:

```python
x = torch.from_numpy(x_values.astype(np.float32)).to(device)
y = torch.from_numpy(y_values.astype(np.float32)).to(device)
plane = torch.complex(x, y)
```

Conceptually, each pixel now stores:

\[
\text{plane[row,column]}=x_{column}+y_{row}i.
\]

For a real-valued `mgrid` step, the stopping value is normally excluded. Floating-point representation can create a one-sample difference, which is why the measured Mandelbrot grid is `2000 x 2001` instead of exactly `2000 x 2000`. This does not affect the visible coordinate range or the fractal calculation.

## 4. Mandelbrot set versus Julia set

The two images use the same equation but represent different mathematical spaces:

| Property | Mandelbrot set | Julia set |
|---|---|---|
| Space being displayed | Parameter space | Dynamical plane |
| Value represented by a pixel | The parameter `c` | The starting value `z0` |
| Initial value | `z0 = 0` for every pixel | Different `z0` at every pixel |
| Parameter | Different `c` at every pixel | One fixed `c` for the whole image |
| Question | For which functions does the critical orbit remain bounded? | Which starting points remain bounded for this one function? |

The difference appears directly in the program:

```python
# Mandelbrot
calculate_escape_counts(torch.zeros_like(mandelbrot_plane), mandelbrot_plane)

# Julia
calculate_escape_counts(julia_plane, julia_constant)
```

For the Mandelbrot image, `mandelbrot_plane` is passed as `c`. For the Julia image, `julia_plane` is passed as the initial `z`, and every element of `julia_constant` has the same value.

## 5. Escape-time algorithm

The program cannot run infinitely, so membership is approximated with an escape-time algorithm. It performs at most 300 iterations for every pixel.

Three tensors describe the current state:

```python
z = z.clone()
counts = torch.zeros(z.shape, dtype=torch.float32, device=device)
active = torch.ones(z.shape, dtype=torch.bool, device=device)
```

- `z` stores the current complex orbit value for every pixel.
- `counts` stores how many iterations the point has remained within the escape radius.
- `active` is `True` only for points that have not escaped.

Each loop executes:

```python
z[active] = z[active] ** 2 + c[active]
active &= torch.abs(z) < 4.0
counts += active
```

The first line updates only active points. This prevents escaped values from being squared repeatedly until they overflow.

The second line calculates the magnitude

\[
|z|=\sqrt{x^2+y^2}
\]

and permanently removes points whose magnitude reaches 4. Because `&=` is a cumulative logical AND, a point can never become active again after escaping.

The final line increments the counter for the surviving points. Therefore:

- A small count means the point escaped quickly.
- A large count means it remained near the fractal boundary for longer.
- A count of 300 means it did not escape during the experiment.

The last category is described as "bounded" in the image, but this is still a finite approximation. Failure to escape after 300 iterations is not a general mathematical proof that an orbit remains bounded forever.

## 6. Why an escape radius works

The reverse triangle inequality gives:

\[
|z^2+c|\geq |z|^2-|c|.
\]

Once `|z|` is sufficiently large, the squared term grows faster than both `|z|` and the fixed addition `c`. The orbit then continues growing and cannot return to a bounded region.

For the Mandelbrot set, any possible member has `|c| <= 2`. If `|z| > 2`, then:

\[
|z_{n+1}|\geq |z_n|^2-2>|z_n|.
\]

Thus radius 2 is the standard sufficient escape boundary. The lab code uses radius 4. This is also safe; it simply records an escaping point a few iterations later, changing its colour slightly without changing whether it eventually escapes.

For the selected Julia constant, `|c|` is approximately `0.815`, so radius 4 is comfortably outside the required boundary there as well.

## 7. High-resolution Mandelbrot design

The original full image in `2-1.py` used:

```text
Real width:         3.0
Imaginary height:   2.6
Spacing:            0.005
Grid:               600 x 520
Pixels:             312,000
Iterations:         200
Maximum updates:    62.4 million
```

The new zoom uses:

```text
Real range:         -0.80 to -0.70
Imaginary range:     0.05 to 0.15
Spacing:             0.00005
Measured grid:       2000 x 2001
Pixels:              4,002,000
Iterations:          500
Maximum updates:     2.001 billion
CPU computation:     33.47 seconds
PNG size:            2700 x 2400 pixels at 300 DPI
```

The actual new spacing is `0.00005`, which is 100 times smaller than `0.005`, so the coordinate sampling is 100 times finer along each axis. The grid contains about 12.8 times as many pixels as the original because the field of view is much smaller. Compared with the previous `0.0001` zoom, halving the spacing doubles both grid dimensions and gives approximately four times as many samples.

The selected region crosses a highly detailed boundary area commonly called Seahorse Valley. Boundary regions are useful for demonstrating resolution because they contain fine spirals and repeated smaller structures. A crop entirely inside the Mandelbrot set would appear mostly black, while a crop far outside it would contain only smooth escape bands.

### Why the full plane is not rendered at this spacing

Keeping the original full range while changing the spacing to `0.00005` would produce approximately:

\[
60000\times52000=3,120,000,000\text{ pixels}.
\]

A single `complex64` tensor of that size would require about 24.96 GB. The calculation also needs `z`, the coordinate components, the count tensor, the Boolean mask, and temporary results. The working memory would exceed roughly 80 GB, and 500 iterations could require up to 1.56 trillion point updates.

Zooming is therefore not only visually useful; it is necessary to keep the experiment practical.

In general, for a fixed rectangular view with spacing `h`, the number of pixels is approximately:

\[
N\approx\frac{\text{width}\times\text{height}}{h^2}.
\]

Halving `h` doubles both dimensions and produces roughly four times as many pixels. Computation scales approximately as:

\[
O(NI),
\]

where `I` is the iteration limit. Memory scales approximately as `O(N)`.

## 8. Julia set design and parameter choice

The Julia image fixes:

\[
c=-0.8+0.156i
\]

and tests starting points over:

```text
Real range:         -1.7 to 1.7
Imaginary range:    -1.3 to 1.3
Spacing:             0.003
Measured grid:       867 x 1134
Pixels:              983,178
Iterations:          300
CPU computation:     2.43 seconds
```

Every pixel follows the same function:

\[
f(z)=z^2-0.8+0.156i.
\]

The filled Julia set `K(c)` is the collection of starting points whose orbits remain bounded. The mathematical Julia set `J(c)` is the boundary of that filled set:

\[
J(c)=\partial K(c).
\]

The saved image is therefore an escape-time rendering around the filled Julia set, not a plot containing only the infinitely thin mathematical boundary.

For a quadratic polynomial, the Julia set is connected exactly when the critical orbit starting from zero is bounded. A separate 10,000-iteration check of the chosen constant produced:

```text
Escape iteration:   253
Magnitude:          approximately 9.27
```

Once that magnitude is reached, the escape inequality guarantees continued growth. The critical orbit therefore escapes, placing this `c` outside the Mandelbrot set and giving a disconnected Julia set. The separated, curling structures in the output agree with that prediction.

## 9. Colouring and image interpretation

The program masks points with the maximum count:

```python
bounded_points = np.ma.masked_equal(counts, max_iterations)
colour_map.set_bad("black")
```

Matplotlib uses black for masked points. Other points use the `turbo` colour map according to their escape counts.

The colours do not represent different mathematical sets. They encode computational behaviour:

- Dark purple points escape quickly.
- Blue, green, and yellow points survive progressively longer.
- Orange and red points are very close to the boundary and escape late.
- Black points did not escape within 300 iterations.

`interpolation="nearest"` ensures Matplotlib does not blur neighbouring pixels together. This is important in a high-resolution fractal because smoothing could invent intermediate colours that were not produced by the calculation.

The `extent` argument maps array coordinates back to meaningful complex-plane values. `origin="lower"` places positive imaginary values above the real axis, matching the normal mathematical orientation of the complex plane.

## 10. CPU, GPU, and tensor transfer

The device is selected with:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

All large tensors are moved to that device before iteration. If CUDA is available, the same tensor expressions execute on the GPU without changing the fractal algorithm.

Matplotlib cannot directly display a CUDA tensor. The completed count tensor is therefore transferred back with:

```python
counts.cpu().numpy()
```

This transfer happens only after all iterations. Moving data between CPU and GPU inside the loop would add substantial overhead and reduce the benefit of GPU computation.

Automatic differentiation is not required. The task evaluates an iterative function but does not calculate gradients or train a model, so no tensor uses `requires_grad=True`.

## 11. Accuracy and limitations

The output depends on three numerical choices:

### Grid spacing

Smaller spacing reveals finer spatial detail but increases memory and computation approximately with the inverse square of the spacing.

### Iteration limit

More iterations distinguish genuinely bounded-looking points from points that escape very slowly. Increasing this value improves boundary classification but raises execution time approximately linearly.

### Floating-point precision

`complex64` is fast and memory-efficient, but it has limited precision. Very deep zooms eventually require `complex128` or specialised arbitrary-precision arithmetic because neighbouring coordinates become too close to distinguish accurately with `float32` components.

The present zoom is shallow enough for `complex64`; its coordinate spacing of `0.00005` is still above the approximately seven-decimal-digit precision limit of `float32` around values near one.

Escape-time colouring also produces discrete colour bands because iteration counts are integers. Smooth colouring could estimate fractional escape times, but that would add mathematics and code not required by this lab.

## 12. Results and conclusions

The experiment demonstrates both required modifications:

1. Decreasing the Mandelbrot spacing from `0.005` to `0.00005` reveals repeated spirals and much finer boundary details. Restricting the view keeps the four-million-point calculation to 33.47 seconds on the available CPU.
2. Exchanging the roles of the grid and the constant changes the parameter-space Mandelbrot calculation into a dynamical-plane Julia calculation. The selected fixed parameter produces a disconnected Julia structure and takes 2.43 seconds to compute.

The main conceptual conclusion is that the Mandelbrot and Julia images are not generated by different equations. They arise from different uses of the same equation: the Mandelbrot set classifies parameters, while a Julia set classifies initial values for one parameter.

## 13. Demonstrator questions and concise answers

**Why is the Mandelbrot initial value zero?**

For the quadratic family `z^2 + c`, zero is the unique finite critical point. Its orbit determines whether `c` belongs to the Mandelbrot set and whether the associated Julia set is connected.

**What changed to create the Julia set?**

The coordinate grid became `z0` instead of `c`, and `c` became one fixed complex number across the grid.

**Why do black points not necessarily prove membership?**

They only show that the point survived the finite limit of 300 iterations. A point might escape after iteration 300.

**Why stop updating escaped points?**

Their final classification is already known. Continuing would waste computation and could overflow the complex tensor.

**Why use a zoom instead of the full plane?**

At spacing `0.00005`, the old full view would need about 3.12 billion pixels and more than 80 GB of working tensor memory.

**What controls the Julia set's appearance?**

The fixed complex parameter `c`. Even small changes to it can alter the orbit structure, boundary, symmetry, and connectedness.

## 14. Sources

1. B. B. Mandelbrot, [Fractal Aspects of the Iteration of z to lambda z(1-z) for Complex lambda and z](https://doi.org/10.1111/j.1749-6632.1980.tb29690.x), *Annals of the New York Academy of Sciences*, 1980.
2. B. B. Mandelbrot, [On the quadratic mapping z to z-squared minus mu for complex mu and z](https://doi.org/10.1016/0167-2789(83)90128-8), *Physica D*, 1983.
3. Y. Jiang, [Infinitely Renormalizable Quadratic Polynomials](https://doi.org/10.1090/S0002-9947-00-02514-9), *Transactions of the American Mathematical Society*, 2000. This paper defines the filled Julia set as bounded orbits and the Julia set as its boundary.
4. R. Brueck, M. Bueger and S. Reitz, [Random iterations of polynomials of the form z-squared plus c-n: connectedness of Julia sets](https://doi.org/10.1017/S0143385799141658), *Ergodic Theory and Dynamical Systems*, 1999.
5. NumPy, [`numpy.mgrid` documentation](https://numpy.org/doc/stable/reference/generated/numpy.mgrid.html).
6. NumPy, [`numpy.meshgrid` documentation](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html).
7. PyTorch, [Complex Numbers documentation](https://docs.pytorch.org/docs/stable/complex_numbers.html).
8. PyTorch, [`torch.masked_select` and Boolean mask documentation](https://docs.pytorch.org/docs/stable/generated/torch.masked_select.html).
9. PyTorch, [CUDA semantics documentation](https://docs.pytorch.org/docs/stable/notes/cuda.html).
