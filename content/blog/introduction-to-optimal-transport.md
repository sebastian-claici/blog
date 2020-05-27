+++
title = "Introduction to Optimal Transport"
date = 2020-05-27T00:00:00-04:00
tags = ["discretetransport"]
categories = ["transport"]
draft = false
+++

There are excellent [tutorials](https://arxiv.org/abs/1801.07745) out there, an amazing Python [package](https://pythonot.github.io/), and a lot
of great books on the topic, and my intention here isn't to replace them. The
only difference is that I provide a bunch of coded examples alongside
the math. You can get something similar, but significantly more developed at
Gabriel Peyre's excellent [numerical tours](https://www.numerical-tours.com/) website.


## Introduction {#introduction}

Optimal transport defines a distance between probability distributions. We can
tell how far apart two points \\(x\\) and \\(y\\) are in Euclidean space by
measuring their Euclidean distance

\begin{equation}
d(x, y) = \sqrt{\sum\_{i=1}^d (x\_i - y\_i)^2}.
\end{equation}

For probability distributions \\(\mu\\) and \\(\nu\\), this no longer makes sense, because we can have
infinite support, and varying mass between the points.

We can try to fix this as follows: Let's say that the distribution \\(\mu\\) is
supported on some set \\(X\\), and \\(\nu\\) is supported on some set \\(Y\\). For each pair
of points \\((x, y)\in X\times Y\\), let's assume that we are given a cost \\(c(x, y)\\)
which measures how difficult it is to move the point \\(x\\) onto the point \\(y\\).
Now, among all maps which take points in \\(X\\) to points in \\(Y\\), the one that has
the lowest cost ought to give us an idea of the difference between \\(\mu\\) and
\\(\nu\\). A couple of issues remain: How can we know that such a map exists, and,
if it does, what conditions does it to be a map from \\(\mu\\) to \\(\nu\\).

Let's deal with the latter of these first. We want a constraint on \\(T\\) that
preserves mass. If we take a subset \\(B\subseteq Y\\), its mass under \\(\nu\\) is
\\(\nu(B)\\). The preimage of \\(B\\) under \\(T\\) is some set \\(A = T^{-1}(B)\\). For mass
preservation to hold, we need that the mass under \\(\mu\\) of \\(A\\) be equal to the
mass under \\(\nu\\) of \\(B\\):

We start with a simple version of the problem.


## The Transport Equation {#the-transport-equation}

If \\(\mu\\) and \\(\nu\\) are discrete distributions, they can be written as

\begin{equation}
\mu = \sum\_{i=1}^n \alpha\_i \delta\_{x\_i}\quad \nu = \sum\_{j=1}^m \beta\_j \delta\_{y\_j}.
\end{equation}

The cost function can be represented as a \\(n\times m\\) matrix \\(C\\) where \\(C\_{ij} =
c(x\_i, y\_j)\\). If we try to look for a map \\(T\\) here, we run into trouble quickly.
Here is an example on \\(\mathbb{R}\\) where such a map does not exist:

\begin{equation}
\mu = \delta\_0\quad \nu = \frac{1}{2}\delta\_{-1} + \frac{1}{2}\delta\_{1}.
\end{equation}

The problem here is that mass preservation needs \\(\frac{1}{2}\\) mass at \\(1\\) and
\\(-1\\), but \\(T\\) is a map that takes all of the mass at a point to another point.
We cannot _split mass_ with our current formulation.

The underlying problem is that \\(T\\) _must_ be a map. We can relax that
constraint. In the discrete setting, the same way we have a variable for the
cost between \\(x\_i\\) and \\(y\_j\\), we can have a variable for the mass moved from
\\(x\_i\\) to \\(y\_j\\). We call this variable \\(T\\), and it, like the cost matrix, is an
\\(n\times m\\) matrix.

The mass preservation constraints on \\(T\\) are that the sum total of masses moved
_to_ \\(y\_j\\) is equal to \\(\beta\_j\\) for each \\(j\\), and that the sum total of mass
moved _from_ \\(x\_i\\) is equal to \\(\alpha\_i\\) for each \\(i\\). We can write this out in matrix
notation as

\begin{equation}
\begin{split}
T \mathbf{1} &= \beta\\\\\\
T^\intercal \mathbf{1} &= \alpha\\\\\\
T &\geq 0.
\end{split}
\end{equation}

We can give a visual interpretation of what such a \\(T\\) looks like. To run this
code, we will need a Python3+ interpreter, `numpy`, `matplotlib`, and `cvxpy`. We can set
up everything with the following shell commands:

```bash
python3 -m venv env

source env/bin/activate
pip install numpy matplotlib
pip install cvxpy
```


## Initial Data {#initial-data}

We can start with samples from simple distributions in two dimensions. Let's
import everything we will need later on first.

```python
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
```

Our initial distributions are randomly distributed points with varying sizes
depending on the weight assigned to each:

```python
m, n = 20, 20
xs = np.random.rand(m, 2) * 0.5
ys = np.random.randn(n, 2) * 0.4

mu = np.random.rand(m)
nu = np.random.rand(n)
mu /= np.sum(mu)
nu /= np.sum(nu)
```

We can plot the distributions.

```python
plt.style.use('ggplot')
fig = plt.figure(figsize=(12,9))

plt.scatter(xs[:, 0], xs[:, 1], s=2000*mu, c='r', edgecolors='k')
plt.scatter(ys[:, 0], ys[:, 1], s=2000*nu, c='b', edgecolors='k')

plt.savefig('scatter.svg', bbox_inches='tight')
```

{{< figure src="/ox-hugo/scatter.svg" >}}


## Cost Matrix {#cost-matrix}

The first thing we need to compute a transportation distance is a cost matrix
between the support of the two measures. In this case, the supports are the
centres of the histograms and given by `x` and `y`. We compute a pairwise
distance matrix between them as:

```python
C = np.zeros((n, m))
for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        C[i, j] = np.linalg.norm(x - y) ** 2
```


## Linear Programming {#linear-programming}

The problem we want to solve is a linear program with a matrix variable.
Normally, you would use a specialised library such as [Python Optimal Transport](https://pythonot.github.io/)
which calls a C++ routine that is blazing fast. We are not going to do that,
since that teaches you nothing.

Instead, we will rely on a linear programming solver. Recall the problem we want
to solve:

\begin{equation}
\begin{aligned}
\underset{T\in\mathbb{R}^{n\times m}}{\mathrm{minimize}}\quad & \sum\_{i, j} C\_{i,j} T\_{i,j}\\\\\\
\text{subject to} \quad&
\begin{cases}
T \mathbf{1} &= \beta\\\\\\
T^\intercal \mathbf{1} &= \alpha\\\\\\
T &\geq 0.
\end{cases}
\end{aligned}
\end{equation}

It is fairly straightforward to write this out in `cvxpy`, a Python convex
optimisation implementation of the popular [CVX](http://cvxr.com/) framework. We import the package
and define our optimisation variable `T`:

```python
T = cp.Variable((n, m))
```

The objective is the sum of the elementwise multiplication of `T` and `C`:

```python
objective = cp.Minimize(cp.sum(cp.multiply(T, C)))
```

The constraints can be written in matrix form using `cp.matmul`:

```python
u = np.ones((n, 1))
v = np.ones((m, 1))
mu = np.reshape(mu, (n, 1))
nu = np.reshape(nu, (m, 1))
constraints = [T >= 0, cp.matmul(T, u) == nu, cp.matmul(T.T, v) == mu]
```

We construct the optimisation problem using `cp.Problem` and handing it the
objective and constraints:

```python
problem = cp.Problem(objective, constraints)
result = problem.solve()
```


## Visualising the Coupling {#visualising-the-coupling}

The optimal coupling is meant to be sparse. We can check this numerically:

```python
print('Number of non-zeros in T is {} (n + m - 1: {})'.format(np.sum(T.value > 1e-5), n + m - 1))
```

```text
Number of non-zeros in T is 39 (n + m - 1: 39)
```

We can also visualise the coupling as a 2D heatmap:

```python
plt.figure(figsize=(5,5))
plt.imshow(T.value)
plt.savefig("coupling-heat.svg", bbox_inches='tight')
```

{{< figure src="/ox-hugo/coupling-heat.svg" >}}

We can also plot the point to point matching based on the values in `T`:

```python
plt.figure(figsize=(12, 9))

plt.scatter(xs[:, 0], xs[:, 1], s=2000*mu, c='r', edgecolors='k')
plt.scatter(ys[:, 0], ys[:, 1], s=2000*nu, c='b', edgecolors='k')

I, J = np.nonzero(T.value > 1e-5)
for i, j in zip(I, J):
    plt.plot([xs[i, 0], ys[j, 0]], [xs[i, 1], ys[j, 1]], 'k-')

plt.savefig("coupling-match.svg", bbox_inches='tight')
```

{{< figure src="/ox-hugo/coupling-match.svg" >}}


## Displacement Interpolation {#displacement-interpolation}

One neat thing about optimal transport is that the interpolation between two
distributions works _geometrically_. That is, we see the circles moving from the
start distribution to the end distribution, and not just the size of the circles
increasing.

Given a transport _map_ \\(T\\), the interpolant between \\(\mu\\) and \\(\nu\\) is the map
\\[
f\_\alpha(x) = ((1 - \alpha) \mathrm{Id} + \alpha T)(x)
.
\\]
We can construct something similar using our transport _plan_ if we allow only
part of the mass from each point to move. Here's how that would look:

```python
def position(xs, ys, T, alpha):
    I, J = np.nonzero(T > 1e-5)
    zs = []
    ws = []
    for i, j in zip(I, J):
        zs.append((1 - alpha) * xs[i] + alpha * ys[j])
        ws.append(T[i, j])

    return np.array(zs), np.array(ws)
```

Because I can never get `matplotlib` to animate anything properly (more my fault
than `matplotlib`'s), I will generate a figure for each intermediate point, and
combine into a .gif using `imagemagick`.

```python
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

alphas = np.linspace(0, 1, num=32)
newcmp = LinearSegmentedColormap.from_list('RdBu', ['r', 'b'], N=32)

xlims = [min(np.min(xs[:, 0]), np.min(ys[:, 0])),
         max(np.max(xs[:, 0]), np.max(ys[:, 0]))]
ylims = [min(np.min(xs[:, 1]), np.min(ys[:, 1])),
         max(np.max(xs[:, 1]), np.max(ys[:, 1]))]
for i, alpha in enumerate(alphas):
    fig = plt.figure(figsize=(12, 9))
    zs, ws = position(xs, ys, T.value, alpha)
    cs = alpha * np.ones(len(zs)) * 255
    plt.scatter(zs[:, 0], zs[:, 1], s=2000*ws, color=newcmp(alpha))
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])
    plt.savefig('interpolant-{:02d}.png'.format(i))
```

The imagemagick command you want is:

```shell
convert -delay 10 -loop 0 *.png interpolant.gif
```

Which produces the following animation:
![](/ox-hugo/interpolant.gif)
