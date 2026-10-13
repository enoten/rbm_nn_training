# Restricted Boltzmann Machine on Fashion-MNIST

This project trains a binary Restricted Boltzmann Machine (RBM) on [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist). The weight and bias updates are the contrastive-divergence formulas written out explicitly. NumPy does the array arithmetic. No RBM class from scikit-learn, TensorFlow, or PyTorch is used.

Two scripts share that training code:

| Script | Dataset download |
| --- | --- |
| `rbm_fashion_mnist.py` | Standard-library HTTP download of the IDX files, then a local parser |
| `rbm_fashion_mnist_libs.py` | `tf.keras.datasets.fashion_mnist.load_data()` only. TensorFlow is not used for the model, the gradients, or the training loop |

## Setup

```bash
pip install numpy
python rbm_fashion_mnist.py
```

The TensorFlow variant also needs TensorFlow:

```bash
pip install numpy tensorflow
python rbm_fashion_mnist_libs.py
```

Default training in `main` uses 128 hidden units, learning rate `0.1`, CD-1, batch size 128, and 5 epochs. A run of `rbm_fashion_mnist.py` on the full training set produced this reconstruction cross-entropy on a fixed probe of 2,048 binarized training images:

| Epoch | Reconstruction cross-entropy |
| --- | --- |
| 1 | 0.2865 |
| 2 | 0.2580 |
| 3 | 0.2352 |
| 4 | 0.2200 |
| 5 | 0.2106 |

The same run scored 0.2166 on the first 2,048 binarized test images. The script writes:

- `output/reconstructions.pgm` — one test image from each of the 10 classes on the top row, and the RBM's mean-field reconstruction on the bottom row
- `output/rbm_weights.npz` — `W`, `a`, `b`, and the per-epoch history

Class order in the grid: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

## Data

Each Fashion-MNIST image is 28×28 grayscale, flattened to 784 visible units. Pixel values are divided by 255 so they lie in \([0, 1]\).

The RBM's visible units are binary. A gray value is treated as a probability. On every presentation of a mini-batch the code draws

$$
v_i \sim \mathrm{Bernoulli}(\text{pixel}_i).
$$

A pixel of 0.8 therefore becomes 1 about 80% of the time and 0 the rest of the time. Repeating this draw across epochs lets one set of binary weights represent intermediate intensities. The reconstruction monitor uses a fixed threshold instead, $$v_i$$ = 1 when the pixel is at least 0.5, so the reported curve is comparable from epoch to epoch.

Labels are used only to pick one test image per class for the picture. The RBM itself is unsupervised.

## The model

An RBM is an undirected graphical model with two layers and no connections inside a layer. Visible units \(v \in \{0,1\}^n\) are the pixels. Hidden units \(h \in \{0,1\}^m\) are latent features. With \(n = 784\) and \(m = 128\), the parameters are

- $$W$$ $$\in$$ $$\mathbb{R}^{n \times m}$$), the weight from visible unit $$i$$ to hidden unit $$j$$
- $$a$$ $$\in$$ $$\mathbb{R}^{n}$$, the visible bias
- $$b$$ $$\in$$ $$\mathbb{R}^{m}$$, the hidden bias

The energy of a joint configuration is

$$
E(v, h) = - a^\top v - b^\top h - v^\top W h.
$$

Low energy means the configuration is preferred. The Boltzmann distribution turns energies into probabilities:

$$
P(v, h) = \frac{1}{Z} \exp\bigl(-E(v, h)\bigr),
\qquad
Z = \sum_{v,h} \exp\bigl(-E(v, h)\bigr).
$$

$$Z$$ sums over all $$2^{n+m}$$ binary states. For 784 visible units and 128 hidden units that sum cannot be computed. Training has to avoid $$Z$$.

The distribution we care about for images is the marginal

$$
P(v) = \sum_h P(v, h) = \frac{1}{Z} \sum_h \exp\bigl(-E(v, h)\bigr).
$$

Summing out the hidden units produces an equivalent free energy of the visible vector alone:

$$
F(v) = -a^\top v - \sum_{j=1}^{m} \log\Bigl(1 + \exp\bigl(b_j + \sum_{i=1}^{n} v_i W_{ij}\bigr)\Bigr),
$$

$$
P(v) = \frac{\exp\bigl(-F(v)\bigr)}{\sum_{v'} \exp\bigl(-F(v')\bigr)}.
$$

The remaining sum is over \(2^{784}\) images, which is still intractable. Contrastive divergence, described below, replaces that sum with a short Markov chain.

### Why the conditionals are simple

The missing edges are what makes an RBM trainable. Because no two visible units share an edge, and no two hidden units share an edge, the conditional distributions factor into independent Bernoullis:

$$
P(h \mid v) = \prod_{j=1}^{m} P(h_j \mid v),
\qquad
P(v \mid h) = \prod_{i=1}^{n} P(v_i \mid h).
$$

Each factor is a logistic sigmoid \(\sigma(x) = 1/(1+e^{-x})\):

$$
P(h_j = 1 \mid v) = \sigma\Bigl(b_j + \sum_{i=1}^{n} v_i W_{ij}\Bigr),
$$

$$
P(v_i = 1 \mid h) = \sigma\Bigl(a_i + \sum_{j=1}^{m} W_{ij} h_j\Bigr).
$$

In matrix form, for a batch whose rows are visible vectors,

$$
P(h=1 \mid v) = \sigma(v W + b),
\qquad
P(v=1 \mid h) = \sigma(h W^\top + a).
$$

These are `p_h_given_v` and `p_v_given_h`. All hidden units can be evaluated, or sampled, in one matrix product. The same is true of the visible units. That block independence is the reason Gibbs sampling on an RBM is a pair of alternating matrix multiplies rather than a sweep over individual pixels.

### Initialization

$$W$$ is drawn from $$mathcal{N}(0, 0.01)$$. Hidden biases $$b$$ start at 0. Visible biases are set to the inverse-sigmoid of the data mean, which is the maximum-likelihood bias of a set of independent pixels:

$$
a_i = \log \frac{p_i}{1 - p_i},
\qquad
p_i = \mathrm{mean}(v_i),
$$

with $$p_i$$ clipped slightly away from 0 and 1. Training then only has to learn the correlations that $$W$$ and $$b$$ represent.

## Gibbs sampling

Gibbs sampling draws from a joint distribution by repeatedly replacing each block of variables with a fresh draw from its conditional distribution given everything else. For a joint $$P(v, h)$$ the transition is

$$
h^{(t)} \sim P(h \mid v^{(t)}),
\qquad
v^{(t+1)} \sim P(v \mid h^{(t)}).
$$

This is a Markov chain. The RBM joint $$P(v, h)$$ is stationary for the chain: if $$(v, h)$$ is already a sample from the joint, one more Gibbs step is still a sample from the joint. Under the usual regularity conditions (the chain can reach every state of positive probability, which it can here because every conditional probability is strictly between 0 and 1), the distribution of $$(v^{(t)}, h^{(t)})$$ converges to $$P(v, h)$$ no matter where the chain started. The early samples are biased toward the start state. That burn-in is the reason a short chain and a fully mixed chain estimate different things, which is the gap between contrastive divergence and exact maximum likelihood.

### Block Gibbs on the bipartite graph

A general Boltzmann machine would resample units one at a time, because each unit's conditional depends on its neighbors in both layers. An RBM's missing intra-layer edges remove that dependence. Given \(v\), the hidden units are independent, so one draw of a uniform random matrix $$U$$ produces the whole hidden sample:

$$
h_j =
\begin{cases}
1 & \text{if } U_j < P(h_j = 1 \mid v), \\
0 & \text{otherwise.}
\end{cases}
$$

Given that $$h$$, the same construction produces every visible unit at once. One full step of the chain is therefore:

1. Compute $$P(v \mid h) = \sigma(h W^\top + a)$$.
2. Draw $$v' \sim \mathrm{Bernoulli}\bigl(P(v \mid h)\bigr)$$.
3. Compute $$P(h \mid v') = \sigma(v' W + b)$$.

That is `gibbs_hvh`. The function returns the sampled visible state, which is the state the chain occupies, and the hidden *probability* given that visible state. The probability is kept because the learning rule below wants an expectation over \(h\), which for a Bernoulli is exactly this sigmoid.

### A numerical step

Take two visible units, one hidden unit, $$W$$ = $$\begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}$$, $$a = 0$$, $$b = 0$$, and a data vector $$v^{(0)} = (1, 0)$$.

$$
P(h=1 \mid v^{(0)}) = \sigma(0.5) \approx 0.622.
$$

Suppose the Bernoulli draw returns $$h = 1$$. The reconstruction probabilities are

$$
P(v_i = 1 \mid h=1) = \sigma(0.5) \approx 0.622
\quad \text{for both } i=1 \text{ and } i=2.
$$

A second pair of coin flips might return $$v^{(1)} = (1, 1)$$. Then

$$
P(h=1 \mid v^{(1)}) = \sigma(0.5 + 0.5) = \sigma(1) \approx 0.731.
$$

The chain has moved from a one-hot pixel pattern toward a pattern in which both pixels are on, because the single hidden unit ties them together. After many such steps the histogram of visited \(v\) approaches the model marginal \(P(v)\). After one step it is only a local perturbation of the data. Contrastive divergence deliberately stops at that local perturbation.

### Samples and probabilities play different roles

Sampling chooses a concrete binary state so the chain can move. The learning statistics, though, are expectations. For a Bernoulli hidden unit the conditional expectation is the probability itself:

$$
\mathbb{E}[h_j \mid v] = P(h_j = 1 \mid v).
$$

Using that probability in the gradient averages over the $$2^m$$ hidden configurations analytically. Using a single sample of $$h$$ in the same slot would be a much noisier estimate of the same quantity. Hinton's practical guide therefore recommends:

- draw binary samples to propagate the Gibbs chain
- plug probabilities into the outer products that update $$W$$, $$a$$, and $$b$$.

The positive phase in `cd_gradients` follows that split. $$h_0$$ used in $$V^\top H$$ is $$P(h \mid v)$$. A separate sample of $$h$$ is drawn only to start `gibbs_hvh`. On the negative end, $$v'$$ is the binary state the chain actually reached, and $$h'$$ is $$P(h \mid v')$$. Intermediate steps, when `cd_k > 1`, resample $$h$$ so the next visible draw is conditioned on a binary hidden state.

### Mean-field reconstruction

`reconstruct` does not sample. It replaces each conditional draw by its mean:

$$
\hat h = P(h=1 \mid v),
\qquad
\hat v = P(v=1 \mid \hat h).
$$

$$\hat v$$ is a grayscale image in $$(0, 1)^{784}$$. It is the picture in the bottom row of `reconstructions.pgm`. This one-pass reconstruction is also what the cross-entropy monitor scores. It is a diagnostic of whether the conditional $$P(v \mid h)$$ sends data back to something like itself. It is not the log-likelihood, and it can keep improving after the chain used for learning has stopped matching the model distribution closely.

## Contrastive divergence

Maximum likelihood would move the parameters to increase $$\sum_{\text{data}} \log P(v)$$. For one observation the gradient with respect to a parameter $$\theta$$ is an exact difference of two expectations:

$$
\frac{\partial \log P(v)}{\partial \theta}
=
-\underbrace{\frac{\partial F(v)}{\partial \theta}}_{\text{positive phase}}
+
\underbrace{\mathbb{E}_{v' \sim P}\!\left[\frac{\partial F(v')}{\partial \theta}\right]}_{\text{negative phase}}.
$$

The positive phase is "what the data wants." It is easy, because the hidden units can be summed out in closed form. The negative phase is "what the model currently believes." It is an average of the same statistic under the model's own image distribution $$(P(v)$$, and that average is the intractable sum over images.

Writing the derivatives out in the original weights and biases gives the standard RBM statistics. Angle brackets $$\langle \cdot \rangle_{\mathrm{data}}$$ mean the average under the data vectors together with $$P(h \mid v)$$. Angle brackets $$\langle \cdot \rangle_{\mathrm{model}}$$ mean the average under the joint $$P(v, h)$$:

$$
\frac{\partial \log P(v)}{\partial W_{ij}}
$$
=
$$
\langle v_i h_j \rangle_{\mathrm{data}} - \langle v_i h_j \rangle_{\mathrm{model}},
$$

$$
\frac{\partial \log P(v)}{\partial a_i}
$$
=
$$
\langle v_i \rangle_{\mathrm{data}} - \langle v_i \rangle_{\mathrm{model}},
$$

$$
\frac{\partial \log P(v)}{\partial b_j}
=
\langle h_j \rangle_{\mathrm{data}}
-
\langle h_j \rangle_{\mathrm{model}}.
$$

The positive half of the weight gradient is $$v_i \, P(h_j=1 \mid v)$$. The positive half of the hidden-bias gradient is $$P(h_j=1 \mid v)$$. The positive half of the visible-bias gradient is the pixel itself. All three negative halves require samples from the model.

### The positive phase

For a mini-batch $$V$$ with $$N$$ rows, the code computes

$$
H = \sigma(V W + b)
$$

and uses the rank-1 (per row) associations $$V^\top H$$. Dividing by $$N$$ turns the sum into a mean. This term increases $$W_{ij}$$ when visible unit $$i$$ and hidden unit $$j$$ turn on together in the data, and it increases $$a_i$$ and $$b_j$$ when those units are on more often than the model expects.

### The negative phase, and why a finished chain is expensive

A faithful Monte Carlo estimate of the model term starts a Gibbs chain and discards a long burn-in, then averages $$v_i$$ $$h_j$$ over many draws from the stationary distribution. Every parameter update on Fashion-MNIST would then cost hundreds or thousands of full-batch reconstructions. Early in training the model is almost random, so most of that work is spent sampling a distribution that the next update immediately changes.

### The CD-k shortcut

Contrastive divergence (Hinton, 2002) changes the start state and the length of the chain.

1. Start the chain at the data vector $$v^{(0)}$$, not at a random image.
2. Run $$k$$ steps of block Gibbs sampling. $$k = 1$$ is the usual choice and the default here (`cd_k=1`).
3. Replace the model expectation by the statistics of that nearby state $$(v^{(k)}, h^{(k)})$$.

The resulting parameter steps, with learning rate $$\eta$$, are

$$
W \leftarrow W + \eta \Bigl(
\langle v h^\top \rangle_{0}
-
\langle v h^\top \rangle_{k}
\Bigr),
$$

$$
a \leftarrow a + \eta \Bigl(
\langle v \rangle_{0}
-
\langle v \rangle_{k}
\Bigr),
$$

$$
b \leftarrow b + \eta \Bigl(
\langle h \rangle_{0}
-
\langle h \rangle_{k}
\Bigr).
$$

Subscript $$0$$ is the data distribution (the positive phase). Subscript $$k$$ is the distribution of the chain after $$k$$ Gibbs steps (the stand-in for the negative phase). For a mini-batch the code evaluates these as

$$
\Delta W = \frac{1}{N}\bigl(V^\top H - {V^{-}}^\top H^{-}\bigr),
\quad
\Delta a = \mathrm{mean}(V - V^{-}),
\quad
\Delta b = \mathrm{mean}(H - H^{-}),
$$

and then

$$
W \leftarrow W + \eta\,\Delta W,
\quad
a \leftarrow a + \eta\,\Delta a,
\quad
b \leftarrow b + \eta\,\Delta b.
$$

$$V^{-}$$ is the matrix of visible samples after $$k$$ steps. $$H^{-}$$ is $$P(h=1 \mid V^{-})$$, not a sample. These three lines are the body of `update`.

### What one CD-1 update does to an image

CD-1 is easier to read as a reconstruction update than as a likelihood proof.

- Clamp the pixels to a real Fashion-MNIST image and compute which hidden features turn on.
- Sample those features and ask the model which pixels it would generate.
- If a weight helped explain the real image, the positive term increases it.
- If that same weight also helped invent the reconstructed image, the negative term decreases it by about the same amount, and the weight barely moves.
- If the reconstruction puts mass on pixels that were off in the data, or misses pixels that were on, the two terms disagree and the weight moves so that the next reconstruction is closer to the data.

So CD-1 chases the model wherever the model disagrees with the data in the immediate neighborhood of the training images. It never has to draw a shoe while it is studying a shirt, unless a single Gibbs step can already wander that far. Early in training it cannot. That is both the source of the speed and the source of the bias.

### CD is a biased estimator of the likelihood gradient

The negative-phase average in maximum likelihood is over the whole model distribution, including images the chain would reach only after mixing. CD-k averages over images reachable in $$k$$ steps from the data. Those two distributions differ, so the CD update is not the gradient of $$\log P(v)$$.

The bias shrinks as $$k$$ grows, because the chain forgets its data initialization and approaches stationarity. The cost grows in the same direction: each extra step is another pair of matrix products of shape $$(N, 784)\times(784, 128)$$ and $$(N, 128)\times(128, 784)$$. CD-1 is a good default for learning filters and reconstructions. Raising `cd_k` is the direct way, in this code, to spend more computation on a less biased negative phase.

A related algorithm, persistent contrastive divergence (PCD, or stochastic maximum likelihood), keeps fantasy particles from the previous update and continues their chains instead of restarting at the data. This repository implements CD-k with a fresh chain from each mini-batch. It does not implement PCD.

### Mini-batches

Each epoch shuffles the 60,000 training rows and walks through them in blocks of 128. The gradients above are means over the block, so a larger batch reduces the variance of both the positive statistics and the $$k$$-step samples. The learning rate `0.1` is applied directly to those means. There is no momentum and no weight decay. Adding either would be an extra term on top of the three CD updates.

Inside the batch loop the pixels are re-binarized with `sample_bernoulli`, then `update` runs one CD-k step. After the epoch, reconstruction cross-entropy is computed on the first 2,048 training images, thresholded at 0.5:

$$
\mathrm{CE}
=
-\frac{1}{N n}
\sum
\Bigl(
v_i \log \hat v_i
+
(1-v_i)\log(1-\hat v_i)
\Bigr),
$$

where $$\hat v = P\bigl(v=1 \mid P(h=1 \mid v)\bigr)$$ and probabilities are clipped to $$[10^{-7}, 1-10^{-7}]$$ before the logarithm. A falling CE means the one-step mean-field reconstruction is assigning higher probability to the thresholded pixels. It is the number printed as `reconstruction cross-entropy`.

### Walk-through of `cd_gradients` for general $$k$$

For a binary batch $$V$$:

1. Positive hidden probabilities: $$H = \sigma(VW + b)$$.
2. Draw $$h \sim \mathrm{Bernoulli}(H)$$. This sample starts the chain and does not enter $$V^\top H$$.
3. Repeat `cd_k` times:
   - Draw $$v^{-} \sim \mathrm{Bernoulli}(\sigma(h W^\top + a))$$.
   - Set $$H^{-} = \sigma(v^{-} W + b)$$.
   - On every iteration except the last, draw a new $$h \sim \mathrm{Bernoulli}(H^{-})$$ so the next visible sample is conditioned on a binary hidden state.
4. Form $$\Delta W$$, $$\Delta a$$, and $$\Delta b$$ from $$(V, H)$$ and $$(v^{-}, H^{-})$$.

For the default $$k = 1$$ the loop body runs once, and the chain is exactly: data $$\to$$ sample $$h$$ $$\to$$ sample $$v^{-}$$ $$\to$$ probability $$H^{-}$$.

## How the pieces fit together

```text
pixel in [0, 1]
    -> Bernoulli draw                 binary visible vector v
    -> sigmoid(v W + b)               positive hidden probability
    -> Bernoulli draw                 hidden sample that drives Gibbs
    -> k steps of gibbs_hvh           negative visible sample and hidden probability
    -> CD updates of W, a, b          explicit formulas in update()
    -> sigmoid(sigmoid(v W + b) W^T + a)
                                      mean-field reconstruction used for the monitor and the PGM
```

## References

- Ackley, D. H., Hinton, G. E., and Sejnowski, T. J. (1985). A learning algorithm for Boltzmann machines. *Cognitive Science*, 9(1), 147–169.
- Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. *Neural Computation*, 14(8), 1771–1800. Introduces CD-k.
- Hinton, G. E. (2012). A practical guide to training restricted Boltzmann machines. In *Neural Networks: Tricks of the Trade* (2nd ed.). The recommendation to put probabilities in the gradient and samples in the chain comes from this note.
- Tieleman, T. (2008). Training restricted Boltzmann machines using approximations to the likelihood gradient. *ICML*. Persistent contrastive divergence.
- Xiao, H., Rasul, K., and Vollgraf, R. (2017). Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms. arXiv:1708.07747.
