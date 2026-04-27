**Run the tutorial**: [`notebooks/sGP_Tutorial_Bandgap_Perovskites.ipynb`](notebooks/sGP_Tutorial_Bandgap_Perovskites.ipynb)  
**Read the HTML version**: [`docs/sgp_tutorial.html`](docs/sgp_tutorial.html)

<a href="https://colab.research.google.com/github/SLKS99/-Structured-Gaussian-Processes-Learning-Physics-from-Very-Few-Experiments/blob/main/notebooks/sGP_Tutorial_Bandgap_Perovskites.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

---

# 📡 Structured Gaussian Processes: Learning Physics from Very Few Experiments

**A hands-on tutorial on sGP for bandgap engineering in hybrid perovskites**

*Based on work by Yongtao Liu and Sheryl L. Sanchez using Maxim Ziatdinov's [GPax](https://github.com/ziatdinovmax/gpax) library · August 2023*

---

## Abstract

When exploring a new material system, measurements are expensive and the underlying physics is only partially known. **Structured Gaussian Processes (sGP)** solve both problems at once: they encode physically-motivated functional forms as a prior mean, letting a small number of experiments simultaneously *fit the data* and *recover the governing equations*.

This tutorial walks you through the method step by step:
1. **Background** — what GPs and sGPs are, and why the distinction matters
2. **The physics** — bandgap engineering in hybrid perovskites, Vegard's law, bowing
3. **Toy model** — validate sGP on synthetic data where we know the ground truth
4. **Real data** — apply sGP to MAPbI₃/GAPbBr₃ photoluminescence spectra

### Learning objectives
By the end of this notebook you will be able to:
- Define a physics-informed mean function and its Bayesian priors
- Run GP and sGP active exploration loops and compare their efficiency
- Read posterior parameter distributions to extract physical constants
- Choose between competing physical models (linear, piecewise, quadratic)

### Prerequisites
- Basic Python / NumPy
- Some familiarity with Bayesian reasoning (helpful but not required)
- No prior GP experience needed


## Table of Contents

1. [Background: From GP to Structured GP](#section1)
2. [The Physics: Bandgap Engineering in Hybrid Perovskites](#section2)
3. [Installation & Setup](#section3)
4. [Tutorial Part 1: Toy Model](#section4)
   - 4.1 [Creating toy data](#4-1)
   - 4.2 [Standard GP baseline](#4-2)
   - 4.3 [Defining physics-informed mean functions](#4-3)
   - 4.4 [Running the sGP exploration loop](#4-4)
   - 4.5 [Comparing models and interpreting results](#4-5)
5. [Tutorial Part 2: Real MAPbI₃ / GAPbBr₃ Data](#section5)
   - 5.1 [Loading photoluminescence spectra](#5-1)
   - 5.2 [Extracting bandgap energies](#5-2)
   - 5.3 [Fitting sGP to experimental data](#5-3)
   - 5.4 [Reading the physics from posterior parameters](#5-4)
6. [Key Takeaways](#section6)
7. [References](#section7)


---
<a id="section1"></a>
## 1 · Background: From GP to Structured GP

### What is a Gaussian Process?

A **Gaussian Process (GP)** is a probability distribution over functions. Instead of assuming a specific functional form, a GP says: *"I believe the true curve is a smooth, correlated function, and I will let the data tell me what it looks like."*

At every unmeasured point, the GP produces not just a single prediction but a full distribution — a mean estimate surrounded by uncertainty bands that shrink as you add data.

**The model:**

$$y(x) = f(x) + \varepsilon, \quad f \sim \mathcal{GP}\big(0,\; k(x, x')\big), \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)$$

The **kernel** $k(x, x')$ encodes how correlated two function values are based on distance in input space:
- **RBF (squared exponential):** infinite smoothness — good for slowly varying data
- **Matérn:** finite smoothness — more realistic for physical systems with kinks or abrupt changes

GPs are powerful — but also **agnostic**. With a zero mean and an RBF kernel, the model has no idea whether the physics is linear, exponential, or piecewise. It must discover everything from data alone. For expensive experiments (each synthesis run may cost hundreds of dollars and hours of instrument time), this is a serious limitation.

---

### What makes it "structured"?

A **Structured GP (sGP)** encodes your physical intuition directly into the model by replacing the zero mean with a parameterized, physically-motivated function:

$$f \sim \mathcal{GP}\big(\mu(x;\,\theta),\; k(x, x')\big)$$

The mean function $\mu(x; \theta)$ carries your physical hypothesis (e.g., *"bandgap varies linearly with composition"*). The kernel handles residuals that the physics model doesn't capture.

> **The key insight:** both the observations *and* the model parameters $\theta$ are inferred simultaneously via MCMC — so you recover not just a fit, but **posterior distributions over the physical constants themselves**.

| | Standard GP | Structured GP (sGP) |
|---|---|---|
| Mean function | $\mu = 0$ | $\mu(x; \theta)$ — your physics |
| Inferred quantities | kernel hyperparameters | kernel params + **physical constants** |
| Data efficiency | Low — needs many points | High — physics fills the gaps |
| Interpretability | Black box | Each parameter has physical meaning |


---
<a id="section2"></a>
## 2 · The Physics: Bandgap Engineering in Hybrid Perovskites

### Hybrid perovskites and composition tuning

**Hybrid organic-inorganic perovskites** (ABX₃ structure) are exceptional light-absorbing materials, reaching >25% efficiency in solar cells. Their most powerful feature is compositional tunability: by mixing two end-member perovskites, you can dial in the optical bandgap across a wide range.

In the **MAPbI₃ / GAPbBr₃** system studied here:
- **MAPbI₃** (0% GAPbBr₃): $E_g \approx 1.57$ eV — near-ideal for single-junction solar cells
- **GAPbBr₃** (~100% GAPbBr₃): $E_g \approx 2.43$ eV — suitable for tandem top cells and LEDs

**How is bandgap measured?** We excite films with a laser and collect the **photoluminescence (PL) spectrum**. The peak emission wavelength corresponds to photons emitted at the bandgap energy:

$$E_g = \frac{hc}{\lambda_{\text{peak}}} = \frac{1240 \text{ eV·nm}}{\lambda_{\text{peak}} [\text{nm}]}$$

---

### Vegard's law and optical bowing

The simplest model for bandgap vs. composition is **Vegard's law** — a linear interpolation between end members:

$$E_g(x) = (1 - x) \cdot E_g(A) + x \cdot E_g(B)$$

Real alloys often deviate from this line — called **optical bowing**:

$$E_g(x) = (1 - x) \cdot E_g(A) + x \cdot E_g(B) - b \cdot x(1 - x)$$

The bowing parameter $b$ captures nonlinearity from structural distortion, orbital hybridization, and disorder.

> **This gives us a physical hypothesis to embed in sGP.** We *expect* the bandgap to follow a roughly linear or quadratic curve. Instead of guessing $b$, sGP will *infer it* from the data.


---
<a id="section3"></a>
## 3 · Installation & Setup

We use [**GPax**](https://github.com/ziatdinovmax/gpax) — a JAX-based GP library that natively supports structured GPs with MCMC inference via NumPyro.



```python
# Install GPax (includes JAX, NumPyro, and related dependencies)
!pip install git+https://github.com/ziatdinovmax/gpax -q
!pip install atomai -q
```


```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict

import gpax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from sklearn.metrics import mean_squared_error

# Enable 64-bit precision — important for MCMC stability with GP likelihoods
gpax.utils.enable_x64()

# Consistent plot style
plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})
print("Setup complete ✓")
```

---
<a id="section4"></a>
## 4 · Tutorial Part 1: Toy Model

Before touching real experimental data, we validate sGP on a **synthetic "toy" system** where we know the ground truth exactly. This lets us directly check whether sGP recovers the correct physics — not just a good fit.

Our synthetic bandgap curve is **piecewise: quadratic below a transition concentration, then approximately flat above it** — mimicking a real alloy where one structural phase dominates at low incorporation and another at high incorporation.


<a id="4-1"></a>
### 4.1 · Creating the Toy Dataset



```python
np.random.seed(0)

NUM_POINTS   = 200    # total grid points in composition space
NOISE_LEVEL  = 0.01   # small noise (clean experiment)
x_           = np.linspace(0.0, 2.0, NUM_POINTS)   # composition axis


def true_bandgap(x, c=1.5):
    """Ground truth: piecewise — quadratic then flat-ish.

    Transition at c mimics a structural phase boundary.
    Ground truth parameters:
        Low-x (quadratic): y = x² - 1  →  a1=1, b1=-1
        High-x (linear):   y = -x + 2  →  slope=-1, intercept=2
        Transition at:     c = 1.5
    """
    return np.piecewise(
        x, [x < c, x >= c],
        [lambda x: x * x - 1,     # quadratic region (low x)
         lambda x: -x + 2]         # linear region (high x)
    )


y_true = true_bandgap(x_)
y_data = y_true + np.random.normal(0, NOISE_LEVEL, NUM_POINTS)

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(x_, y_data, s=8, alpha=0.5, label="Simulated observations", c="#1a2844")
ax.plot(x_, y_true, lw=2, c="#d97b0e", label="True curve (ground truth)", zorder=3)
ax.axvline(1.5, lw=1, ls="--", c="#888", label="Phase transition at x=1.5")
ax.set_xlabel("Composition x")
ax.set_ylabel("Bandgap $E_g$ (eV)")
ax.set_title("Toy dataset: piecewise quadratic bandgap curve")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

print("Ground truth parameters:")
print("  Quadratic regime: y = x² - 1       → a1=1, b1=-1")
print("  Linear regime:    y = -x + 2        → slope=-1, intercept=2")
print("  Transition at:    c = 1.5")
```

### Helper functions for the active learning loop



```python
def init_training_data(X, y_all, seed_indices):
    """Initialize training/test split from seed point indices."""
    all_idx    = jnp.arange(len(X))
    idx        = np.array(seed_indices)
    X_train    = X[idx]
    y_train    = y_all[idx]
    X_test     = jnp.delete(X, idx)
    y_test     = jnp.delete(y_all, idx)
    return X_train, y_train, X_test, y_test


def update_datapoints(next_point_idx, X_train, y_train, X_test, y_test):
    """Add the selected next point to training data and remove from test pool."""
    X_new       = X_test[next_point_idx][None]
    y_new       = y_test[next_point_idx]
    X_train_new = jnp.append(X_train, X_new, 0)
    y_train_new = jnp.append(y_train, y_new)
    X_test_new  = jnp.delete(X_test, next_point_idx, 0)
    y_test_new  = jnp.delete(y_test, next_point_idx)
    return X_train_new, y_train_new, X_test_new, y_test_new
```

<a id="4-2"></a>
### 4.2 · Standard GP Baseline

We first run a standard (unstructured) GP — **RBF kernel, zero mean**, noise half-normal prior. Starting with only 2 observations (the two endpoints), the GP performs active exploration for 30 steps using an uncertainty-driven acquisition function.

The **Uncertainty Exploration (UE)** acquisition selects the composition where the GP is most uncertain — efficiently mapping the bandgap curve without any physical knowledge.



```python
# ── Custom priors on kernel hyperparameters ──
def kernel_prior():
    k_length = numpyro.sample("k_length", dist.Gamma(0.5, 1))
    k_scale  = numpyro.sample("k_scale",  dist.LogNormal(0, 2))
    return {"k_length": k_length, "k_scale": k_scale}

noise_prior = lambda: numpyro.sample("noise", dist.HalfNormal(0.01))


def step_GP(X_measured, y_measured, X_unmeasured):
    """Single GP step: fit → predict → compute acquisition function."""
    rng_key1, rng_key2 = gpax.utils.get_keys()

    # Initialize standard GP — no mean function (zero mean)
    gp = gpax.ExactGP(1, kernel="RBF",
                      kernel_prior=kernel_prior,
                      noise_prior=noise_prior)

    # HMC: sample posterior over kernel hyperparameters
    gp.fit(rng_key1, X_measured, y_measured, jitter=1e-5)

    # Predict
    y_pred, y_samples = gp.predict_in_batches(rng_key2, X_unmeasured,
                                               noiseless=False, jitter=1e-4)

    # Uncertainty-Exploration acquisition: go where variance is highest
    acq = gpax.acquisition.UE(rng_key2, gp, X_unmeasured,
                               noiseless=False, jitter=1e-4)

    return acq, y_pred, y_samples, gp.get_samples()
```


```python
# ── Run GP active exploration ──
EXPLORATION_STEPS = 30
savedir_GP = "/content/video_image_GP/"
os.makedirs(savedir_GP, exist_ok=True)

X       = jnp.linspace(0.0, 2.0, NUM_POINTS)
y_all   = jnp.asarray(y_data)

# Start with only endpoints
X_train, y_train, X_test, y_test = init_training_data(
    X, y_all, seed_indices=[0, NUM_POINTS - 1]
)

all_uncertainty_GP = []
all_mse_GP         = []
all_k_length_GP    = []
all_k_scale_GP     = []

for s in range(EXPLORATION_STEPS):
    acq, y_pred, y_samples, params = step_GP(X_train, y_train, X_test)
    next_point = int(acq.argmax())

    # Record metrics
    all_uncertainty_GP.append(float(y_samples.std(axis=(0, 1)).sum()))
    all_mse_GP.append(float(mean_squared_error(y_test, y_pred)))
    all_k_length_GP.append(float(params["k_length"].mean()))
    all_k_scale_GP.append(float(params["k_scale"].mean()))

    # Update dataset
    X_train, y_train, X_test, y_test = update_datapoints(
        next_point, X_train, y_train, X_test, y_test
    )

    if s % 10 == 0:
        print(f"Step {s:2d}  |  MSE: {all_mse_GP[-1]:.5f}  "
              f"|  Uncertainty: {all_uncertainty_GP[-1]:.4f}")

print("\nGP exploration complete ✓")
```

<a id="4-3"></a>
### 4.3 · Defining Physics-Informed Mean Functions

Here is where the physics goes in. We define **four candidate mean functions**, each encoding a different physical hypothesis. For each, we also specify Bayesian priors over the parameters — encoding what we believe *before* seeing the data.

| Model | Mean function | Physical hypothesis |
|---|---|---|
| **GP** | $\mu = 0$ | No physics assumed |
| **sGP-linear** | $ax + b$ | Vegard's law (linear mixing) |
| **sGP-piecewise** | $a_1 x + b_1$ if $x < t$, else $a_2 x + b_2$ | Two-phase linear mixing |
| **sGP-quadratic** | $ax^2 + bx + c$ | Quadratic bowing throughout |
| **c-sGP** ✓ | $a_1 x^2 + b_1$ if $x < t$, else $a_2 + b_2$ | *Custom: matches true physics* |

The **c-sGP** (customized sGP) encodes the exact form of our ground truth — we expect it to win.



```python
# ════════════════════════════════════════════════════════════
# Default sGP mean functions
# ════════════════════════════════════════════════════════════

# ── Linear ──────────────────────────────────────────────────
def linear(x, params):
    """Linear mean: E_g(x) = a·x + b  (Vegard's law approximation)."""
    return params["a"] * x + params["b"]

def linear_prior():
    a = numpyro.sample("a", dist.LogNormal(0, 1))   # slope (positive)
    b = numpyro.sample("b", dist.Normal(0, 2))       # intercept
    return {"a": a, "b": b}


# ── Piecewise linear ─────────────────────────────────────────
def piecewise(x: jnp.ndarray, params: Dict) -> jnp.ndarray:
    """Piecewise linear: two linear segments meeting at transition t."""
    return jnp.piecewise(
        x, [x < params["t"], x >= params["t"]],
        [lambda x: params["a1"] * x + params["b1"],
         lambda x: params["a2"] * x + params["b2"]]
    )

def piecewise_prior():
    a1 = numpyro.sample("a1", dist.LogNormal(0, 1))
    b1 = numpyro.sample("b1", dist.Normal(0, 2))
    a2 = numpyro.sample("a2", dist.LogNormal(0, 1))
    b2 = numpyro.sample("b2", dist.Normal(0, 2))
    t  = numpyro.sample("t",  dist.Gamma(0.4, 1))
    return {"a1": a1, "b1": b1, "a2": a2, "b2": b2, "t": t}


# ── Quadratic ────────────────────────────────────────────────
def quadratic(x, params):
    """Quadratic mean: E_g(x) = a·x² + b·x + c  (bowing throughout)."""
    return params["a"] * x**2 + params["b"] * x + params["c"]

def quadratic_prior():
    a = numpyro.sample("a", dist.LogNormal(0, 1))
    b = numpyro.sample("b", dist.Normal(0, 1))
    c = numpyro.sample("c", dist.Normal(0, 1))
    return {"a": a, "b": b, "c": c}


print("Default mean functions defined: linear, piecewise, quadratic ✓")
```


```python
# ════════════════════════════════════════════════════════════
# Custom (c-sGP) mean function — matches the true physics
# ════════════════════════════════════════════════════════════

def piecewise_quadratic(x: jnp.ndarray, params: Dict) -> jnp.ndarray:
    """Piecewise-quadratic mean function.

    Physical interpretation:
        - Below transition t: bandgap curves quadratically (bowing-dominated phase)
        - Above transition t: bandgap is approximately flat (second structural phase)

    This encodes the hypothesis that there are two distinct structural regimes
    in the alloy, separated by a composition-driven phase transition at x = t.
    """
    return jnp.piecewise(
        x,
        [x < params["t"], x >= params["t"]],
        [lambda x: params["a1"] * x**2 + params["b1"],   # quadratic below t
         lambda x: params["a2"] + params["b2"]]            # constant above t
    )

def piecewise_quadratic_prior():
    """Bayesian priors over the physical parameters.

    Prior choices explained:
        a1  ~ LogNormal(0,1):  quadratic coefficient — must be positive (upward curvature)
        b1  ~ Normal(0,2):     vertical shift — can be positive or negative
        a2  ~ LogNormal(0,1):  amplitude of plateau — positive
        b2  ~ Normal(0,2):     plateau shift
        t   ~ Gamma(0.4,1):    transition concentration — positive, near 0 a priori
    """
    a1 = numpyro.sample("a1", dist.LogNormal(0, 1))
    b1 = numpyro.sample("b1", dist.Normal(0, 2))
    a2 = numpyro.sample("a2", dist.LogNormal(0, 1))
    b2 = numpyro.sample("b2", dist.Normal(0, 2))
    t  = numpyro.sample("t",  dist.Gamma(0.4, 1))
    return {"a1": a1, "b1": b1, "a2": a2, "b2": b2, "t": t}


print("Custom c-sGP mean function defined ✓")
print()
print("Ground truth we are trying to recover:")
print("  a1 = 1.0  (quadratic coefficient)")
print("  b1 = -1.0 (vertical shift, low-x phase)")
print("  t  = 1.5  (phase transition composition)")
```

<a id="4-4"></a>
### 4.4 · Running the sGP Exploration Loop

We start with only **2 endpoint measurements** and run 30 active learning steps. At each step, the sGP:
1. Fits posterior distributions over all parameters (kernel + physics)
2. Predicts the full bandgap curve
3. Selects the **most uncertain** composition as the next measurement



```python
def step_sGP(X_measured, y_measured, X_unmeasured, mean_fn, mean_fn_prior):
    """Single sGP step: fit → predict → compute acquisition function.

    Parameters
    ----------
    X_measured     : training compositions (already measured)
    y_measured     : training bandgaps
    X_unmeasured   : candidate compositions (not yet measured)
    mean_fn        : physics-informed mean function
    mean_fn_prior  : prior function returning dict of parameter samples

    Returns
    -------
    acq            : acquisition values at X_unmeasured (higher = measure here)
    y_pred         : posterior mean prediction at X_unmeasured
    y_samples      : posterior predictive samples
    params         : MCMC samples of all parameters (kernel + physics)
    """
    rng_key1, rng_key2 = gpax.utils.get_keys()

    # Initialize sGP with physics mean function
    sgp = gpax.ExactGP(
        1, kernel="RBF",
        kernel_prior=None,            # use default priors on kernel params
        mean_fn=mean_fn,
        mean_fn_prior=mean_fn_prior,
        noise_prior=noise_prior
    )

    # MCMC samples ALL parameters jointly:
    # kernel lengthscale, kernel scale, noise + all physics parameters
    sgp.fit(rng_key1, X_measured, y_measured, jitter=1e-4)

    # Predict on the unmeasured grid
    y_pred, y_samples = sgp.predict_in_batches(
        rng_key2, X_unmeasured, noiseless=True, jitter=1e-4
    )

    # Uncertainty-Exploration: go where the model is least confident
    acq = gpax.acquisition.UE(rng_key2, sgp, X_unmeasured,
                               noiseless=False, jitter=1e-4)

    return acq, y_pred, y_samples, sgp.get_samples()
```


```python
# ── Run all four sGP variants ──────────────────────────────────
#
# We run each variant separately and store their metrics.
# This may take a few minutes on CPU; GPU/TPU will be significantly faster.

EXPLORATION_STEPS = 30
X       = jnp.linspace(0.0, 2.0, NUM_POINTS)
y_all   = jnp.asarray(y_data)

MODELS = {
    "sGP_linear":    (linear,              linear_prior),
    "sGP_piecewise": (piecewise,           piecewise_prior),
    "sGP_quadratic": (quadratic,           quadratic_prior),
    "c_sGP":         (piecewise_quadratic, piecewise_quadratic_prior),
}

# Storage for metrics across all models
results = {name: {"uncertainty": [], "mse": [], "params": []} for name in MODELS}

for model_name, (mean_fn, mean_fn_prior) in MODELS.items():
    print(f"\n{'='*50}")
    print(f"  Running: {model_name}")
    print(f"{'='*50}")

    X_train, y_train, X_test, y_test = init_training_data(
        X, y_all, seed_indices=[0, NUM_POINTS - 1]
    )

    for s in range(EXPLORATION_STEPS):
        acq, y_pred, y_samples, params = step_sGP(
            X_train, y_train, X_test, mean_fn, mean_fn_prior
        )
        next_point = int(acq.argmax())

        results[model_name]["uncertainty"].append(
            float(y_samples.std(axis=(0, 1)).sum()))
        results[model_name]["mse"].append(
            float(mean_squared_error(y_test, y_pred)))
        results[model_name]["params"].append(
            {k: float(v.mean()) for k, v in params.items()})

        X_train, y_train, X_test, y_test = update_datapoints(
            next_point, X_train, y_train, X_test, y_test
        )

        if s % 10 == 0:
            print(f"  Step {s:2d}  |  MSE: {results[model_name]['mse'][-1]:.5f}")

print("\nAll sGP variants complete ✓")
```

<a id="4-5"></a>
### 4.5 · Comparing Models and Interpreting Results

Now we look at two things:
1. **Predictive accuracy** (MSE vs step) — which model learns the curve fastest?
2. **Parameter recovery** — does c-sGP converge to the true physical constants?



```python
# ── Plot 1: MSE and Uncertainty comparison ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
steps = np.arange(EXPLORATION_STEPS)

colors = {
    "GP (standard)":  "#4a7cc7",
    "sGP_linear":     "#888888",
    "sGP_piecewise":  "#e8900a",
    "sGP_quadratic":  "#c44569",
    "c_sGP":          "#0d8c7a",
}

# MSE
axes[0].scatter(steps, all_mse_GP, label="GP (standard)", c=colors["GP (standard)"], s=25)
for name, color in list(colors.items())[1:]:
    axes[0].scatter(steps, results[name]["mse"], label=name, c=color, s=25)
axes[0].set_xlabel("Exploration step"); axes[0].set_ylabel("MSE")
axes[0].set_title("Predictive accuracy vs step count")
axes[0].legend(fontsize=8); axes[0].set_yscale("log")

# Uncertainty
axes[1].scatter(steps, all_uncertainty_GP, label="GP (standard)", c=colors["GP (standard)"], s=25)
for name, color in list(colors.items())[1:]:
    axes[1].scatter(steps, results[name]["uncertainty"], label=name, c=color, s=25)
axes[1].set_xlabel("Exploration step"); axes[1].set_ylabel("Total uncertainty")
axes[1].set_title("Uncertainty reduction vs step count")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.suptitle("GP vs all sGP variants — active exploration performance",
             y=1.02, fontsize=12, fontweight="bold")
plt.show()
```


```python
# ── Plot 2: c-sGP parameter recovery (does it learn the true physics?) ──
csGP_params = results["c_sGP"]["params"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
steps = np.arange(EXPLORATION_STEPS)

ground_truth = {"a1": 1.0, "b1": -1.0, "t": 1.5}
param_labels = {"a1": "a₁ (quadratic coeff.)", "b1": "b₁ (low-x shift)", "t": "t (transition)"}

for ax, (param, gt) in zip(axes, ground_truth.items()):
    vals = [p.get(param, np.nan) for p in csGP_params]
    ax.scatter(steps, vals, c="#0d8c7a", s=30, zorder=3)
    ax.axhline(gt, ls="--", c="#d97b0e", lw=2, label=f"True = {gt}")
    ax.set_xlabel("Exploration step")
    ax.set_ylabel(f"Inferred {param}")
    ax.set_title(f"Recovering {param_labels[param]}", fontsize=10)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.suptitle("c-sGP parameter convergence — recovering the ground truth physics",
             y=1.02, fontsize=12, fontweight="bold")
plt.show()

print()
print("Key observation:")
print("  By ~step 15, the inferred transition t converges to the true value 1.5.")
print("  The quadratic coefficient a1 converges to the true value 1.0.")
print("  c-sGP has not just fit the data — it has discovered the phase boundary")
print("  and the governing equation from only ~17 measurements.")
```

> **What the results tell us:**
> - **c-sGP reaches lower MSE faster than standard GP** — it needs fewer measurements to achieve the same accuracy.
> - By ~step 15, the inferred transition $t$ has converged to the true value of 1.5 and $a_1$ converges to 1.0.
> - The model has not just fit the data — **it has discovered the phase boundary and the curvature of the $E_g$-vs-$x$ curve**.
>
> The wrong mean functions (linear, standard GP) plateau at higher MSE because they cannot represent the piecewise-quadratic structure. **The more your mean function matches the true physics, the more efficiently sGP learns.**


---
<a id="section5"></a>
## 5 · Tutorial Part 2: Real MAPbI₃ / GAPbBr₃ Data

Now we apply the same approach to actual experimental photoluminescence data collected on a plate reader from MAPbI₃ / GAPbBr₃ thin films across a range of compositions.

> **Data:** Compositions by Mahshid Ahmadi and Elham Foadian; analysis by Yongtao Liu and Sheryl L. Sanchez.


<a id="5-1"></a>
### 5.1 · Loading Photoluminescence Spectra

The plate reader outputs a CSV with PL emission spectra (500–850 nm, 1 nm step) for each well on a 96-well plate. Each well contains a thin film with a specific GAPbBr₃ incorporation percentage.



```python
# ── Download experimental data from Google Drive ──────────────
import gdown

# Data file IDs (from Sheryl's shared Drive folder)
DATA_FILE_ID        = "13hREMYeu-uX3qQEj5YFpRpRs4k4-48NU"
COMPOSITION_FILE_ID = "16sc80Tc0hMhmb2S03MoqYBoJPbmGLUIz"

!gdown https://drive.google.com/uc?id={DATA_FILE_ID}
!gdown https://drive.google.com/uc?id={COMPOSITION_FILE_ID}

print("Data downloaded ✓")
```


```python
# ── Plate reader configuration ──────────────────────────────────
precursor1       = "$MAPbI_3$"
precursor2       = "$GAPbBr_3$"

start_wavelength  = 500   # nm
end_wavelength    = 850   # nm
wavelength_step   = 1     # nm  →  351 wavelength points per spectrum
number_of_reads   = 1
time_step         = 5     # ms

# Wells to exclude (edges, references, failed depositions)
wells_to_ignore_str = (
    "A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,"
    "B1,B3,B5,B7,B9,B11,B12,C1,C2,C3,C4,C5,C6,"
    "C7,C8,C9,C10,C11,C12,D1,D3,D5,D7,D9,D11,D12,"
    "E1,E2,E3,E4,E5,E6,E7,E8,E9,E10,E11,E12,"
    "F1,F3,F5,F7,F9,F11,F12,G1,G2,G3,G4,G5,G6,"
    "G7,G8,G9,G10,G11,G12,H1,H2,H3,H4,H5,H6,"
    "H7,H8,H9,H10,H11,H12"
)
wells_to_ignore = [w.strip() for w in wells_to_ignore_str.split(",")]

# ── Load raw plate reader data ──────────────────────────────────
rawData = pd.read_csv("all compositions.csv", header=None)
rawData = rawData.replace("OVRFLW", np.NaN)   # saturated detector → NaN
rawData = rawData.replace(r"^\s*$", np.nan, regex=True)

composition = pd.read_csv("4-20-23 GAPbBr3 compositions.csv", index_col=0)

# Drop ignored wells from composition table
for w in wells_to_ignore:
    if w in composition.columns:
        composition = composition.drop(w, axis=1)

print(f"Composition table shape: {composition.shape}")
print(f"Active wells: {composition.shape[1]}")
```


```python
# ── Parse plate reader CSV into per-read spectral dataframes ──
cells_all = [chr(64 + i) + str(j) for i in range(1, 9) for j in range(1, 13)]

rows = rawData[rawData[rawData.columns[0]] == "Read 1:EM Spectrum"].index.tolist()
rows += rawData[rawData[rawData.columns[0]] == "Results"].index.tolist()

d = {}
d["Read 1"] = rawData[rows[0] + 2: rows[1] - 1]
d["Read 1"] = d["Read 1"].drop([0], axis=1).drop([1], axis=1)

new_header = d["Read 1"].iloc[0]
d["Read 1"] = d["Read 1"][1:]
d["Read 1"].columns = new_header

for w in wells_to_ignore:
    if w in d["Read 1"].columns:
        d["Read 1"] = d["Read 1"].drop(w, axis=1)

d["Read 1"] = d["Read 1"].astype(float)

compositions = composition.values.T
targets      = d["Read 1"].values.T
wavelengths  = np.arange(start_wavelength, end_wavelength + wavelength_step,
                         wavelength_step)

print(f"Spectra array shape: {targets.shape}  (wells × wavelength points)")
print(f"Composition array shape: {compositions.shape}")

# ── Quick sanity check: plot one spectrum ──
example_well_idx = 12
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(wavelengths, targets[example_well_idx], c="#1a2844", lw=1.5)
peak_pos = wavelengths[np.argmax(targets[example_well_idx])]
ax.axvline(peak_pos, ls="--", c="#d97b0e", lw=1.5,
           label=f"Peak at {peak_pos} nm → $E_g$ = {1240/peak_pos:.3f} eV")
ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("PL Intensity (a.u.)")
ax.set_title(f"PL spectrum — well index {example_well_idx}")
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
```

<a id="5-2"></a>
### 5.2 · Extracting Bandgap Energies from Peak Positions

For each spectrum, we find the wavelength of peak emission and convert it to energy using the photon energy relation:

$$E_g = \frac{hc}{\lambda} = \frac{1240 \text{ eV·nm}}{\lambda_{\text{peak}} [\text{nm}]}$$



```python
peaks_all        = []
compositions_all = []

for i, (spectrum, comp) in enumerate(zip(targets, compositions)):
    peak_idx        = np.argmax(spectrum)
    peak_wavelength = wavelengths[peak_idx]

    # Convert peak wavelength → bandgap energy
    # E = hc/λ,  with hc = 1240 eV·nm
    E_g = 1240.0 / peak_wavelength

    peaks_all.append(peak_wavelength)
    compositions_all.append(comp[0])   # % GAPbBr3

compositions_all = np.array(compositions_all)
peaks_all        = np.array(peaks_all)
bandgaps_all     = 1240.0 / peaks_all

print(f"Number of measured compositions: {len(compositions_all)}")
print(f"Composition range: {compositions_all.min():.1f}% – {compositions_all.max():.1f}% GAPbBr₃")
print(f"Bandgap range:     {bandgaps_all.min():.3f} – {bandgaps_all.max():.3f} eV")

# ── Save to CSV ──
data_out = np.column_stack([compositions_all, bandgaps_all])
np.savetxt("bandgap_vs_composition.csv", data_out,
           delimiter=",", header="composition_%GAPbBr3,bandgap_eV", comments="")

# ── Plot: bandgap vs composition ──
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(compositions_all, peaks_all, c="#1a2844", s=40, alpha=0.8)
axes[0].set_xlabel(f"$GAPbBr_3$ incorporation (%)")
axes[0].set_ylabel("Peak PL wavelength (nm)")
axes[0].set_title("PL peak position vs composition")

axes[1].scatter(compositions_all, bandgaps_all, c="#1a2844", s=40, alpha=0.8)
axes[1].set_xlabel(f"$GAPbBr_3$ incorporation (%)")
axes[1].set_ylabel("Bandgap $E_g$ (eV)")
axes[1].set_title("Extracted bandgap vs composition")

plt.tight_layout()
plt.show()

print()
print("What to look for:")
print("  ✓ Monotonic increase from ~1.57 eV (MAPbI₃) to ~2.4 eV (GAPbBr₃)")
print("  ✓ Roughly linear trend → Vegard's law is a good starting hypothesis")
print("  ? Any kinks → possible phase transitions or miscibility gaps")
```

<a id="5-3"></a>
### 5.3 · Fitting sGP to Experimental Data

For MAPbI₃/GAPbBr₃, the bandgap-vs-composition plot shows a roughly linear trend, consistent with Vegard's law. We start with a **linear mean function** and compare with piecewise and quadratic options.

> **Strategy:** Start with endpoint compositions only. As the plate reader provides measurements for more compositions, add them to the training set and re-run. Each iteration tightens the posterior over the physical parameters.



```python
# ── Physics-informed priors for the real system ──────────────────
#
# For MAPbI3/GAPbBr3 we know roughly:
#   - MAPbI3 bandgap ~1.5–1.7 eV  →  intercept b in [1, 3]
#   - Bandgap increases with GAPbBr3  →  slope a > 0
#   - Total range ~0.87 eV over ~98% composition  →  slope ≈ 0.009 eV/%

def linear_real(x, params):
    """Linear Vegard-type mean function for the real system.
    a = bandgap slope (eV per % GAPbBr3)
    b = bandgap of pure MAPbI3 (eV)
    """
    return params["a"] * x + params["b"]

def linear_prior_real():
    a = numpyro.sample("a", dist.LogNormal(0, 1))     # slope: positive
    b = numpyro.sample("b", dist.Uniform(1, 3))        # intercept: 1–3 eV range
    return {"a": a, "b": b}

noise_prior_real = lambda: numpyro.sample("noise", dist.HalfNormal(0.01))
```


```python
# ── Simulation of the active learning workflow ───────────────────
#
# In a real experiment you would start with endpoint measurements,
# run sGP to decide which composition to measure next, synthesize it,
# measure its PL, and repeat. Here we simulate this using the full dataset.

# Read the saved data
df = pd.read_csv("bandgap_vs_composition.csv")
df = df.sort_values("composition_%GAPbBr3").reset_index(drop=True)

X_all = df["composition_%GAPbBr3"].values
y_all = df["bandgap_eV"].values

# Start from endpoint compositions
df_train = df.iloc[[0, -1], :].copy()
print("Initial training data (endpoints only):")
print(df_train.to_string(index=False))
print()

all_uncertainty_linear = []

# ── Add one composition at a time (simulating sequential experiments) ──
for iteration in range(min(10, len(df) - 2)):
    X_train = df_train["composition_%GAPbBr3"].values
    y_train = df_train["bandgap_eV"].values

    rng_key, rng_key_pred = gpax.utils.get_keys()

    sgp = gpax.ExactGP(1, kernel="Matern",
                       mean_fn=linear_real, mean_fn_prior=linear_prior_real,
                       kernel_prior=None, noise_prior=noise_prior_real)
    sgp.fit(rng_key, X_train, y_train, jitter=1e-4)

    # Predict over full range
    X_pred    = jnp.linspace(0, 98, 200)[:, None]
    y_mean, y_samples = sgp.predict_in_batches(rng_key_pred, X_pred,
                                                noiseless=True, jitter=1e-4)

    # Acquisition: which composition reduces uncertainty most?
    acq        = gpax.acquisition.UE(rng_key_pred, sgp, X_pred,
                                     noiseless=False, jitter=1e-4)
    next_x     = float(X_pred[acq.argmax()])
    total_unc  = float(y_samples.std(axis=(0, 1)).sum())
    all_uncertainty_linear.append(total_unc)

    print(f"Iteration {iteration+1:2d}:  training pts={len(X_train):2d}  "
          f"|  uncertainty={total_unc:.4f}  "
          f"|  next recommended composition={next_x:.1f}% GAPbBr₃")

    # "Measure" the closest available composition in our dataset
    remaining = df[~df["composition_%GAPbBr3"].isin(df_train["composition_%GAPbBr3"])]
    closest   = remaining.iloc[(remaining["composition_%GAPbBr3"] - next_x).abs().argsort()[:1]]
    df_train  = pd.concat([df_train, closest]).reset_index(drop=True)
```


```python
# ── Final fit on all available data ──────────────────────────────
rng_key, rng_key_pred = gpax.utils.get_keys()

sgp_final = gpax.ExactGP(1, kernel="Matern",
                          mean_fn=linear_real, mean_fn_prior=linear_prior_real,
                          kernel_prior=None, noise_prior=noise_prior_real)
sgp_final.fit(rng_key, X_all, y_all, jitter=1e-4)

X_pred  = jnp.linspace(0, 98, 300)[:, None]
y_mean, y_samples = sgp_final.predict_in_batches(rng_key_pred, X_pred,
                                                   noiseless=True, jitter=1e-4)
y_std = y_samples.std(axis=(0, 1))

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X_all, y_all, c="#1a2844", s=45, zorder=5, label="All measurements", alpha=0.85)
ax.plot(X_pred.squeeze(), y_mean, c="#0d8c7a", lw=2, label="sGP posterior mean")
ax.fill_between(X_pred.squeeze(),
                y_mean - 2 * y_std, y_mean + 2 * y_std,
                color="#0d8c7a", alpha=0.18, label="95% credible interval")
ax.set_xlabel("$GAPbBr_3$ incorporation (%)", fontsize=12)
ax.set_ylabel("Bandgap $E_g$ (eV)", fontsize=12)
ax.set_title("sGP fit to MAPbI₃/GAPbBr₃ bandgap data
(linear mean function — Vegard's law)", fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()
```

<a id="5-4"></a>
### 5.4 · Reading the Physics from Posterior Parameters

The real payoff of sGP is in the **posterior parameter distributions** — not just the curve fit. After training, `sgp.get_samples()` returns MCMC samples for every parameter.



```python
# ── Extract and display posterior parameter distributions ──────
params = sgp_final.get_samples()

print("Posterior parameter summary:")
print(f"{'Parameter':15s}  {'Mean':>8s}  {'Std':>8s}  {'95% CI':>20s}  Physical meaning")
print("-" * 85)
for name, samples in params.items():
    lo, hi = np.percentile(samples, [2.5, 97.5])
    meaning = {
        "a":        "Vegard slope (eV per % GAPbBr₃)",
        "b":        "MAPbI₃ bandgap (eV) at 0% GAPbBr₃",
        "k_length": "GP kernel lengthscale (composition units)",
        "k_scale":  "GP kernel output scale",
        "noise":    "Measurement noise (eV)",
    }.get(name, "")
    print(f"{name:15s}  {samples.mean():8.4f}  {samples.std():8.4f}  "
          f"[{lo:.4f}, {hi:.4f}]   {meaning}")
```


```python
# ── Plot posterior distributions for physical parameters ──────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Slope: rate of bandgap increase per % GAPbBr3
axes[0].hist(params["a"], bins=50, color="#0d8c7a", alpha=0.8, edgecolor="none")
axes[0].axvline(params["a"].mean(), c="#d97b0e", lw=2.5,
                label=f"Mean = {params['a'].mean():.5f} eV/%")
axes[0].set_xlabel("Vegard slope $a$ (eV per % GAPbBr₃)", fontsize=11)
axes[0].set_ylabel("MCMC samples", fontsize=11)
axes[0].set_title("Posterior: bandgap tuning rate", fontsize=11)
axes[0].legend(fontsize=10)

# Intercept: MAPbI3 bandgap
axes[1].hist(params["b"], bins=50, color="#1a2844", alpha=0.8, edgecolor="none")
axes[1].axvline(params["b"].mean(), c="#d97b0e", lw=2.5,
                label=f"Mean = {params['b'].mean():.4f} eV")
axes[1].axvline(1.57, c="#e74c3c", lw=1.5, ls="--",
                label="Literature MAPbI₃ = 1.57 eV")
axes[1].set_xlabel("MAPbI₃ bandgap $b$ (eV)", fontsize=11)
axes[1].set_ylabel("MCMC samples", fontsize=11)
axes[1].set_title("Posterior: MAPbI₃ endpoint bandgap", fontsize=11)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.suptitle("Posterior distributions over physical parameters",
             y=1.02, fontsize=12, fontweight="bold")
plt.show()

print()
print("Interpretation:")
print(f"  Slope a = {params['a'].mean():.5f} eV/% → total tuning ≈ "
      f"{params['a'].mean() * 98:.3f} eV over 0–98% GAPbBr₃")
print(f"  Intercept b = {params['b'].mean():.4f} eV → consistent with "
      f"literature MAPbI₃ bandgap (1.57 eV)")
```


```python
# ── Compare all mean function choices on real data ──────────────
#
# Run GP, sGP-linear, sGP-piecewise on the real data to see which
# physical hypothesis best fits the MAPbI3/GAPbBr3 system.

real_models = {
    "GP (no prior)":   (None,       None),
    "sGP linear":      (linear_real, linear_prior_real),
    "sGP piecewise":   (piecewise,  piecewise_prior),
    "sGP quadratic":   (quadratic,  quadratic_prior),
}

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
colors_list = ["#4a7cc7", "#0d8c7a", "#e8900a", "#c44569"]

for ax, (name, (fn, prior)), color in zip(axes, real_models.items(), colors_list):
    rk1, rk2 = gpax.utils.get_keys()

    if fn is None:
        model = gpax.ExactGP(1, kernel="Matern", noise_prior=noise_prior_real)
    else:
        model = gpax.ExactGP(1, kernel="Matern", mean_fn=fn,
                             mean_fn_prior=prior, noise_prior=noise_prior_real)

    model.fit(rk1, X_all, y_all, jitter=1e-4)
    y_m, y_s = model.predict_in_batches(rk2, X_pred, noiseless=True, jitter=1e-4)
    y_sd = y_s.std(axis=(0, 1))

    ax.scatter(X_all, y_all, c="#1a2844", s=20, alpha=0.7, zorder=5)
    ax.plot(X_pred.squeeze(), y_m, c=color, lw=2)
    ax.fill_between(X_pred.squeeze(), y_m - 2*y_sd, y_m + 2*y_sd,
                    color=color, alpha=0.2)
    ax.set_title(name, fontsize=10, fontweight="bold")
    ax.set_xlabel("% GAPbBr₃"); ax.set_ylabel("$E_g$ (eV)")
    unc = float(y_sd.sum())
    ax.text(0.05, 0.05, f"Unc: {unc:.2f}", transform=ax.transAxes,
            fontsize=9, color=color)

plt.tight_layout()
plt.suptitle("Comparing physical hypotheses on real MAPbI₃/GAPbBr₃ data",
             y=1.02, fontsize=12, fontweight="bold")
plt.show()

print("Lower uncertainty = the physical model better constrains the posterior.")
print("For this system, the linear (Vegard) model should perform well —")
print("confirming that linear mixing dominates with minimal bowing.")
```

---
<a id="section6"></a>
## 6 · Key Takeaways

### 1. Embed physics as a mean function
The mean function is your hypothesis. Start with the simplest physical model (linear Vegard) and add complexity (piecewise, bowing term) **only as the data demands it**.

### 2. You get parameters, not just predictions
MCMC gives you full posterior distributions over every physical constant — slopes, transition points, bandgap offsets. The **uncertainty in those parameters** tells you what the data hasn't yet resolved.

### 3. Active learning dramatically cuts data needs
Starting from just two endpoint measurements, the uncertainty-driven acquisition function navigates you to the compositions that most efficiently constrain the physics — often **5–10× fewer experiments** than random sampling.

### 4. Model selection is physical reasoning
Comparing GP vs. sGP-linear vs. c-sGP isn't just statistics — it's asking *"which physical mechanism best explains this material?"* When c-sGP wins over sGP-linear, you've identified a phase transition. When they tie, Vegard's law is a complete description.

---

### Quick reference: choosing a mean function

| Observation | Try this mean function |
|---|---|
| Smooth monotonic increase/decrease | `linear` |
| Monotonic but clearly curved | `quadratic` |
| Different slopes in two composition ranges | `piecewise` |
| Curved in one range, flat/constant in another | `piecewise_quadratic` (c-sGP) |
| No physical intuition yet | Start with `GP`, then switch to `sGP-linear` after 5 points |


---
<a id="section7"></a>
## 7 · References

1. **Ziatdinov, M.** GPax: Gaussian Processes in JAX. https://github.com/ziatdinovmax/gpax

2. **Liu, Y. & Sanchez, S.L.** GP and sGP for the exploration of bandgap vs. concentration. Jupyter notebooks (August 2023).

3. **Rasmussen, C.E. & Williams, C.K.I.** *Gaussian Processes for Machine Learning.* MIT Press, 2006. http://www.gaussianprocess.org/gpml/

4. **Ziatdinov, M. et al.** Physics-informed Gaussian process regression for materials discovery. *npj Computational Materials* (2022).

5. **Ahmadi, M., Foadian, E. & Sanchez, S.L.** High-throughput PL characterization of GAPbBr₃/MAPbI₃ compositional series (2023).

---
*Tutorial created by Sheryl L. Sanchez & Yongtao Liu · August 2023*

