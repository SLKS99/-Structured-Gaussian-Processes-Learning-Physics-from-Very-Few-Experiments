# Structured Gaussian Processes (sGP) — Bandgap Perovskites Tutorial

This repository contains a hands-on tutorial notebook and a nicely formatted HTML version for reading.

## What to read / run

- **Notebook (run this)**: `notebooks/sGP_Tutorial_Bandgap_Perovskites.ipynb`
- **HTML (read this)**: `docs/sgp_tutorial.html`

## Quickstart (Windows, Python venv)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Then open `notebooks/sGP_Tutorial_Bandgap_Perovskites.ipynb`.

## Quickstart (Conda)

```powershell
conda env create -f environment.yml
conda activate sgp-tutorial
jupyter lab
```

## Export the notebook to HTML

```powershell
.\scripts\export_html.ps1
```

That will write an updated `docs/sgp_tutorial.html`.

## Repo layout

```text
notebooks/   Jupyter notebook(s) you execute
docs/        Rendered HTML for easy reading/sharing
scripts/     Helper scripts (launching, exporting)
assets/      Images/data (if you add any)
```

## Notes on dependencies

The notebook installs `gpax` from GitHub and uses JAX/NumPyro. If you run into JAX installation issues on your machine, install a compatible JAX build first, then re-run `pip install -r requirements.txt`.

