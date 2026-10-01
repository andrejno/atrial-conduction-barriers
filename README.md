# Reconstructing atrial conduction barriers from sparse observations using an anisotropic phase-field model

Andrej Novak

[Website preview](docs/index.html) · [Executed notebook](notebooks/Atrial_Reconstruction.ipynb) · [Data attribution](code/resources/DATA_ATTRIBUTION.md)

This study reconstructs continuous atrial barrier fields from incomplete observations and tests the propagation they support. Synthetic experiments and six clinical left-atrial maps show how missing measurements affect reconstruction, and why geometric closure or passive conductance alone can leave electrical behaviour unresolved.

[![Clinical atrial substrate](docs/images/patient_reconstruction_rotation.webp)](docs/index.html)

## Run

```bash
python -m pip install -r requirements.txt
python code/reproduce.py --workers 6
```

The input data are included. Computations run in `.research/`; the terminal reports the output directory. Add `--fresh` for a new run or `--media` to regenerate animations with FFmpeg. `--prepare-only` verifies and extracts the inputs without running simulations.

```bash
jupyter lab notebooks/Atrial_Reconstruction.ipynb
```

The notebook contains manuscript tables, figures and four videos. Timing measurements depend on the machine.

## Website

Open `docs/index.html` locally. Upload the repository contents to GitHub in batches, or push them with Git. For GitHub Pages, select **Settings → Pages → Deploy from a branch → main → /docs**.

Set the published website link after choosing the repository address, then include the changed files in the upload:

```bash
python code/set_github_url.py USERNAME/REPOSITORY
```

After a run with `--media`, `python code/build_clinical_interactive.py` and `python code/build_ring_interactive.py` rebuild the interactive viewers from the saved results.

EP Team: Andrej Novak, Ivan Zeljkovic, Ante Lisicic, Ana Jordan, Nikola Pavlovic, Sime Manola
