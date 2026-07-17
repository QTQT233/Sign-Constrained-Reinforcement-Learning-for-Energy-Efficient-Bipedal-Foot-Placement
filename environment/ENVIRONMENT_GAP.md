# Training/evaluation environment lock still required

The exact Python, PyTorch, CUDA, SciPy, NumPy, and platform versions used for
the original training runs were not retained in the supplied result folders.
An executable local Conda environment was discovered during the audit and its
principal versions are recorded in `local-audit-environment.txt`; this is a
valid rerun candidate, but it is not proof that every historical checkpoint was
trained with exactly those versions.

Before public release, export the original environment from the machine or
environment that can load and rerun the checkpoints, for example as both a
`conda-lock.yml` (or `environment.yml`) and `pip-freeze.txt`. Record:

- Python, PyTorch, CUDA toolkit, cuDNN, NumPy, and SciPy versions;
- CPU/GPU model and operating system;
- deterministic-algorithm settings;
- the command used for every training and evaluation entry point.

Do not invent or back-fill versions from the current analysis environment.
