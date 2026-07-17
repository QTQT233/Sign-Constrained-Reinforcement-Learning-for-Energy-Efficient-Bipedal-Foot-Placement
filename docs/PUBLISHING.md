# Publishing the complete reproducibility package

Target repository:
`QTQT233/Sign-Constrained-Reinforcement-Learning-for-Energy-Efficient-Bipedal-Foot-Placement`.

The remote `main` branch currently contains the 37 historical root files but
not the structured `src/`, `data/`, `models/`, `analysis/`, `results/`, `docs/`,
`environment/`, `tests/`, and `tools/` trees. Do not force-push the old
`Paper_1` history over this repository. Publish through a branch and draft PR.

## One-time Windows setup

```powershell
winget install --id GitHub.cli
gh auth login
gh auth status
```

## Safe branch-based upload

```powershell
git clone https://github.com/QTQT233/Sign-Constrained-Reinforcement-Learning-for-Energy-Efficient-Bipedal-Foot-Placement.git
cd Sign-Constrained-Reinforcement-Learning-for-Energy-Efficient-Bipedal-Foot-Placement
git switch -c codex/reproducibility-package

robocopy "C:\Users\Admin\Documents\Review\analysis_ms_qi14\github_repo" . /E /XD .git __pycache__

python -m unittest discover -s tests -v
python tools/run_frozen_tvlqr.py --check-only
git status -sb
git add -A
git commit -m "Add complete reproducibility package"
git push -u origin codex/reproducibility-package

gh pr create --draft --base main --head codex/reproducibility-package `
  --title "Add complete reproducibility package" `
  --body "Adds frozen sources, checkpoints, trial-level outputs, deterministic analysis, TVLQR seed-0 reproduction, provenance manifests, tests, and release documentation."
```

Review the draft PR file list before merging. After merge, create an immutable
release tag, archive the same commit and large data bundle in a DOI repository,
and replace `[DOI TO BE MINTED]` in the data-availability materials only after
the public record exists.

## Release gate

- `python -m unittest discover -s tests -v` passes;
- `python tools/run_frozen_tvlqr.py --check-only` verifies 12 source files and
  four checkpoints;
- `MANIFEST.csv` and `CHECKSUMS.sha256` are regenerated after the last edit;
- repository and author-owned data are explicitly licensed under Apache-2.0;
- no bracketed DOI/tag placeholders remain in the submitted manuscript;
- the GitHub release and DOI record cross-reference the same Git commit.
