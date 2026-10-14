# HanziFlow

**From noise to characters: conditional diffusion, recognizability, and stroke complexity.**

HanziFlow is a STATS 100A Honors project studying whether a small class-conditional diffusion model can generate selected Chinese character images, and how corruption level and character stroke complexity relate to recognizability. The planned baseline uses Python and PyTorch; the project is currently at the planning and repository-scaffolding stage. No model has been trained and no results are available yet.

## Research questions

1. How often do generated images depict the requested character, and what errors appear?
2. As Gaussian corruption increases, how do recognizability and restoration change?
3. How do these outcomes vary across selected characters with different stroke counts?

The study will begin with a small, controlled font-rendering prototype. Handwriting may become the main data domain if a suitable dataset can be acquired, parsed, split by writer, and used under its terms. The initial target is a modest set of characters; the final list and data source will be recorded before training.

## Project status

- [x] Research plan and experiment protocol drafted in the planning chat.
- [x] Repository created and starter structure added.
- [ ] Confirm the final project requirements and data domain.
- [ ] Audit data sources, licenses, and character coverage.
- [ ] Implement and validate the forward diffusion process.
- [ ] Train a small conditional model and evaluate it with a fixed protocol.
- [ ] Report results, limitations, and reproducibility details.

The checked items describe planning and setup only. They do not indicate that data acquisition, implementation, or training is complete.

## Repository map

```text
configs/             Proposed experiment settings
data/                Data acquisition notes; source images stay out of Git
docs/                Research question, mathematics, data, model, and protocol notes
metadata/            Character manifest schema
notebooks/           Focused probability and implementation demonstrations
results/             Notes and, later, traceable experiment outputs
scripts/             Future data and experiment entry points
src/hanziflow/       Python package skeleton
```

The Python modules currently contain interface placeholders. They do not implement diffusion, training, sampling, or evaluation.

## Setup

The package metadata targets Python 3.11 or newer. At this stage it has no runtime dependencies because the model and data pipeline have not been implemented.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

PyTorch and platform-specific dependencies will be added after benchmarking the available Mac and CUDA environments. No training command is available yet.

## Data and responsible use

See [data/README.md](data/README.md) before acquiring any data. Keep raw datasets, restricted samples, credentials, local machine paths, and large checkpoints out of Git. Preserve source attribution and license notices. A font-rendered glyph is a synthetic sample, not an independent handwriting example.

The proposed data and evaluation choices are provisional until the sources, terms, labels, and splits have been audited. Do not interpret placeholder configuration values or planned analyses as findings.

## Reproducibility goals

Before reporting results, record the data manifest and split, configuration, random seeds, software and device details, checkpoint, raw metrics, and figure-generation steps. Keep failures and deviations in the research log. The experiment protocol and model card should describe limitations as well as successful examples.

## License

No project code license has been selected yet. Third-party fonts and datasets may have separate terms; review and preserve those terms before use or redistribution.
