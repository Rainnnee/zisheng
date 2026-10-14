# Project proposal

## Working title

字生 (Zìshēng): From Noise to Chinese Characters

## Research question

Can a small class-conditional diffusion model generate recognizable images of selected Chinese characters, and how do corruption level and stroke complexity relate to recognizability?

## Planned scope

The baseline is a small Python/PyTorch image-generation study with a carefully documented character subset, a forward Gaussian corruption process, a conditioned denoiser, reverse sampling, and a fixed evaluation protocol. Printed-font rendering is the controlled prototype. Handwriting is a possible main domain if access, parsing, writer-level splits, coverage, and use conditions are verified promptly.

## Research questions

1. How often do generated samples depict the requested class, and which visual errors occur?
2. How does recognizability change with corruption level, and how well does denoising preserve character identity and image structure?
3. How do the measured outcomes vary across the selected characters' stroke counts?

## Current status

Planning and repository scaffolding only. Data acquisition, implementation, training, and evaluation have not started in this repository. All design choices remain provisional until documented and checked against the course contract and data sources.
