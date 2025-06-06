# EMres
Diffusion-based image restoration pipeline for degraded early music manuscripts, including bleedthrough removal, staff line recovery, and note repair with contextual inpainting

## Repository Status

This repository contains the final code for the two diffusion models that make up the complete restoration pipeline described in the thesis. These include:

- **Combined diffusion model** for ink bleedthrough removal and staff line restoration  
- **Note restoration model** for recovering missing or degraded musical symbols

Additional experimental configurations discussed in the thesis are based on similar code structures but are not included here.  
If you are interested in these variants, feel free to reach out.


## Attribution
This repository builds upon and adapts parts of the [IR-SDE](https://github.com/Algolzw/image-restoration-sde) codebase (Original license: MIT).

Key components derived from IRSDE include: 
- The baseline Conditional U-Net architecture, which we extended and modified for our purposes.
- Training and testing routines, as well as various utility functions, were adapted with minor modifications.

All rights belong to the original authors.

