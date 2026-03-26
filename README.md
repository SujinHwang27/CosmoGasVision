# IGM Tomography

3D Gas Density Reconstruction from Lyman-alpha Spectra using Implicit Neural Representations

## Dependencies

This project uses uv for dependency management.

### Install Dependencies

```bash
uv sync
```

### Available Packages

- numpy: Numerical computing
- scipy: Scientific computing (signal processing, interpolation)
- matplotlib: Visualization
- h5py: HDF5 file handling for simulation data

## Generate Spectra

```bash
cd Sherwood/src
python get_spectra.py <physics> <redshift> [--nspec N] [--SN S] [--seed R]
```

Parameters:
- `physics`: Physics model (1-4): 1=nofeedback, 2=stellarwind, 3=windAGN, 4=windstrongAGN
- `redshift`: Redshift value (0.1, 0.3, 2.2, or 2.4)
- `--nspec`: Number of spectra to generate (default: 1000)
- `--SN`: Signal-to-noise ratio (optional)
- `--seed`: Random seed for reproducibility (optional)

Output files: `wave.npy`, `flux.npy`, `axis.npy`, `tau.npy`, `vel.npy`

## Read Halo Data

```bash
cd SherwoodIGM_gal
python readhalo.py
```

Converts binary halo catalog (`halolist_012.dat`) to CSV format.

## Project Structure

```
ComputerVisionProject/
├── Sherwood/                    # Line-of-sight processed data
│   ├── Physics1_nofeedback/    # Physics model 1 (no feedback)
│   ├── Physics2_stellarwind/   # Physics model 2 (stellar winds)
│   ├── Physics3_windAGN/       # Physics model 3 (wind + AGN)
│   ├── Physics4_windstrongAGN/  # Physics model 4 (strong AGN)
│   └── src/                    # Spectrum generation code
├── SherwoodIGM_gal/            # Raw HDF5 simulation snapshots
├── documents/                  # Project documentation
├── pyproject.toml              # Project configuration (uv)
└── README.md                   # This file
```
