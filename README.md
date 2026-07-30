# Master Thesis: Atomistic Simulations & Machine Learning Analysis Suite
**Author:** Lorenzo Atanasio (`Lorenzo-Atanasio-2000-hub`)  
**Academic Context:** Master's Thesis Repository in Computational Chemistry / Materials Science

---

## Overview
This repository contains a high-performance Python analysis suite designed to parse, evaluate, and visualize dataset structures and training logs from ab initio and classical atomistic molecular dynamics (MD) simulations. The workflow is specifically optimized for deep learning interatomic potentials frameworks like **DeepMD-kit**, as well as standard atomistic engines such as **CP2K**, **LAMMPS**, and **TraVis**.

The codebase is engineered with strict production standards, prioritizing memory efficiency, complete vectorization via NumPy, and advanced scientific plotting (Matplotlib, Seaborn, Scipy) to handle extensive simulation trajectories and high-throughput training files.

---

## 📂 Repository Structure

```text
master-thesis/
│
├── .gitignore                  # Prevents tracking heavy trajectory/log binary data (*.xyz, *.npy)
├── README.md                   # Main documentation portal (this file)
│
├── Data_Analysis/                     # CORE ANALYSIS SCRIPT SUITE
│   ├── adf_combiner.py                 # Merges and styles Angular Distribution Functions
│   ├── angles_distribution.py          # Vectorized, flexible Bond Angle Distribution (BADF) calculator
│   ├── cp2k_density_calculator_new.py  # Extracts equilibrium density out of CP2K cell-optimizations
│   ├── plot_lcurve_all_components.py   # Full 5-panel stacked training metric visualizer
│   ├── deepmd_train_err_plot.py        # 3-panel stacked energy/force/virial RMSE visualizer
│   ├── plot_learning_global.py         # Dual-axis (twinx) global learning curve monitor
│   ├── pca_on_DeepMD_dataset.py        # Dimensionality reduction (PCA) on configuration spaces
│   ├── lammps_density_average.py       # Computes production thermo averages discarding burn-in phases
│   ├── traj_merger.py                  # Structure-preserving multi-frame XYZ trajectory concatenator
│   ├── distance_analizer.py            # High-speed vectorized first coordination shell analyzer
│   ├── rdf_combiner.py                 # Semicolon-separated TraVis RDF combiner (pm to Å conversion)
│   └── energy_distribution.py          # Multi-system energy distribution normalized per atom (eV/atom)
│
└── simulation_templates/               # SIMULATION INPUT TEMPLATES
    ├── cp2k/
    |   ├── METAdyn.in                  # Perform metadynamic simulations
    |   ├── NPT_MD.in                   # Perform NPT simulations
    |   ├── NVT_MD.in                   # Perform NVT simulations
    |   ├── CELL_OPT.in                 # Perform cell optimization
    |   └── plumed.dat                  # Metadynamic variables
    |
    ├── lammps/
    |   ├── NPT_input.lammps            # Perform NPT simulations
    |   ├── NVT_input.lammps            # Perform NVT simulations
    |   └── minimize_input.lammps       # Perform an Energy minimization
    |
    └── deepmd/input.json
        └── input.json                  # Deep-MD model training input

```

---

##  Detailed Script Catalog & Capabilities

### 1. Neural Network Potential Evaluation (DeepMD-kit)
*   **`plot_learning_global.py`**: Monitors global model loss convergence by tracking validation trends against learning rate decay curves using a clean dual-axes layout.
*   **`deepmd_train_err_plot.py`** & **`plot_lcurve_all_components.py`**: Multi-panel stacked vertical grid layouts mapping Root Mean Square Error (RMSE) fields for Energy, Force, and Virial metrics. Uses shared X-axis parameters to guarantee publication-quality presentation without label overlaps.

### 2. Structural & Configuration Space Analysis
*   **`angles_distribution.py`**: A vectorized tool mapping bond angle distribution functions (BADF) around a chosen coordination center. It automatically generates all triplet permutations (e.g., S-P-S, S-P-O) from a neighbors list using NumPy broadcasting.
*   **`distance_analizer.py`**: Maps closest interatomic contacts (e.g., Li–O bonds) across files using fast multi-dimensional matrix operations, bypassing sluggish nested loops. Includes Scipy 1D Gaussian smoothing filters.
*   **`pca_on_DeepMD_dataset.py`**: Flattens high-dimensional `coord.npy` arrays and maps the sampling variety of training datasets into a 2D PC1/PC2 configuration landscape. Automatically accounts for supercell replica expansions.
*   **`rdf_combiner.py`** & **`adf_combiner.py`**: Combines Radial and Angular Distribution tracks (e.g., from TraVis). Handles automatic text stripping and rescales pm distances to Ångströms.

### 3. Thermodynamic Parsing & Utilities
*   **`energy_distribution.py`**: Extracts total potential energies from `energy.npy` matrices and filters them against atom counts (`type.raw`) to plot unified, normalized energy densities (**eV/atom**), resolving scale skews across supercells.
*   **`lammps_density_average.py`**: Dynamically mines structural columns inside LAMMPS logs. Includes configurable discard factors to remove equilibration phases (burn-in) before extracting standard deviations.
*   **`cp2k_density_calculator_new.py`**: Text-mining parser tracking volume traces in CP2K cell optimization loops to instantly resolve density trends.
*   **`traj_merger.py`**: Concatenates partitioned sequential multi-frame trajectory files into a single matrix stream while maintaining the precise headers required by VMD and Ovito.

---

##  Getting Started

### Prerequisites
Ensure your local Python instance is equipped with standard scientific libraries:
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

### Advanced Usage Examples
All standalone utilities are wrapped in command-line argument interfaces (CLI) for ease of integration.

*   **Plot complete stacked DeepMD log metrics:**
    ```bash
    python analysis_tools/plot_lcurbe_all_components.py path/to/lpath.log --smooth 5 --log
    ```
*   **Perform automatic bond angle distributions analysis around Phosphorus (P):**
    ```bash
    python analysis_tools/angles_distribution.py --dataset ./dataset --center P --neighbors S O --cutoff 2.6 --sigma 1.5
    ```
*   **Extract minimum distance distributions for Li–O contacts:**
    ```bash
    python analysis_tools/distance_analizer.py --dataset ./dataset --ref Li --targets O --sigma 2.0
    ```
*   **Generate an atomic-normalized energy density overview map:**
    ```bash
    python analysis_tools/energy_distribution.py --dataset ./dataset --output ./plots
    ```

---

##  Research Integrity & Confidentiality Notice
To respect academic intellectual property rights, non-disclosure agreements, and unpublished experimental models belonging to the university research laboratory, **no raw production trajectory files or proprietary coordinates are tracked within this public registry**. 

Large native outputs are filtered locally using strict operational criteria outlined in the `.gitignore` asset file.
