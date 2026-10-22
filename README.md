# EstIP-Building-Toolbox

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23558227.svg)](https://doi.org/10.5281/zenodo.23558227)  
Developed at [IOER](https://www.ioer.de/en/)  
Copyright © Denis Reiter, Dresden, 2026  
All rights reserved.


## Overview
This set of tools is meant to enrich building data with various features, enabling additional processing steps down the line. It was built as part of the **EstIP population estimation pipeline** and serves as the foundation for the corresponding [dataset series](https://data.fdz.ioer.de/dataverse/root/?q=EstIP). Therefore, the structure and output is tailored towards EstIP, but it can be used in different contexts, too. The output of this toolbox is directly compatible with the actual population estimation pipeline, which can be found [here](https://github.com/denisreiter/EstIP-Estimation-Pipeline).

### Related publications:

1. Reiter, D., Jehling, M. and Hecht, R. (2023) ‘Benefits of using address-based dasymetric mapping in micro-level census disaggregation’, _AGILE: GIScience Series_, 4, pp. 1–7.Available at: https://doi.org/10.5194/agile-giss-4-38-2023.

2. Reiter, D., Jehling, M. and Hecht, R. (2026) ‘Urban form metrics for building occupancy modelling: applications and impacts’, _Buildings & Cities_, 7(1), pp. 1035–1051. Available at: https://doi.org/10.5334/bc.749.

3. Reiter, D., Jehling, M. and Hecht, R. (2026) ‘Intercensal building‑based population estimation’, _International Journal of Digital Earth_, 19(2), p. 2738267. Available at: https://doi.org/10.1080/17538947.2026.2738267.


## Features
- Built for 3D building data (LoD2)
- Modular pipeline consisting of 6 individual scripts
- Logical processing steps are bundled together
- Full output delivers 35 building features, plus some optional ones
- Reproducible configuration via files


## Requirements
- Windows 11 (currently the only supported OS)
- Python 3.11
- Working conda/miniforge installation


## Installation
#### For end-users:
```bash
pip install estip-building-toolbox
```
#### For development:
1. Clone the repository.
    ```bash
    git clone https://github.com/denisreiter/EstIP-Building-Toolbox.git
    ```
2. Open a conda/miniforge prompt.
3. Change to the directory where you cloned the repo into.
    ```bash
    cd <path/to/your/cloned/repo>
    ```
4. Create and activte a new environment:
   ```bash
   conda env create --file environment.yml
   conda activate estip-building-toolbox
   ```


## Configuration 
The toolbox consists of six indivdual processing scripts (s1 - s6). Each scripts variables are controlled via its corresponding .yaml config file, where paths and field names need to be set by the user. A minimal working collection of config files and datasets can be found under `examples/`. 

The config files can be saved for reproducability. Config flags given via command line overwrite flags given inside config files.

`run_all_cfg.yaml` controls which scripts are run during a full pipeline run. Use comments `#` to exclude scripts from execution in a full pipeline run, or remove the entries entirely.


## Assumptions & Limitations 
- All data should be projected to EPSG:3035.
- Input buildings ideally should be converted footprints from a 3D CityGML model inside a .gpkg file (see [an example](https://doi.org/10.71830/GVNGUA) here).
- All datasets need to be stored inside a single workspace `.gpkg` file.


## Usage 

### If you installed the package via `pip`:
1. Adapt paths and field names in the config files `cfg_s1.yaml` - `cfg_s6.yaml`.
2. Activate `estip-building-toolbox` (or any alternative env you installed the package in) in your conda prompt.
3. Use the example dataset provided under `examples/` for demonstration, or your own dataset.
4. Single script execution:
    ```bash
    run-s1 --config <path/to/your/config/cfg_s1.yaml>
    ```
5. Run entire pipeline:
    ```bash
    run-all --config <path/to/your/config/run_all_cfg.yaml>
    ```


### If you are using the development version:
1. Adapt paths and field names in the config files `cfg_s1.yaml` - `cfg_s6.yaml`.
2. Activate `estip-building-toolbox` (or any alternative env you installed the package in) in your conda prompt.
3. Use the example dataset provided under `examples/` for demonstration, or your own dataset.
4. Change to the directory where you cloned the repo into.
    ```bash
    cd <path/to/your/cloned/repo>
    ```
4. Single script execution: 
    ```bash
    python -m scripts.s1_preprocessing.cli --config <path/to/your/config/cfg_s1.yaml>
    ```
5. Run entire pipeline:
    ```bash
    python -m scripts.run_all --config <path/to/your/config/run_all_cfg.yaml>
    ```


## Pipeline Overview
#### s1_preprocessing
- Input: LoD2 buildings, LoD2 building parts
- Tasks: Cleaning, reprojecting, joining parts info, assign residential building types.
- Features: "CNT_PRTS", "RF_TYPE", "FNC_CODE"  
_Note: This step can be skipped if building data already is preprocessed._
#### s2_basic_geometric_features
- Input: Output from s1 or raw buildings
- Tasks: Calculate basic geometric features which are not dependent on context.
- Features: "SHPX_2D", "RATIO_C_A", "RATIO_LW_MRR", "RATIO_MRR_A", "CNT_NDS", "CNT_HLS", "A", "C", "H", "V", "RATIO_HLL_V"
#### s3_neighbourhood_features
- Input: Output from s1
- Tasks: Calculate features dependent on neighbouring buildings within a configurable radius. Fields are dynamically named based on radius. Default: 50m.
- Features: "DIST_NGHB", "RATIO_SW", "CNT_NGHB_Rxx", "DIST_NGHB_Rxx"
#### s4_context_features
- Input: Output from s3, Addreses (points), Blocks (polygons), Streets (polylines)
- Tasks: Calculates features and ratios based on contextual information.
- Features: "DIST_STRT", "CNT_ADD", "BLCK_TYPE", "CNT_BLCK", "A_SUM_BLCK", "A_MEAN_BLCK", "A_STD_BLCK", "V_SUM_BLCK", "V_MEAN_BLCK", "V_STD_BLCK", "MEAN_DIST_NGHB_BLCK", "RATIO_A_BLCK", "RATIO_A_SUM", "RATIO_V_SUM", "RATIO_A_MEAN", "RATIO_V_MEAN"  
_Note: Each ancillary dataset is optional, but at least one has to provided. Corresponding features will not be processed for those not not provided._
#### s5_spatial_assignment
- Input: Output from s4, muncipalities, grid
- Tasks: Adds municipality, reginonal type (optional) and grid cell ID to buildings.
- Features: "MUNICIP", "REG_TYPE", "MAPPED_CELLCODE"
#### s6_population_mapping
- Input: Output from s6, grid
- Tasks: Dasymetrically maps population from census raster to buildings by grid cell ID.
- Features: "MAPPED_POPULATION"  
_Note: This step can be skipped if no population data is available._


## Citation 
Reiter, Denis (2026): EstIP-Building-Toolbox (Version 0.1.0) [Software]. https://github.com/denisreiter/EstIP-Building-Toolbox
