# Nonnegative spatial factorization for multivariate count data

This repository contains supporting code to facilitate reproducible analysis. For details see the [preprint](https://arxiv.org/abs/2110.06122). If you find bugs please create a github issue. An [installable python package](https://github.com/willtownes/spatial-factorization-py) 
is also under development.

### Authors

Will Townes and Barbara Engelhardt

### Abstract

Gaussian processes are widely used for the analysis of spatial data due to their nonparametric flexibility and ability to quantify uncertainty, and recently developed scalable approximations have facilitated application to massive datasets. For multivariate outcomes, linear models of coregionalization combine dimension reduction with spatial correlation. However, their real-valued latent factors and loadings are difficult to interpret because, unlike nonnegative models, they do not recover a parts-based representation. We present nonnegative spatial factorization (NSF), a spatially-aware probabilistic dimension reduction model that naturally encourages sparsity. We compare NSF to real-valued spatial factorizations such as MEFISTO and nonspatial dimension reduction methods using simulations and high-dimensional spatial transcriptomics data. NSF identifies generalizable spatial patterns of gene expression. Since not all patterns of gene expression are spatial, we also propose a hybrid extension of NSF that combines spatial and nonspatial components, enabling quantification of spatial importance for both observations and features.

## Description of Repository Contents
All scripts should be run from the top level directory.

### models

TensorFlow implementations of probabilistic factor models
* *cf.py* - nonspatial models (factor analysis and probabilistic nonnegative matrix factorization).
* *mefisto.py* - wrapper around the MEFISTO implementation in the [mofapy2](https://github.com/bioFAM/mofapy2/commit/8f6ffcb5b18d22b3f44ff2a06bcb92f2806afed0) python package.
* *sf.py* - nonnegative and real-valued spatial process factorization (NSF and RSF).
* *sfh.py* - NSF hybrid model, includes both spatial and nonspatial components.

### scrna

Analysis of spatial transcriptomics data
* *sshippo* - Slide-seqV2 mouse hippocampus
* *visium_brain_sagittal* - Visium mouse brain (anterior sagittal section)
* *xyzeq_liver* - XYZeq mouse liver/tumor

### simulations

Data generation and model fitting for the ggblocks and quilt simulations.
* *benchmark.py* - can be called as a command line script to facilitate benchmarking of large numbers of scenarios and parameter combinations.
* *benchmark_gof.py* - compute goodness of fit and other metrics on fitted models.
* *bm_mixed* - mixed spatial and nonspatial factors
* *bm_sp* - spatial factors only. Within this folder, the notebooks
`04_quilt_exploratory.ipy` and `05_ggblocks_exploratory.ipy` have many 
visualizations of the various models compared in the paper.
* *sim.py* - functions for creating the simulated datasets.

### utils

Python modules containing functions and classes needed by scripts and model implementation classes.
* *benchmark.py* - functions used in fitting models to datasets and pickling the objects for later evaluation. Can be called as a command line script to facilitate automation.
* *benchmark_gof.py* - script with basic command line interface for computing goodness-of-fit, sparsity, and timing statistics on large numbers of fitted model objects
* *misc.py* - miscellaneous convenience functions useful in preprocessing (normalization and reversing normalization), postprocessing, computing benchmarking statistics, parameter manipulation, and reading and writing pickle and CSV files.
* *nnfu.py* - nonnegative factor model utility functions for rescaling and regularization. Useful in initialization and postprocessing.
* *postprocess.py* - postprocessing functions to facilitate interpretation of nonnegative factor models.
* *preprocess.py* - data loading and preprocessing functions. Normalization of count data, rescaling spatial coordinates for numerical stability, deviance functions for feature selection (analogous to [scry](https://doi.org/doi:10.18129/B9.bioc.scry)), conversions between AnnData and TensorFlow objects. 
* *training.py* - classes for fitting TensorFlow models to data, including caching with checkpoints, automatic handling of numeric instabilities, and ConvergenceChecker, which uses a cubic spline to detect convergence of a stochastic optimizer trace.
* *visualize.py* - plotting functions for making heatmaps to visualize spatial and nonspatial factors, as well as some goodness-of-fit metrics.

## System requirements

We used the following versions in our analyses: Python 3.8.10, tensorflow 2.5.0, tensorflow probability 0.13.0, scanpy 1.8.0, squidpy 1.1.0, scikit-learn 0.24.2, pandas 1.2.5, numpy 1.19.5, scipy 1.7.0. 
We used the MEFISTO implementation from the mofapy2 Python package, installed from the GitHub development branch at commit 8f6ffcb5b18d22b3f44ff2a06bcb92f2806afed0.

```
pip install git+git://github.com/bioFAM/mofapy2.git@8f6ffcb5b18d22b3f44ff2a06bcb92f2806afed0
```

Graphics were generated using either matplotlib 3.4.2 in Python or ggplot2 3.3.5 in R (version 4.1.0). The R packages Seurat 0.4.3, SeuratData 0.2.1, and SeuratDisk 0.0.0.9019 were used for some initial data manipulations.

Computationally-intensive model fitting was done on Princeton's [Della cluster](https://researchcomputing.princeton.edu/systems/della). Analyses that were less computationally intensive were done on personal computers with operating system MacOS version 12.4.
