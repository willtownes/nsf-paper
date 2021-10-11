# Nonnegative spatial factorization for multivariate count data

This repository contains supporting code to facilitate reproducible analysis. For details see the preprint. If you find bugs please create a github issue. 

### Authors

Will Townes and Barbara Engelhardt

## Description of Repository Contents

### models

TensorFlow implementations of probabilistic factor models
* *cf.py* - nonspatial models (factor analysis and probabilistic nonnegative matrix factorization).
* *mefisto.py* - wrapper around the MEFISTO implementation in the [mofapy2](https://github.com/bioFAM/mofapy2/commit/8f6ffcb5b18d22b3f44ff2a06bcb92f2806afed0) python package.
* *pf.py* - nonnegative and real-valued spatial process factorization (NSF and RSF).
* *pfh.py* - NSF hybrid model, includes both spatial and nonspatial components.

### scrna

Analysis of spatial transcriptomics data
* *sshippo* - Slide-seqV2 mouse hippocampus
* *visium_brain_sagittal* - Visium mouse brain (anterior sagittal section)
* *xyzeq_liver* - XYZeq mouse liver/tumor

### simulations

Data generation and model fitting for the ggblocks and quilt simulations

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