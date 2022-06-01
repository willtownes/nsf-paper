#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage:

python -m simulations.benchmark 2 simulations/bm_sp

Fit a model to a dataset and save the results.

Slightly different from utils.benchmark, which assumes there is only one possible
data file under the parent directory. Here, we allow the data directory to contain
multiple H5AD files and ensure the saved models indicate the scenario in the "key" (filepath).

Result:
  * Loads data, fits a model
  * pickles a fitted ModelTrainer object under [dataset]/models/ directory

Pickled model naming conventions:

file scheme for spatial models:
[dataset]/models/S[scenario]/V[val frac]/L[factors]/[likelihood]/[model]_[kernel]_M[inducing_pts]/epoch[epoch].pickle

file scheme for nonspatial models:
[dataset]/models/S[scenario]/V[val frac]/L[factors]/[model]/epoch[epoch].pickle

@author: townesf
"""
from os import path
from argparse import ArgumentParser
from utils.misc import read_csv_oneline
from utils import benchmark as ubm

def benchmark(ID,pth):
  """
  Run benchmarking on dataset for the model specified in benchmark.csv in row ID.
  """
  # dsplit = dataset.split("/data/")
  # pth = dsplit[0]
  csv_file = path.join(pth,"results/benchmark.csv")
  #header of CSV is row zero
  p = read_csv_oneline(csv_file,ID-1)
  opath = path.join(pth,"models",p["key"]) #p["key"] includes "S{}/" for simulations
  print("{}".format(p["key"]))
  if path.isfile(path.join(opath,"converged.pickle")):
    print("Benchmark already complete, exiting.")
    return None
  else:
    print("Starting benchmark.")
    train_frac = ubm.val2train_frac(p["V"])
    dataset = path.join(pth,"data/S{}.h5ad".format(p["scenario"])) #different in simulations
    D,fmeans = ubm.load_data(dataset,model=p['model'],lik=p['lik'],sz=p['sz'],
                             train_frac=train_frac, flip_yaxis=False)
    fit = ubm.init_model(D,p,opath,fmeans=fmeans)
    tro = ubm.fit_model(D,fit,p,opath)
    return tro

def arghandler(args=None):
    """parses a list of arguments (default is sys.argv[1:])"""
    parser = ArgumentParser()
    parser.add_argument("id", type=int,
                        help="line in benchmark csv from which to get parameters")
    parser.add_argument("path", type=str,
                        help="top level directory containing with subfolders 'data', 'models', and 'results'.")
    args = parser.parse_args(args) #if args is None, this will automatically parse sys.argv[1:]
    return args

if __name__=="__main__":
  # #input from argparse
  # ID = 2 #slurm job array uses 1-based indexing (2,3,...,43)
  # #start with 2 to avoid header row of CSV
  # DATASET = "simulations/bm_sp/data/S1.h5ad"
  args = arghandler()
  tro = benchmark(args.id, args.path)
