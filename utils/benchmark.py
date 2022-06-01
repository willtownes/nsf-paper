#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage:

python -m utils.benchmark 2 scrna/visium_brain_sagittal/data/visium_brain_sagittal_J2000.h5ad

Fit a model to a dataset and save the results.

Inputs:
  * integer row ID (from slurm job array), pulls out one set of params from csv
  * parent directory of the dataset (eg scrna/visium_brain_sagittal)
  * (relative) file path to H5AD file with anndata dataset object
  with following:
    - rows (observations) randomly shuffled
    - cols (features) in decreasing order of importance (eg deviance)
    - spatial coordinates in .obsm['spatial']
  * (optional) relative file path to a JSON configuration file with
  additional parameters specific to this dataset.

Result:
  * Loads data, fits a model
  * pickles a fitted ModelTrainer object under [dataset]/models/ directory

Pickled model naming conventions:

file scheme for spatial models:
[dataset]/models/V[val frac]/L[factors]/[likelihood]/[model]_[kernel]_M[inducing_pts]/epoch[epoch].pickle

file scheme for nonspatial models:
[dataset]/models/V[val frac]/L[factors]/[model]/epoch[epoch].pickle

Created on Sat Jun  5 14:30:05 2021

@author: townesf
"""

#import json
#import numpy as np
import pandas as pd
from os import path,listdir,rmdir
from shutil import move
from argparse import ArgumentParser
from contextlib import suppress
from janitor import expand_grid
import tensorflow_probability as tfp
tfk = tfp.math.psd_kernels

from models import sf,cf,sfh,mefisto
from utils import training,misc
from utils.preprocess import load_data
from utils.visualize import gof,get_sparsity

#file scheme for spatial models:
#[dataset]/models/V[validation data fraction]/L[factors]/[likelihood]/[model]_[kernel]_M[inducing_pts]/epoch[epoch].pickle
#file scheme for nonspatial models:
#[dataset]/models/V[validation data fraction]/L[factors]/[likelihood]/[model]/epoch[epoch].pickle

def make_param_df(L, sp_mods, ns_mods, M, sz, V=5, kernels=["MaternThreeHalves"]):
  """
  L: number of latent dimensions/ components
  sp_mods: spatial models
  ns_mods: nonspatial models
  M: number of inducing points
  sz: size factor scheme (eg "constant")
  V: integer, percentage of data that is validation data (typically 5)
  kernels: list of kernel names for spatial models
  """
  cfg = {"V":V,"L":L, "model":sp_mods+ns_mods}#,"nb"
  spatial_cfg = {"model":sp_mods,
                 "kernel":kernels,
                 "M":M}
  mefisto_cfg = {"V":V,"L":L,
                 "model":["MEFISTO-G"],
                 "kernel":["ExponentiatedQuadratic"],
                 "M":M}
  a = expand_grid(others=cfg)
  b = expand_grid(others=spatial_cfg)
  a = a.merge(b,on="model",how="left",copy=False).convert_dtypes()
  m = expand_grid(others=mefisto_cfg)
  res = pd.concat([a,m],ignore_index=True)
  res[["model","lik0"]] = res["model"].str.split("-",expand=True)
  lconv = pd.DataFrame({"lik0":["P","N","G"],"lik":["poi","nb","gau"]})
  res = res.merge(lconv,on="lik0",how="left",copy=False)
  del res["lik0"]
  szconv = expand_grid(others={"lik":["poi","nb"],"sz":sz})
  res = res.merge(szconv,on="lik",how="left",copy=False)
  res.loc[pd.isna(res["sz"]),"sz"] = "constant"
  res["key"] = res.apply(misc.params2key,axis=1)
  res["converged"] = False
  return res

def correct_inducing_pts(csv_file, Ntr, ret=False):
  """
  csv_file : a results file from benchmarking
  Ntr : number of training observations in the dataset

  If the number of inducing points for spatial models is greater than Ntr,
  Truncate to a value of Ntr which is the actual number of IPs used
  Changes this in the CSV file as well as any folder names
  """
  pth = csv_file.split("/results/")[0]
  mpth = path.join(pth,"models")
  res = pd.read_csv(csv_file)
  for i in range(res.shape[0]):
    M = res.loc[i,"M"]
    if pd.isna(M) or M<=Ntr:
      continue
    else: #M>Ntr
      key = res.loc[i,"key"]
      old_path = path.join(mpth,key)
      new_key = key.replace("_M"+str(int(M)),"_M"+str(int(Ntr)))
      new_path = path.join(mpth,new_key)
      if path.isdir(new_path): #dir already exists, copy files
        for f in listdir(old_path):
          move(path.join(old_path, f), new_path)
        rmdir(old_path)
      else: #new dir does not exist yet
        #if old dir also does not exist, make no changes
        with suppress(FileNotFoundError):
          move(path.join(mpth,key),new_path)
      res.loc[i,"M"] = Ntr
      res.loc[i,"key"] = new_key
  res.to_csv(csv_file,index=False)
  if ret: return res

def choose_dataset(lik=None,model=None):
  if lik is None or lik=="gau":
    if model is None or model in ("RSF","FA"):
      use="ctr"
    else:
      use="norm"
  else:
    use="raw"
  return use

def init_model(D,p,opath,fmeans=None):
  """
  Initialize a model object for later fitting
  D: nested dictionary containing data, created by load_data
  p: dict-like object of model parameters
  opath: where should the pickled object be stored on disk (a folder)
  use: which dataset should be used for model fitting
  fmeans: for centered data only, the feature means needed for later prediction

  modifies p in-place if number of inducing pts is less than unique spatial locs

  returns:
    initialized model object
  """
  Ntr,J = D["raw"]["tr"]["Y"].shape
  is_spatial=False
  with suppress(TypeError):
  #  M = int(par['inducing_points'])
    ker = getattr(tfk,p['kernel']) #tensorflow probability object
    is_spatial=True

  L = p['L']
  if is_spatial:
    Z=misc.kmeans_inducing_pts(D["raw"]["tr"]["X"], p['M'])
    p['M']=Z.shape[0]

  #initialize the model object
  nonneg = p['model'] in ("NSF","PNMF","NSFH")
  # init_loadings_flag = True
  if p['model'] in ("NSF","RSF"):
    fit = sf.SpatialFactorization(J, L, Z, lik=p['lik'], psd_kernel=ker,
                                  nugget=1e-5, length_scale=0.1, disp="default",
                                  nonneg=nonneg, isotropic=True,
                                  feature_means=fmeans)
  elif p['model'] in ("PNMF","FA"):
    fit = cf.CountFactorization(Ntr, J, L, lik=p['lik'], nonneg=nonneg,
                                disp="default", feature_means=fmeans)
  elif p['model'] in ("NSFH","RSFH"):
    fit = sfh.SpatialFactorizationHybrid(Ntr, J, L, Z, lik=p['lik'],
                                         nonneg=nonneg, psd_kernel=ker,
                                         isotropic=True, nugget=1e-5,
                                         length_scale=0.1, disp="default",
                                         feature_means=fmeans)
    # init_loadings_flag = False
  elif p['model']=="MEFISTO":
    fit = mefisto.MEFISTO(D["norm"]["tr"], L, inducing_pts=p['M'], quiet=False,
                          pickle_path=opath)
  else:
    raise ValueError("model type {} not recognized".format(p['model']))
  # if init_loadings_flag:
  use = choose_dataset(lik=p['lik'],model=p['model'])
  Dtr = D[use]["tr"]
  fit.init_loadings(Dtr["Y"], X=Dtr["X"], sz=Dtr["sz"], shrinkage=0.3)
  return fit

def fit_model(D,fit,p,opath):
  #run the training procedure
  if p['model']=="MEFISTO":
    fit.train()
    return None
  else:
    tro = training.ModelTrainer(fit, lr=0.01, pickle_path=opath, max_to_keep=3)
    use = choose_dataset(lik=p['lik'],model=p['model'])
    tro.train_model(D[use]["tf"][0], D[use]["tf"][1], Dval=None, S=3, verbose=True,
                    num_epochs=10000, ckpt_freq=50, kernel_hp_update_freq=10,
                    status_freq=10, span=100, tol=5e-5, pickle_freq=None,
                    lr_reduce=0.5, maxtry=10)
    return tro

def val2train_frac(V):
  return (100.0-V)/100

def benchmark(ID,dataset):
  """
  Run benchmarking on dataset for the model specified in benchmark.csv in row ID.
  """
  # dsplit = dataset.split("/")
  dsplit = dataset.split("/data/")
  # dname = dsplit[1]
  # pth = path.join(*dsplit[:2])
  pth = dsplit[0]
  csv_file = path.join(pth,"results/benchmark.csv")
  # print(csv_file)
  #CFG = "config.json"
  #header of CSV is row zero
  p = misc.read_csv_oneline(csv_file,ID-1)
  # with suppress(FileNotFoundError):
  #   with open(path.join(pth,CFG)) as ifile:
  #       par.update(json.load(ifile))

  opath = path.join(pth,"models",p["key"])
  print("{}".format(p["key"]))
  if path.isfile(path.join(opath,"converged.pickle")):
    print("Benchmark already complete, exiting.")
    return None
  else:
    print("Starting benchmark.")
    train_frac = val2train_frac(p["V"])
    D,fmeans = load_data(dataset,model=p['model'],lik=p['lik'],sz=p['sz'],
                         train_frac=train_frac)
    fit = init_model(D,p,opath,fmeans=fmeans)
    tro = fit_model(D,fit,p,opath)
    return tro

def load(pkl):
  """
  Load either a fitted MEFISTO object or ModelTrainer object from pickle path pkl
  """
  if "MEFISTO" in pkl:
    fit = mefisto.MEFISTO.from_pickle(pkl)
    tro = None
  else:
    tro = training.ModelTrainer.from_pickle(pkl)
    fit = tro.model
  return fit,tro

def get_metrics(fit,Dtr,Dval=None,tro=None):
  res = {}
  if tro is None: #MEFISTO
    res["converged"] = fit.converged
    res["epochs"] = fit.epoch
    res["ptime"] = fit.ptime
    res["wtime"] = fit.wtime
    res["elbo_avg_tr"] = fit.elbos[-1]/Dtr["Y"].shape[0]
  else:
    res["converged"] = tro.converged
    res["epochs"] = tro.epoch.numpy()
    res["ptime"] = tro.ptime.numpy()
    res["wtime"] = tro.wtime.numpy()
    res["elbo_avg_tr"] = -fit.validation_step(Dtr, S=10).numpy()
    if Dval:
      res["elbo_avg_val"] = -fit.validation_step(Dval, S=10).numpy()
  dev = gof(fit,Dtr,Dval=Dval,S=10,plot=False)
  for i in dev["tr"]:
    res["dev_tr_"+i] = dev["tr"][i]
  res["rmse_tr"] = dev["rmse_tr"]
  with suppress(KeyError):
    for i in dev["val"]:
      res["dev_val_"+i] = dev["val"][i]
    res["rmse_val"] = dev["rmse_val"]
  res["sparsity"] = get_sparsity(fit)
  return res

def update_results(dataset,val_pct=5,todisk=True,verbose=True):
  # dataset = "scrna/visium_brain_sagittal/data/visium_brain_sagittal_J2000.h5ad"
  dsplit = dataset.split("/data/")
  pth = dsplit[0]
  csv_file = path.join(pth,"results/benchmark.csv")
  res = pd.read_csv(csv_file)
  train_frac = val2train_frac(val_pct)
  D,fmeans = load_data(dataset, model=None, lik=None, sz="constant",
                       train_frac=train_frac)
  #Dsz_sc is the same data for lik=(nb,poi) and model=(NSF,PNMF,NSFH)
  Dsz_sc,fmeans2 = load_data(dataset, model="NSF", lik="poi", sz="scanpy",
                             train_frac=train_frac)
  # mnames = ["epochs","ptime","wtime","dev_tr","dev_val"]
  #row = res.iloc[149,:]
  def row_metrics(row):
    row = dict(row) #originally row is a pandas.Series object
    # metrics = row[["converged"]+mnames] #old values
    if row["V"]!=val_pct: #skip rows with different val pct
      return row
    # or not row['converged']
    if not "converged" in row or pd.isnull(row["converged"]):# or row[mnames].isnull().any():
      pkl = path.join(pth,"models",row["key"])
      # use = choose_dataset(lik=row['lik'],model=row['model'])
      DD = Dsz_sc if row["sz"]=="scanpy" else D
      with suppress(FileNotFoundError):
        fit,tro = load(pkl)
        if verbose: print(row["key"])
        metrics = get_metrics(fit,DD["raw"]["tr"],DD["raw"]["val"],tro=tro)
        row.update(metrics)
    return row
  res = res.apply(row_metrics,axis=1,result_type="expand")
  if todisk: res.to_csv(csv_file,index=False)
  return res

def arghandler(args=None):
    """parses a list of arguments (default is sys.argv[1:])"""
    parser = ArgumentParser()
    parser.add_argument("id", type=int,
                        help="line in benchmark csv from which to get parameters")
    parser.add_argument("dataset", type=str,
                        help="location of scanpy H5AD data file")
    args = parser.parse_args(args) #if args is None, this will automatically parse sys.argv[1:]
    return args

if __name__=="__main__":
  # #should batch_size and/or learning_rate be input params?
  # #input from argparse
  # ID = 2 #slurm job array uses 1-based indexing (2,3,...,43)
  # #start with 2 to avoid header row of CSV
  # DATASET = "scrna/visium_brain_sagittal/data/visium_brain_sagittal_J2000.h5ad"
  args = arghandler()
  tro = benchmark(args.id, args.dataset)
