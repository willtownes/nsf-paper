#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python -m simulations.benchmark_gof simulations/bm_sp
"""
import numpy as np
import pandas as pd
from os import path
from argparse import ArgumentParser
from contextlib import suppress
from scanpy import read_h5ad
from scipy.spatial.distance import pdist,cdist
from scipy.stats.stats import pearsonr,spearmanr
from sklearn.cluster import KMeans
from sklearn import metrics

from utils import postprocess
from utils.preprocess import load_data
from utils import benchmark as ubm

def compare_to_truth(ad, Ntr, fit, model):
  ad = ad[:Ntr,:]
  X = ad.obsm["spatial"]
  #extract ground truth factors and loadings
  tru = postprocess.interpret_nonneg(ad.obsm["spfac"], ad.varm["spload"], 
                                     lda_mode=False)
  F0 = tru["factors"]
  W0 = tru["loadings"]
  FFd0 = pdist(F0) #cell-cell distances (vectorized)
  WWd0 = pdist(W0) #gene-gene distances (vectorized)
  ifit = postprocess.interpret_fit(fit,X,model)
  F = ifit["factors"]
  W = ifit["loadings"]
  Fpc = cdist(F0.T,F.T,metric=lambda x,y: abs(pearsonr(x,y)[0])).max(axis=1)
  Fsc = cdist(F0.T,F.T,metric=lambda x,y: abs(spearmanr(x,y)[0])).max(axis=1)
  Wpc = cdist(W0.T,W.T,metric=lambda x,y: abs(pearsonr(x,y)[0])).max(axis=1)
  Wsc = cdist(W0.T,W.T,metric=lambda x,y: abs(spearmanr(x,y)[0])).max(axis=1)
  res = {}
  res["factors_pearson"] = Fpc.tolist()
  res["factors_spearman"] = Fsc.tolist()
  res["loadings_pearson"] = Wpc.tolist()
  res["loadings_spearman"] = Wsc.tolist()
  FFd = pdist(F)
  WWd = pdist(W)
  res["dfactors_pearson"] = pearsonr(FFd0,FFd)[0]
  res["dfactors_spearman"] = spearmanr(FFd0,FFd)[0]
  res["dloadings_pearson"] = pearsonr(WWd0,WWd)[0]
  res["dloadings_spearman"] = spearmanr(WWd0,WWd)[0]
  nclust = W0.shape[1]
  km0 = KMeans(n_clusters=nclust).fit(W0).labels_
  km1 = KMeans(n_clusters=nclust).fit(W).labels_
  res["loadings_clust_ari"] = metrics.adjusted_rand_score(km0, km1)
  return res

def compare_to_truth_mixed(ad, Ntr, fit, model):
  ad = ad[:Ntr,:].copy()
  X = ad.obsm["spatial"]
  #extract factors and loadings
  tru = postprocess.interpret_nonneg_mixed(ad.obsm["spfac"], ad.varm["spload"], 
                                           ad.obsm["nsfac"], ad.varm["nsload"], 
                                           lda_mode=False)
  F0 = tru["spatial"]["factors"]
  W0 = tru["spatial"]["loadings"]
  H0 = tru["nonspatial"]["factors"]
  V0 = tru["nonspatial"]["loadings"]
  FH0 = np.concatenate((F0,H0),axis=1)
  WV0 = np.concatenate((W0,V0),axis=1)
  alpha0 = W0.sum(axis=1) #true spatial importances
  ifit = postprocess.interpret_fit(fit,X,model)
  if model == "NSFH":
    F = ifit["spatial"]["factors"]
    W = ifit["spatial"]["loadings"]
    H = ifit["nonspatial"]["factors"]
    V = ifit["nonspatial"]["loadings"]
    FH = np.concatenate((F,H),axis=1)
    WV = np.concatenate((W,V),axis=1)
    alpha = W.sum(axis=1) #estimated spatial importances
  else:
    FH = ifit["factors"]
    WV = ifit["loadings"]
    if model =="NSF":
      alpha = np.ones_like(alpha0)
    elif model=="PNMF":
      alpha = np.zeros_like(alpha0)
    else:
      alpha = None
  Fpc = cdist(FH0.T,FH.T,metric=lambda x,y: abs(pearsonr(x,y)[0])).max(axis=1)
  Fsc = cdist(FH0.T,FH.T,metric=lambda x,y: abs(spearmanr(x,y)[0])).max(axis=1)
  Wpc = cdist(WV0.T,WV.T,metric=lambda x,y: abs(pearsonr(x,y)[0])).max(axis=1)
  Wsc = cdist(WV0.T,WV.T,metric=lambda x,y: abs(spearmanr(x,y)[0])).max(axis=1)
  res = {}
  res["factors_pearson"] = Fpc.tolist()
  res["factors_spearman"] = Fsc.tolist()
  res["loadings_pearson"] = Wpc.tolist()
  res["loadings_spearman"] = Wsc.tolist()
  if alpha is None:
    sp_imp_dist = None
  else:
    sp_imp_dist = np.abs(alpha-alpha0).sum() #L1 distance
  res["spatial_importance_dist"] = sp_imp_dist
  nclust = WV0.shape[1]
  km0 = KMeans(n_clusters=nclust).fit(WV0).labels_
  km1 = KMeans(n_clusters=nclust).fit(WV).labels_
  res["loadings_clust_ari"] = metrics.adjusted_rand_score(km0, km1)
  return res
  
def row_metrics(row,pth,verbose=True,mode="bm_sp"):
  row = dict(row) #originally row is a pandas.Series object
  train_frac = ubm.val2train_frac(row["V"])
  if not "converged" in row or not row['converged']:# or row[mnames].isnull().any():
    dataset = path.join(pth,"data","S{}.h5ad".format(row["scenario"]))
    ad = read_h5ad(path.normpath(dataset))
    pkl = path.join(pth,"models",row["key"])
    if row["sz"]=="scanpy":
      D,fmeans = load_data(ad, model="NSF", lik="poi", sz="scanpy",
                           train_frac=train_frac, flip_yaxis=False)
    else:
      D,fmeans = load_data(ad, model=None, lik=None, sz="constant",
                           train_frac=train_frac, flip_yaxis=False) 
    with suppress(FileNotFoundError):
      fit,tro = ubm.load(pkl)
      if verbose: print(row["key"])
      metrics = ubm.get_metrics(fit,D["raw"]["tr"],D["raw"]["val"],tro=tro)
      row.update(metrics)
      Ntr = D["raw"]["tr"]["X"].shape[0]
      if mode=="bm_sp":
        metrics2 = compare_to_truth(ad, Ntr, fit, row["model"])
      elif mode=="bm_mixed":
        metrics2 = compare_to_truth_mixed(ad, Ntr, fit, row["model"])
      else:
        raise ValueError("mode must be either 'bm_sp' or 'bm_mixed'")
      row.update(metrics2)
  return row

def update_results(pth,todisk=True,verbose=True):
  """
  different than the utils.benchmark.update_results because it loads a separate
  dataset and model for each row of the results.csv to accommmodate multiple
  simulation scenarios
  """
  # pth = "simulations/bm_sp"
  csv_file = path.join(pth,"results/benchmark.csv")
  res = pd.read_csv(csv_file)
  res = res.apply(row_metrics, args=(pth,), axis=1, result_type="expand", 
                  verbose=verbose, mode=path.split(pth)[-1])
  if todisk: res.to_csv(csv_file,index=False)
  return res

def arghandler(args=None):
    """parses a list of arguments (default is sys.argv[1:])"""
    parser = ArgumentParser()
    parser.add_argument("path", type=str,
                        help="top level directory containing subfolders 'data', 'models', and 'results'.")
    args = parser.parse_args(args) #if args is None, this will automatically parse sys.argv[1:]
    return args

if __name__=="__main__":
  args = arghandler()
  res = update_results(args.path, todisk=True)
