#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 18:10:06 2021

@author: townesf
"""
import pathlib
import numpy as np
#from pickle import dump,load
from math import ceil
from copy import deepcopy
from dill import dump,load
from pandas import read_csv as pd_read_csv
from tensorflow import clip_by_value
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.utils import sparsefuncs
from anndata import AnnData
from squidpy.gr import spatial_neighbors,spatial_autocorr

def mkdir_p(pth):
  pathlib.Path(pth).mkdir(parents=True,exist_ok=True)

def rm_glob(pth,glob="*"):
  """
  Remove all files in the pth folder matching glob.
  """
  for i in pathlib.Path(pth).glob(glob):
    i.unlink(missing_ok=True)

def poisson_loss(y,mu):
    """
    Equivalent to the Tensorflow Poisson loss
    https://www.tensorflow.org/api_docs/python/tf/keras/losses/Poisson
    It's the negative log-likelihood of Poisson without the log y! constant
    """
    with np.errstate(divide='ignore',invalid='ignore'):
        res = mu-y*np.log(mu)
    return np.mean(res[np.isfinite(res)])

def poisson_deviance(y,mu,agg="sum",axis=None):
    """
    Equivalent to "KL divergence" between y and mu:
    https://scikit-learn.org/stable/modules/decomposition.html#nmf
    """
    with np.errstate(divide='ignore',invalid='ignore'):
        term1 = y*np.log(y/mu)
    if agg=="sum":
      aggfunc = np.sum
    elif agg=="mean":
      aggfunc = np.mean
    else:
      raise ValueError("agg must be 'sum' or 'mean'")
    term1 = aggfunc(term1[np.isfinite(term1)], axis=axis)
    return term1 + aggfunc(mu - y, axis=axis)

def rmse(y,mu):
  return np.sqrt(((y-mu)**2).mean())

def dev2ss(dev):
  """
  dev: a 1d array of deviance values (one per feature)
  returns: a dictionary with summary statistics (mean, argmax, and max)
  """
  return {"mean":dev.mean(), "argmax":dev.argmax(), "max":dev.max(), "med":np.median(dev)}

def make_nonneg(x):
  return clip_by_value(x,0.0,np.inf)

def make_grid(N,xmin=-2,xmax=2,dtype="float32"):
  x = np.linspace(xmin,xmax,num=int(np.sqrt(N)),dtype=dtype)
  return np.stack([X.ravel() for X in np.meshgrid(x,x)],axis=1)

def kmeans_inducing_pts(X,M):
  M = int(M)
  Z = np.unique(X, axis=0)
  unique_locs = Z.shape[0]
  if M<unique_locs:
    Z=KMeans(n_clusters=M).fit(X).cluster_centers_
  return Z

def lnormal_approx_dirichlet(L):
  """
  Approximate a symmetric, flat Dirichlet (alpha=L) of dimension L
  by L independent lognormal distributions.

  The approximation is by matching the marginal means and variances.

  Returns the tuple of (mu,sigma) lognormal parameters
  """
  sigma2 = np.log(2*L)-np.log(L+1) #note this is zero if L=1
  mu = -np.log(L)-sigma2/2.0 #also zero if L=1
  return mu, np.sqrt(sigma2)

def pickle_to_file(obj,fpath):
  """
  Given a tensorflow module 'model' and a string file path 'fpath',
  save a pickled version of model to the file at fpath
  """
  with open(fpath,"wb") as ofile:
    dump(obj,ofile)

def unpickle_from_file(fpath):
  """
  Given a tensorflow module 'model' and a string file path 'fpath',
  save a pickled version of model to the file at fpath
  """
  with open(fpath,"rb") as ifile:
    return load(ifile)

def reverse_normalization(X, feature_means=None, transform=None, sz=None,
                          inplace=False, round_int=True):
  """
  X: normalized data matrix with rows=observations
  sz: total counts for each row from before the normalization

  Rescales each row "i" of X to have total count equal to sz[i]

  modified from scanpy _normalize_data function

  transform: a function such as np.expm1 to reverse eg the log1p transformation
  feature_means: a vector of means that are added to columns of X to reverse
    any centering of the data

  Typically a normalization would be
  1. divide by size factor
  2. log1p transform
  3. Subtract off means of each feature

  So this function will do those in reverse order
  1. Add feature_means to columns of X (features)
  2. expm1 transform
  3. multiply by size factor

  If sz is None, round_int is ignored.
  """
  X = X.copy() if not inplace else X
  if issubclass(X.dtype.type, (int, np.integer)):
    X = X.astype(np.float64)
  if feature_means is not None:
    np.add(X, feature_means, out=X) #feature_means must broadcast with X
  if transform is not None:
    transform(X, out=X)
  if sz is not None:
    after = np.ravel(X.sum(axis=1))
    sz = np.asarray(sz) / after
    if issparse(X):
      sparsefuncs.inplace_row_scale(X, sz)
    else:
      np.multiply(X, sz[:, None], out=X)
    if round_int:
      np.round(X, out=X)
  if not inplace: return X

def rpad(x,size,vals=np.nan):
  #see discussion: https://stackoverflow.com/q/38191855
  #pad the 1D numpy array x out to size by adding vals to the right side
  #if size<len(x) does nothing and just returns x
  # xlen = len(x)
  xpad = size-len(x)
  if xpad<0: return x
  return np.pad(x, pad_width=(0,xpad), mode="constant", constant_values=vals)

def t2np(X):
  return X.numpy().mean(axis=0)

def params2key(p):
  """
  p: a dict with keys L, lik, model, sz and optionally
  kernel, M, and V

  returns a string for the pickle path
  """
  if p["lik"] in ("poi","nb"):
    p = deepcopy(p)
    p["lik"] = p["lik"]+"_sz-"+p["sz"]
  m = p["model"]
  fmt = "L{L}/{lik}/{model}"
  if "V" in p:
    fmt = "V{V}/"+fmt
  if m in ("PNMF","FA"):
    pass
  elif m in ("NSF","RSF","MEFISTO"):
    fmt+="_{kernel}_M{M}"
  elif m in ("NSFH","RSFH"):
    fmt+="_T{T}_{kernel}_M{M}"
    if "T" not in p:
      p = deepcopy(p)
      p["T"] = ceil(p["L"]/2.)
  else:
    raise ValueError("Invalid model {}".format(m))
  #handle case where
  # if "M" in p:
  #   p["M"] = round(p["M"]) #make sure M is an integer
  return fmt.format(**p)

def read_csv_oneline(csv_file,row_id):
  """
  Read a single row from a csv file into a pandas DataFrame
  row_id is an index of the lines in the csv file
  row zero is considered the header
  so row_id must be at least one
  the maximum row_id is the total number of lines in the file minus one
  """
  assert row_id>0
  #if row_id=1, skip=[]
  #if row_id=2, skip=[1]
  #if row_id=(file length-1), skip=[1,2,3,...,file_length-2]
  skip = range(1,row_id)
  #returns a pandas Series object
  return pd_read_csv(csv_file,skiprows=skip,nrows=1).iloc[0,:]

def dims_autocorr(factors,coords,sort=True):
  """
  factors: (num observations) x (num latent dimensions) array
  coords: (num observations) x (num spatial dimensions) array
  sort: if True (default), returns the index and I statistics in decreasing
    order of autocorrelation. If False, returns the index and I statistics
    according to the ordering of factors.

  returns: an integer array of length (num latent dims), "idx"
    and a numpy array containing the Moran's I values for each dimension

    indexing factors[:,idx] will sort the factors in decreasing order of spatial
    autocorrelation.
  """
  ad = AnnData(X=factors,obsm={"spatial":coords})
  spatial_neighbors(ad)
  df = spatial_autocorr(ad,mode="moran",copy=True)
  if not sort: #revert to original sort order
    df.sort_index(inplace=True)
  idx = np.array([int(i) for i in df.index])
  return idx,df["I"].to_numpy()

def nan_to_zero(X):
  X = deepcopy(X)
  X[np.isnan(X)] = 0
  return X
