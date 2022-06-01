#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions used to preprocess data and initialize things

Created on Wed May 19 09:56:01 2021

@author: townesf
"""

import numpy as np
from os import path
from math import ceil
from copy import deepcopy
from scipy import sparse
from contextlib import suppress
from scanpy import read_h5ad
from tensorflow import constant
from tensorflow.data import Dataset

def deviancePoisson(X, sz=None):
  """
  X is matrix-like with observations in ROWs and features in COLs,
  sz is a 1D numpy array that is same length as rows of X
    sz are the size factors for each observation
    sz defaults to the row means of X
    set sz to 1.0 to have "no size factors" (ie Poisson for absolute counts instead of relative).

  Note that due to floating point errors the results may differ slightly
  between deviancePoisson(X) and deviancePoisson(X.todense()), use higher
  floating point precision on X to reduce this.
  """
  dtp = X.dtype
  X = X.astype("float64") #convert dtype to float64 for better numerics
  if sz is None:
    sz = np.ravel(X.mean(axis=1))
  else:
    sz = np.ravel(sz).astype("float64")
  if len(sz)==1:
    sz = np.repeat(sz,X.shape[0])
  feature_sums = np.ravel(X.sum(axis=0))
  try:
    with np.errstate(divide='raise'):
      ll_null = feature_sums*np.log(feature_sums/sz.sum())
  except FloatingPointError:
    raise ValueError("column with all zeros encountered, cannot compute deviance")
  if sparse.issparse(X): #sparse scipy matrix
    LP = sparse.diags(1./sz)@X
    #LP is a NxJ matrix with same sparsity pattern as X
    LP.data = np.log(LP.data) #log transform nonzero elements only
    ll_sat = np.ravel(X.multiply(LP).sum(axis=0))
  else: #dense numpy array
    X = np.asarray(X) #avoid problems with numpy matrix objects
    sz = np.expand_dims(sz,-1) #coerce to Nx1 array for broadcasting
    with np.errstate(divide='ignore',invalid='ignore'):
      ll_sat = X*np.log(X/sz) #sz broadcasts across rows of X
    ll_sat = ll_sat.sum(axis=0, where=np.isfinite(ll_sat))
  return (2*(ll_sat-ll_null)).astype(dtp)

def denseBinomialDeviance(X,sz=None):
  """
  Xt is a 2D numpy array with observations in COLUMNs and features in ROWs,
    Xt is the transpose of the usual count data matrix
  sz is a 1D numpy array that is same length as cols of X
    sz should be the column sums of X
  This function does not work with scipy sparse arrays or numpy matrices!
  """
  if sz is None: sz = X.sum(axis=0)
  P = X/sz
  L1P = np.log1p(-P)
  with np.errstate(divide="ignore",invalid="ignore"):
    ll_sat = X*(np.log(P)-L1P)+sz*L1P
  ll_sat = ll_sat.sum(axis=1, where=np.isfinite(ll_sat))
  sz_sum = sz.sum()
  feature_sums = X.sum(axis=1)
  p = feature_sums/sz_sum
  l1p = np.log1p(-p)
  ll_null = feature_sums*(np.log(p)-l1p)+sz_sum*l1p
  return 2*(ll_sat-ll_null)

def rescale_spatial_coords(X,box_side=4):
  """
  X is an NxD matrix of spatial coordinates
  Returns a rescaled version of X such that aspect ratio is preserved
  But data are centered at zero and area of equivalent bounding box set to
  box_side^D
  Goal is to rescale X to be similar to a N(0,1) distribution in all axes
  box_side=4 makes the data fit in range (-2,2)
  """
  xmin = X.min(axis=0)
  X -= xmin
  x_gmean = np.exp(np.mean(np.log(X.max(axis=0))))
  X *= box_side/x_gmean
  return X - X.mean(axis=0)

# def split_anndata(ad, train_frac=0.95):
#   """
#   Split an anndata object into a training and validation set
#   """
#   N = ad.shape[0]
#   Ntr = round(train_frac*N)
#   return ad[:Ntr,:], ad[Ntr:,:]

def scanpy_sizefactors(Y):
  sz = Y.sum(axis=1,keepdims=True)
  return sz/np.median(sz)

def anndata_to_train_val(ad, layer=None, nfeat=None, train_frac=0.95,
                         sz="constant", dtp="float32", flip_yaxis=True):
  """
  Convert anndata object ad to a training data dictionary
  and a validation data dictionary
  Requirements:
  * rows of ad are pre-shuffled to ensure random split of train/test
  * spatial coordinates in ad.obsm['spatial']
  * features (cols) of ad sorted in decreasing importance (eg with deviance)
  """
  if nfeat is not None: ad = ad[:,:nfeat]
  N = ad.shape[0]
  Ntr = round(train_frac*N)
  X = ad.obsm["spatial"].copy().astype(dtp)
  if flip_yaxis: X[:,1] = -X[:,1]
  X = rescale_spatial_coords(X)
  if layer is None: Y = ad.X
  else: Y = ad.layers[layer]
  with suppress(AttributeError):
    Y = Y.toarray() #in case Y is a sparse matrix
  Y = Y.astype(dtp)
  Dtr = {"X":X[:Ntr,:], "Y":Y[:Ntr,:]}
  Dval = {"X":X[Ntr:,:], "Y":Y[Ntr:,:]}
  if sz=="constant":
    Dtr["sz"] = np.ones((Ntr,1),dtype=dtp)
    Dval["sz"] = np.ones((N-Ntr,1),dtype=dtp)
  elif sz=="mean":
    Dtr["sz"] = Dtr["Y"].mean(axis=1,keepdims=True)
    Dval["sz"] = Dval["Y"].mean(axis=1,keepdims=True)
  elif sz=="scanpy":
    Dtr["sz"] = scanpy_sizefactors(Dtr["Y"])
    Dval["sz"] = scanpy_sizefactors(Dval["Y"])
  else:
    raise ValueError("unrecognized size factors 'sz'")
  Dtr["idx"] = np.arange(Ntr)
  if Ntr>=N: Dval = None #avoid returning an empty array
  return Dtr,Dval

def center_data(Dtr_n,Dval_n=None):
  Dtr_c = deepcopy(Dtr_n)
  feature_means=Dtr_c["Y"].mean(axis=0)
  Dtr_c["Y"] -= feature_means
  if Dval_n:
    Dval_c = deepcopy(Dval_n)
    Dval_c["Y"] -= feature_means
  else:
    Dval_c = None
  return feature_means,Dtr_c,Dval_c

def minibatch_size_adjust(num_obs,batch_size):
  """
  Calculate adjusted minibatch size that divides
  num_obs as evenly as possible
  num_obs : number of observations in full data
  batch_size : maximum size of a minibatch
  """
  nbatch = ceil(num_obs/float(batch_size))
  return int(ceil(num_obs/nbatch))

def prepare_datasets_tf(Dtrain,Dval=None,shuffle=False,batch_size=None):
  """
  Dtrain and Dval are dicts containing numpy np.arrays of data.
  Dtrain must contain the key "Y"
  Returns a from_tensor_slices conversion of Dtrain and a dict of tensors for Dval
  """
  Ntr = Dtrain["Y"].shape[0]
  if batch_size is None:
    #ie one batch containing all observations by default
    batch_size = Ntr
  else:
    batch_size = minibatch_size_adjust(Ntr,batch_size)
  Dtrain = Dataset.from_tensor_slices(Dtrain)
  if shuffle:
    Dtrain = Dtrain.shuffle(Ntr)
  Dtrain = Dtrain.batch(batch_size)
  if Dval is not None:
    Dval = {i:constant(Dval[i]) for i in Dval}
  return Dtrain, Ntr, Dval

def load_data(dataset, model=None, lik=None, train_frac=0.95, sz="constant", 
              flip_yaxis=True):
  """
  dataset: the file path of a scanpy anndata h5ad file 
  --OR-- the AnnData object itself
  p: a dict-like object of model parameters
  """
  try:
    ad = read_h5ad(path.normpath(dataset))
  except TypeError:
    ad = dataset
  kw1 = {"nfeat":None, "train_frac":train_frac, "dtp":"float32",
         "flip_yaxis":flip_yaxis}
  kw2 = {"shuffle":False,"batch_size":None}
  D = {"raw":{}}
  Dtr,Dval = anndata_to_train_val(ad,layer="counts",sz=sz,**kw1)
  D["raw"]["tr"] = Dtr
  D["raw"]["val"] = Dval
  D["raw"]["tf"] = prepare_datasets_tf(Dtr,Dval=Dval,**kw2)
  fmeans=None
  if lik is None or lik=="gau":
    #normalized data
    Dtr_n,Dval_n = anndata_to_train_val(ad,layer=None,sz="constant",**kw1)
    D["norm"] = {}
    D["norm"]["tr"] = Dtr_n
    D["norm"]["val"] = Dval_n
    D["norm"]["tf"] = prepare_datasets_tf(Dtr_n,Dval=Dval_n,**kw2)
    if model is None or model in ("RSF","FA"):
      #centered features
      fmeans,Dtr_c,Dval_c = center_data(Dtr_n,Dval_n)
      D["ctr"] = {}
      D["ctr"]["tr"] = Dtr_c
      D["ctr"]["val"] = Dval_c
      D["ctr"]["tf"] = prepare_datasets_tf(Dtr_c,Dval=Dval_c,**kw2)
  return D,fmeans

# def split_data_tuple(D,train_frac=0.8,shuffle=True):
#   """
#   D is list of data [X,Y,sz]
#   leading dimension of each element of D must be the same
#   train_frac: fraction of observations that should go into training data
#   1-train_frac: observations for validation data
#   """
#   n = D[0].shape[0]
#   idx = list(range(n))
#   if shuffle: random.shuffle(idx)
#   ntr = round(train_frac*n)
#   itr = idx[:ntr]
#   ival = idx[ntr:n]
#   Dtr = []
#   Dval = []
#   for d in D:
#     shp = d.shape
#     if len(shp)==1:
#       Dtr.append(d[itr])
#       Dval.append(d[ival])
#     else:
#       Dtr.append(d[itr,:])
#       Dval.append(d[ival,:])
#   return tuple(Dtr),tuple(Dval)

# def calc_hidden_layer_sizes(J,T,N):
#   """
#   J is input dimension, T is output dimension, N is number of hidden layers.
#   Returns a tuple of length (N) containing the widths of hidden layers.
#   The spacing forms a linear decline from J to T
#   """
#   delta = float(J-T)/(N+1) #increment size
#   res = np.round(J-delta*np.array(range(1,N+1)))
#   #res = res.astype("int32").tolist()
#   #return [J]+res+[T]
#   return res.astype("int32")
