#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Postprocessing functions, used for interpretation of fitted models

Created on Wed May 19 10:03:53 2021

@author: townesf
"""

import numpy as np
from copy import deepcopy
from pandas import DataFrame
from utils.misc import t2np
from utils.nnfu import normalize_rows,normalize_cols

def compare_loadings(W,Wtrue):
  W,_ = normalize_rows(W)
  Wtrue,_ = normalize_rows(Wtrue)
  WWt = W@W.T
  WWt_true = Wtrue@Wtrue.T
  return WWt.flatten(), WWt_true.flatten()

def rescale_as_lda(factors,loadings,sort=True):
  """
  Rescale nonnegative factors and loadings matrices to be
  comparable to LDA:
  Rows of factor matrix sum to one, cols of loadings matrix sum to one
  """
  W = deepcopy(loadings)
  eF = deepcopy(factors)
  W,wsum = normalize_cols(W)
  eF,eFsum = normalize_rows(eF*wsum)
  if sort:
    o = np.argsort(-eF.sum(axis=0))
    return W[:,o],eF[:,o],eFsum
  else:
    return W,eF,eFsum

def interpret_nonneg(factors,loadings,lda_mode=False,sort=True):
  """
  Rescale factors and loadings from a nonnegative factorization
  to improve interpretability. Two possible rescalings:

  1. Soft clustering of observations (lda_mode=True):
  Rows of factor matrix sum to one, cols of loadings matrix sum to one
  Returns a dict with keys: "factors", "loadings", and "factor_sums"
  factor_sums is the "n" in the multinomial
  (ie the sum of the counts per observations)

  2. Soft clustering of features (lda_mode=False):
  Rows of loadings matrix sum to one, cols of factors matrix sum to one
  Returns a dict with keys: "factors", "loadings", and "feature_sums"
  feature_sums is similar to an intercept term for each feature
  """
  if lda_mode:
    W,eF,eFsum = rescale_as_lda(factors,loadings,sort=sort)
    return {"factors":eF,"loadings":W,"totals":eFsum}
  else: #spatialDE mode
    eF,W,Wsum = rescale_as_lda(loadings,factors,sort=sort)
    return {"factors":eF,"loadings":W,"totals":Wsum}

def interpret_npf(fit,X,S=10,**kwargs):
  """
  fit: object of type PF with non-negative factors
  X: spatial coordinates to predict on
  returns: interpretable loadings W, factors eF, and total counts vector
  """
  Fhat = t2np(fit.sample_latent_GP_funcs(X,S=S,chol=False)).T #NxL
  return interpret_nonneg(np.exp(Fhat),fit.W.numpy(),**kwargs)

def interpret_ncf(fit,S=10,**kwargs):
  Hhat = t2np(fit.sample_latent_factors(S=S)) #NxL
  return interpret_nonneg(np.exp(Hhat),fit.V.numpy(),**kwargs)

def interpret_npfh(fit,X,S=10,lda_mode=False,sort=True):
  spat = interpret_npf(fit.spat,X,S=S,lda_mode=lda_mode,sort=sort)
  nsp = interpret_ncf(fit.nsp,S=S,lda_mode=lda_mode,sort=sort)
  #nu is either the total count for each observation or for each cell,
  #depending on lda_mode
  nu = spat["totals"]+nsp["totals"]
  alpha = (spat["totals"]/nu)[:,None]
  resc = "factors" if lda_mode else "loadings"
  spat[resc] *=alpha
  nsp[resc] *=(1-alpha)
  del spat["totals"]
  del nsp["totals"]
  return {"spatial":spat,"nonspatial":nsp,"totals":nu}

def npfh_factor_importance(inpfh,lda_mode=True):
  mat = "factors" if lda_mode else "loadings"
  imp_F = inpfh["spatial"][mat].sum(axis=0)
  imp_H = inpfh["nonspatial"][mat].sum(axis=0)
  imp = np.concatenate((imp_F,imp_H))
  labs = np.array(["spatial"]*len(imp_F)+["nonspatial"]*len(imp_H))
  # id1 = np.arange(len(imp_F))+1
  # id2 = np.arange(len(imp_H))+1
  # ids = np.concatenate((id1,id2))
  d = DataFrame({"factor_type":labs,"weight":imp})
  return d.sort_values("weight", axis=0, ascending=False, inplace=False)
