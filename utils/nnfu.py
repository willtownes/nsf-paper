#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for working with nonnegative factor models. See also
postprocess.py for more complex functions to facilitate interpretation.

Created on Thu Sep 23 13:30:10 2021

@author: townesf
"""
import numpy as np
from sklearn.decomposition import NMF
# from scipy.special import logsumexp
from utils.misc import lnormal_approx_dirichlet

def normalize_cols(W):
  """
  Rescale the columns of a matrix to sum to one
  """
  wsum = W.sum(axis=0)
  return W/wsum, wsum

def normalize_rows(W):
  """
  Rescale the rows of a matrix to sum to one
  """
  wsum = W.sum(axis=1)
  return (W.T/wsum).T, wsum

def shrink_factors(F,shrinkage=0.2):
  a = shrinkage
  if 0<a<1:
    fsum = F.sum(axis=1,keepdims=True)
    F = F*(1-a)+a*fsum/float(F.shape[1]) #preserve rowsums
  return F

def shrink_loadings(W,shrinkage=0.2):
  a = shrinkage
  if 0<a<1:
    wsum = W.sum(axis=0)
    W = W*(1-a)+a*wsum/float(W.shape[0]) #preserve colsums
  return W

def regularized_nmf(Y, L, sz=1, pseudocount=1e-2, factors=None,
                    loadings=None, shrinkage=0.2, **kwargs):
  """
  Run nonnegative matrix factorization on (obs x feat) matrix Y
  The factors and loadings matrices are shrunk toward an approximately
  symmetric Dirichlet distribution (equal weight to all features and topics).
  Factors are converted to log scale.

  Parameters
  ----------
  Y : numpy array
    Nonnegative matrix
  L : integer
    Number of nonnegative components
  sz : numeric, optional
    size factors. The default is 1.
  pseudocount : numeric, optional
    Small number to add to nonnegative factors before log transform.
    The default is 1e-2.
  factors : numpy array, optional
    User provided factor matrix. The default is None.
  loadings : numpy array, optional
    User provided loadings matrix. The default is None.
  shrinkage : numeric between zero and one, optional
    How much to shrink toward symmetric Dirichlet. The default is 0.2.
  **kwargs : additional keyword arguments passed to sklearn.decomposition.NMF

  Returns
  -------
  Factors on the log-scale and loadings on the nonnegative scale
  """
  eF = factors
  W = loadings
  if eF is None or W is None:
    nmf = NMF(L,**kwargs)
    eF = nmf.fit_transform(Y)#/sz
    W = nmf.components_.T
  W = shrink_loadings(W, shrinkage=shrinkage)
  wsum = W.sum(axis=0)
  eF = shrink_factors(eF*wsum, shrinkage=shrinkage)
  F = np.log(pseudocount+eF)-np.log(sz)
  prior_mu, prior_sigma = lnormal_approx_dirichlet(max(L,1.1))
  beta0 = prior_mu*np.ones(L)
  wt_to_W = F.mean(axis=0)- beta0
  F-= wt_to_W
  W*= np.exp(wt_to_W-np.log(wsum))
  return F,W

# def shrink_nmf(factors,loadings,shrinkage=0.2):
#   F = factors
#   W = loadings
#   a = shrinkage
#   if 0<a<1:
#     fsum = F.sum(axis=1,keepdims=True)
#     wsum = W.sum(axis=0)
#     J,L = W.shape
#     F = F*(1-a)+a*fsum/float(L) #preserve rowsums
#     W = W*(1-a)+a*wsum/float(J) #preserve colsums
#   return F,W

# def balance_nonneg_components(logfactors,loadings):
#   F = logfactors
#   W,wsum = normalize_cols(loadings) #colsums(W) now equal 1
#   F_lse = logsumexp(F,axis=0)
#   ld = F_lse+np.log(wsum) #total weight for each component
#   J,L = W.shape
#   N = F.shape[0]
#   #divide proportionally between loadings and factors
#   a = N/(N+J)
#   return F-F_lse+a*ld, W*np.exp((1-a)*ld)
