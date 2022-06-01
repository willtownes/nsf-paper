#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial Factorization- Hybrid version with both spatial and nonspatial factors

Created on Wed Apr 21 17:02:48 2021

@author: townesf
"""

import numpy as np
import tensorflow as tf
from os import path
from math import ceil
from sklearn.decomposition import TruncatedSVD
from tensorflow.linalg import matmul, matrix_transpose
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
from utils import nnfu
from utils.misc import t2np,reverse_normalization,params2key,dims_autocorr
from models import cf,sf,likelihoods
dtp = "float32"

class SpatialFactorizationHybrid(tf.Module):
  def __init__(self, N, J, L, Z, T=None, lik="poi", nonneg=True,
               psd_kernel=tfk.MaternThreeHalves, isotropic=True,
               nugget=1e-5, length_scale=0.1, disp="default",
               feature_means=None, **kwargs):
    """
    Initialize a non-negative process factorization model with L spatial factors
    and T non-spatial factors.

    Parameters
    ----------
    N : TYPE
      Number of observations in training data.
    J : TYPE
      Number of features in training data.
    L : TYPE
      Total number of latent factors.
    T : TYPE
      Number of spatial factors, if None, set to ceil(L/2) by default
    Z : TYPE
      Inducing point locations (observations in rows, spatial dims in cols).
    nonneg : bool
      Should latent factors be nonnegative or real-valued
    """
    super().__init__(**kwargs)
    assert L>0
    if T is None: T = ceil(L/2.)
    assert T>0 and T<L
    # self.is_spatial=True
    self.lik = lik
    self.nonneg = nonneg
    self._disp0 = disp
    self.spat = sf.SpatialFactorization(J, T, Z, nonneg=nonneg, lik=lik,
                                        isotropic=isotropic,
                                        psd_kernel=psd_kernel, nugget=nugget,
                                        length_scale=length_scale, disp=None,
                                        feature_means=None) #spatial
    self.nsp = cf.CountFactorization(N, J, L-T, lik=lik, nonneg=nonneg,
                                     disp=None, feature_means=None) #nonspatial
    self.disp = likelihoods.init_lik(self.lik, J, disp=self._disp0, dtp=dtp)
    # self.trvars_all = self.trainable_variables
    # self.trvars_nonkernel = tuple(i for i in self.trvars_all if i.name[:10]!="gp_kernel/")
    if self.lik=="gau" and not self.nonneg:
      self.feature_means = feature_means
    else:
      self.feature_means = None
    self.trvars_nonkernel = self.spat.trvars_nonkernel+self.nsp.trvars
    if self.disp is not None:
      self.trvars_nonkernel+=self.disp.trainable_variables
    self.trvars_kernel = self.spat.trvars_kernel

  def get_dims(self):
    T = self.spat.W.shape[1]
    L = T+self.nsp.V.shape[1]
    return L,T

  def init_loadings(self,Y,X=None,sz=1,**kwargs):
    """
    Use either PCA or NMF to initialize the loadings matrix from data Y
    """
    L,T = self.get_dims()
    if self.nonneg:
      init_nsfh_with_nmf(self,Y,X=X,sz=sz,**kwargs)
    else: #real-valued factors
      if self.lik in ("poi","nb"):
        pass #use GLM-PCA?
      elif self.lik=="gau":
        fit = TruncatedSVD(L).fit(Y)
        self.spat.set_loadings(fit.components_[:T,:].T)
        self.nsp.set_loadings(fit.components_[T:,:].T)
      else:
        raise likelihoods.InvalidLikelihoodError

  def generate_pickle_path(self,sz,base=None):
    T = self.spat.W.shape[1]
    L = T + self.nsp.V.shape[1]
    pars = {"L":L, "T":T, "lik":self.lik, "sz":sz,
            "model":"NSFH" if self.nonneg else "RSFH",
            "kernel":self.spat.psd_kernel.__name__,
            "M":self.spat.Z.shape[0]}
    pth = params2key(pars)
    if base: pth = path.join(base,pth)
    return pth

  def sample_predictive_mean(self, X, idx=None, sz=1, S=1, kernel=None,
                             mu_z=None, Kuu_chol=None, chol=True, ns_qpars=None):
    """
    s_cached_vars: spatial cached vars, see method of NSF class
    ns_qloc,ns_qscale: nonspatial cached vars, see method of PCF class
    """
    #note, if Kuu_chol is not None, then chol flag is ignored
    F = self.spat.sample_latent_GP_funcs(X, S=S, kernel=kernel, mu_z=mu_z,
                                         Kuu_chol=Kuu_chol, chol=chol) #SxTxN
    H = self.nsp.sample_latent_factors(S=S, idx=idx, N=X.shape[0],
                                       qpars=ns_qpars) #SxNx(L-T)
    if self.nonneg:
      Lam1 = matmul(self.spat.W, tf.exp(F))#SxJxN
      Lam2 = matmul(tf.exp(H), self.nsp.V, transpose_b=True) #SxNxJ
      Lam = matrix_transpose(Lam1)+Lam2 #SxNxJ
      if self.lik=="gau":
        return Lam
      else:
        return sz*Lam
    else:
      Lam1 = matmul(self.spat.W, F) #SxJxN
      Lam2 = matmul(H, self.nsp.V, transpose_b=True) #SxNxJ
      Lam = matrix_transpose(Lam1)+Lam2
      if self.lik=="gau": #identity link, size factors ignored
        return Lam #SxNxJ
      else: #log link (poi, nb)
        return tf.exp(tf.math.log(sz)+Lam) #SxNxJ

  def elbo_avg(self, X, Y, idx=None, sz=1, S=1, Ntot=None, chol=True):
    batch_size, J = Y.shape
    if Ntot is None: Ntot = batch_size #no minibatch, all observations provided
    ker = self.spat.get_kernel()
    mu_z = self.spat.get_mu_z()
    Kuu_chol = self.spat.get_Kuu_chol(kernel=ker,from_cache=(not chol))
    #kl_terms is not affected by minibatching so use reduce_sum
    kl_term1 = tf.reduce_sum(self.spat.eval_kl_term(mu_z, Kuu_chol))
    ns_qpars = self.nsp.index_qpars(idx=idx,N=batch_size)
    if idx is None: #prediction mode, sample from prior of H
      kl_term2 = 0.0
    else: #training mode, sample from variational posterior of H
      kl_term2 = tf.reduce_mean(self.nsp.eval_kl_term(ns_qpars))
    Mu = self.sample_predictive_mean(X, idx=idx, sz=sz, S=S, kernel=ker, mu_z=mu_z,
                                     Kuu_chol=Kuu_chol, ns_qpars=ns_qpars) #SxNxJ
    eloglik = likelihoods.lik_to_distr(self.lik, Mu, self.disp).log_prob(Y)
    return J*tf.reduce_mean(eloglik) - kl_term1/Ntot - kl_term2

  def train_step(self, D, optimizer, optimizer_k, S=1, Ntot=None, chol=True):
    """
    Executes one training step and returns the loss.
    D is training data: a tensorflow dataset (from slices) of (X,Y,sz,idx)
    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape(persistent=True) as tape:
      loss = -self.elbo_avg(D["X"], D["Y"], idx=D["idx"], sz=D["sz"], S=S,
                            Ntot=Ntot, chol=chol)
    try:
      gradients = tape.gradient(loss, self.trvars_nonkernel)
      if chol:
        gradients_k = tape.gradient(loss, self.trvars_kernel)
        optimizer_k.apply_gradients(zip(gradients_k, self.trvars_kernel))
      optimizer.apply_gradients(zip(gradients, self.trvars_nonkernel))
    finally:
      del tape
    return loss

  def validation_step(self, D, S=1, chol=False):
    """
    Compute the validation loss on held-out data D
    D is a tensorflow dataset (from slices) of (X,Y,sz)
    """
    return -self.elbo_avg(D["X"], D["Y"], idx=None, sz=D["sz"], S=S, chol=chol)

  def predict(self, Dtr, Dval=None, S=10):
    """
    Here Dtr,Dval should be raw counts (not normalized or log-transformed)

    returns the predicted training data mean and validation data mean
    on the original count scale
    """
    Mu_tr = t2np(self.sample_predictive_mean(Dtr["X"], sz=Dtr["sz"],
                                             idx=Dtr["idx"], S=S))
    if self.lik=="gau":
      sz_tr = Dtr["Y"].sum(axis=1)
      #note self.feature_means is None if self.nonneg=True
      reverse_normalization(Mu_tr, feature_means=self.feature_means,
                            transform=np.expm1, sz=sz_tr, inplace=True)
    if Dval:
      Mu_val = t2np(self.sample_predictive_mean(Dval["X"], sz=Dval["sz"],
                                                idx=None, S=S))
      if self.lik=="gau":
        sz_val = Dval["Y"].sum(axis=1)
        reverse_normalization(Mu_val, feature_means=self.feature_means,
                              transform=np.expm1, sz=sz_val, inplace=True)
    else:
      Mu_val = None
    return Mu_tr,Mu_val

def init_nsfh_with_nmf(fit, Y, X=None, sz=1, pseudocount=1e-2, factors=None,
                       loadings=None, shrinkage=0.2):
  L,T = fit.get_dims()
  # M = fit.spat.Z.shape[0] #number of inducing points
  kw = likelihoods.choose_nmf_pars(fit.lik)
  FH,WV = nnfu.regularized_nmf(Y, L, sz=sz, pseudocount=pseudocount,
                               factors=factors, loadings=loadings,
                               shrinkage=shrinkage, **kw)
  # eF = factors
  # W = loadings
  # if eF is None or W is None:
  #   kw = likelihoods.choose_nmf_pars(fit.lik)
  #   nmf = NMF(L,**kw)
  #   eF = nmf.fit_transform(Y/sz)
  #   W = nmf.components_.T
  # W,wsum = normalize_cols(W)
  # eF *= wsum
  # eFm2 = eF.mean()/2
  # eF/=eFm2
  # W*=eFm2
  if X is not None:
    #sort factors in decreasing spatial autocorrelation
    moran_idx,moranI = dims_autocorr(FH,X)
    FH = FH[:,moran_idx]
    WV = WV[:,moran_idx]
  #spatial part
  fit.spat.set_loadings(WV[:,:T]) #W=spatial loadings
  U,beta0,beta = sf.smooth_spatial_factors(FH[:,:T], fit.spat.Z.numpy(), X=X)
  fit.spat.beta0.assign(beta0[:,None],read_value=False)
  fit.spat.delta.assign(U.T,read_value=False)
  if beta is not None: fit.spat.beta.assign(beta,read_value=False)
  #nonspatial part
  H = FH[:,T:]
  fit.nsp.set_loadings(WV[:,T:]) #V=nonspatial loadings
  beta0 = H.mean(axis=0)
  fit.nsp.ploc.assign(beta0,read_value=False)
  fit.nsp.qloc.assign(H,read_value=False)
  # fit.spat.init_loadings(Y, X=X, pseudocount=pseudocount,
  #                        factors=eF[:,:T], loadings=W[:,:T])
  # fit.nsp.init_loadings(Y, X=X, pseudocount=pseudocount,
  #                       factors=eF[:,T:], loadings=W[:,T:])