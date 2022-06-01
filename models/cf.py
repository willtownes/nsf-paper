#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(probabilistic) Count Factorization.

Essentially a probabilistic version of NMF or factor analysis

Created on Tue Apr 13 17:16:22 2021

@author: townesf
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from os import path
from scipy.special import logsumexp
from tensorflow.linalg import matmul
from sklearn.decomposition import TruncatedSVD

from models import likelihoods
from utils import misc,nnfu
# from utils.postprocess import normalize_cols,shrink_nmf,balance_nonneg_components,interpret_nonneg

tfd = tfp.distributions
tfb = tfp.bijectors
tv = tfp.util.TransformedVariable
#dtp = tf.dtypes.float32
dtp = "float32"
#rng = tf.random.Generator()
rng = np.random.default_rng()

class CountFactorization(tf.Module):
  def __init__(self, N, J, L, lik="poi", nonneg=True, disp="default",
               feature_means=None, **kwargs):
    super().__init__(**kwargs)
    self.lik = lik
    self.nonneg = nonneg
    self._disp0 = disp
    if self.nonneg:
      self.V = tf.Variable(rng.exponential(size=(J,L)), dtype=dtp,
                           constraint=misc.make_nonneg, name="loadings")
    else:
      self.V = tf.Variable(rng.normal(size=(J,L)), dtype=dtp, name="loadings")
    self._init_misc(N)
    if self.lik=="gau" and not self.nonneg:
      self.feature_means = feature_means
    else:
      self.feature_means = None

  def _init_misc(self, N):
    """
    Basically this is what would have been in __init__ but we put here to be able
    to call it more than once
    """
    J,L = self.V.shape
    prior_mu, prior_sigma = misc.lnormal_approx_dirichlet(max(L,1.1))
    #H_prior is a univariate normal distribution
    #may need to change to MultivariateNormalDiag
    self.ploc = tf.Variable(prior_mu*tf.ones(L), dtype=dtp, name="prior_location")
    self.pscale = tv(prior_sigma*tf.ones(L), tfb.Softplus(), dtype=dtp, name="prior_scale")
    qloc = rng.normal(loc=prior_mu, scale=.01, size=(N,L))
    qscale = np.exp(rng.normal(loc=np.log(prior_sigma), scale=.01, size=(N,L)))
    self.qloc = tf.Variable(qloc, dtype=dtp, name="variational_location")
    self.qscale = tv(qscale, tfb.Softplus(), dtype=dtp, name="variational_scale")
    self.disp = likelihoods.init_lik(self.lik, J, disp=self._disp0, dtp=dtp)
    self.trvars = tuple(i for i in self.trainable_variables)

  def get_dims(self):
    return self.V.shape[1]

  def get_loadings(self):
    return self.V.numpy()

  def set_loadings(self,Vnew):
    self.V.assign(Vnew,read_value=False)

  def init_loadings(self,Y,X=None,sz=1,nmf=True,**kwargs):
    """
    Use either PCA or NMF to initialize the loadings matrix from data Y
    """
    if self.nonneg:
      if nmf: init_pnmf_with_nmf(self, Y, sz=sz, **kwargs)
      else: init_pnmf_rand(self, Y, sz=sz)
    else: #real-valued factors
      if self.lik in ("poi","nb"):
        pass #use GLM-PCA?
      elif self.lik=="gau":
        fit = TruncatedSVD(self.V.shape[1]).fit(Y)
        self.set_loadings(fit.components_.T)
      else:
        raise likelihoods.InvalidLikelihoodError

  def generate_pickle_path(self,sz,base=None):
    """
    sz : str
      Indicate what type of size factors are used (eg 'none' or 'scanpy').
    base : str, optional
      Parent directory for saving pickle files. The default is cwd.
    """
    pars = {"L":self.V.shape[1], "lik":self.lik, "sz":sz,
            "model":"PNMF" if self.nonneg else "FA"
            }
    pth = misc.params2key(pars)
    if base: pth = path.join(base,pth)
    return pth

  def index_qpars(self, idx=None, N=None):
    """
    Returns the (qloc,qscale) variational parameters corresponding to
    various sampling scenarios.
    1) Sample from vposterior for all observations (default):
      idx=None, N=None
    2) Sample from vposterior for specified index
      idx= a tensor or np array, N is ignored
    3) Sample from prior
      idx=None, N= a positive integer
    """
    if idx is not None: #draw from variational posterior for a minibatch
      return tf.gather(self.qloc,idx), tf.gather(self.qscale,idx)
    else: #no minibatch index
      if N is None: #draw from full variational posterior
        return self.qloc,self.qscale
      else: #draw from prior with specified number of predictive observations (N)
        qloc = tf.tile(tf.expand_dims(self.ploc,0),(N,1))
        qscale = tf.tile(tf.expand_dims(self.pscale,0),(N,1))
        return qloc, qscale

  def eval_kl_term(self, qpars):
    """
    KL divergence from prior to variational distribution.
    Note that qloc and qscale are (batch-size)xL tensors
    """
    ph = tfd.MultivariateNormalDiag(loc=self.ploc, scale_diag=self.pscale)
    qh = tfd.MultivariateNormalDiag(loc=qpars[0], scale_diag=qpars[1])
    return qh.kl_divergence(ph) #expect this to be vector of length batch_size

  def sample_latent_factors(self, S=1, idx=None, N=None, qpars=None):
    """
    Draw random samples of the latent factors H
    The sampling comes from the variational approximation to the posterior.
    This function is needed to compute the expected log-likelihood term of the
    ELBO.
    S is the number of random samples to draw from latent GPs
    If posterior is False, the draws are from the prior instead
    """
    if qpars is None:
      qpars = self.index_qpars(idx=idx,N=N)
    qloc, qscale = qpars
    N = qloc.shape[0]
    L = self.V.shape[1]
    eps = tf.random.normal((S,N,L))
    return qloc+qscale*eps #H, shape SxNxL

  def sample_predictive_mean(self, sz=1, S=1, idx=None, N=None, qpars=None):
    H = self.sample_latent_factors(S=S, idx=idx, N=N, qpars=qpars) #SxNxL
    if self.nonneg:
      if self.lik!="gau": H+=tf.math.log(sz)
      return matmul(tf.exp(H),self.V,transpose_b=True) #SxNxJ
    else:
      Lam = matmul(H,self.V, transpose_b=True)
      if self.lik=="gau":
        return Lam
      else:
        return tf.exp(tf.math.log(sz)+Lam)

  def elbo_avg(self, Y, sz=1, idx=None, S=1):
    """
    Parameters
    ----------
    Y : numpy array of multivariate outcomes (NxJ)
    idx : integer index of minibatch observations. If None, H is drawn from prior
        instead of the variational posterior
    S : integer, optional
        Number of random GP function evaluations to use. The default is 1.
        Larger S=more accurate approximation to true ELBO but slower

    Returns
    -------
    The numeric evidence lower bound value, divided by the number of observations

    Note: if Y is a minibatch then this is an unbiased estimate not the actual ELBO
    """
    batch_size,J = Y.shape
    qpars = self.index_qpars(idx=idx,N=batch_size)
    if idx is None: #prediction mode, sample from prior of H, q==p
      kl_term = 0.0
    else: #training mode, sample from variational posterior of H
      #kl_terms **is** affected by minibatching so use reduce_mean
      kl_term = tf.reduce_mean(self.eval_kl_term(qpars))
    Mu = self.sample_predictive_mean(sz=sz, S=S, qpars=qpars, idx=idx, N=batch_size) #SxNxJ
    eloglik = likelihoods.lik_to_distr(self.lik, Mu, self.disp).log_prob(Y)
    return J*tf.reduce_mean(eloglik) - kl_term

  def train_step(self, D, optimizer, *args, S=1, **kwargs):
    """
    Executes one training step and returns the loss.
    D is training data: a tensorflow dataset (from slices) of (Y,idx)
    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    *args, **kwargs: included only for compatibility with train_step signature of PF
    eg optimizer_k arg and Ntot and chol kwargs not needed here.
    This enables a unified interface in the ModelTrainer class
    """
    with tf.GradientTape() as tape:
      loss = -self.elbo_avg(D["Y"], sz=D["sz"], idx=D["idx"], S=S)
    gradients = tape.gradient(loss, self.trvars)
    # loss, gradients = self.elbo_avg_grad(D, S=S, Ntot=Ntot)
    optimizer.apply_gradients(zip(gradients, self.trvars))
    return loss

  def validation_step(self, D, S=1, **kwargs):
    """
    Compute the validation loss on held-out data D
    D is a tensorflow dataset (from slices) of (Y,)
    """
    return -self.elbo_avg(D["Y"], sz=D["sz"], idx=None, S=S)

  def predict(self, Dtr, Dval=None, S=10):
    """
    Here Dtr,Dval should be raw counts (not normalized or log-transformed)

    returns the predicted training data mean and validation data mean
    on the original count scale
    """
    Mu_tr = misc.t2np(self.sample_predictive_mean(sz=Dtr["sz"], S=S,
                                                  idx=Dtr["idx"]))
    if self.lik=="gau":
      sz_tr = Dtr["Y"].sum(axis=1)
      #note self.feature_means is None if self.nonneg=True
      misc.reverse_normalization(Mu_tr, feature_means=self.feature_means,
                                 transform=np.expm1, sz=sz_tr, inplace=True)
    if Dval:
      Yval = Dval["Y"]
      Mu_val = misc.t2np(self.sample_predictive_mean(sz=Dval["sz"], S=S,
                                                     idx=None, N=Yval.shape[0]))
      if self.lik=="gau":
        sz_val = Yval.sum(axis=1)
        misc.reverse_normalization(Mu_val, feature_means=self.feature_means,
                                   transform=np.expm1, sz=sz_val, inplace=True)
    else:
      Mu_val = None
    return Mu_tr,Mu_val

def init_pnmf_rand(fit, Y, X=None, sz=1):
  # V,vsum = postprocess.normalize_cols(fit.V.numpy().astype("float64"))
  V = fit.V.numpy()
  lvsum = np.log(V.sum(axis=0))
  H = fit.qloc.numpy()+lvsum
  #make H have same rowsums as Y/sz
  H -= logsumexp(H,axis=1,keepdims=True) #rowSums(eH)=1
  H += np.log(Y.sum(axis=1,keepdims=True))-np.log(sz)
  wt_to_V = H.mean(axis=0) - fit.ploc.numpy()
  H -= wt_to_V
  V *= np.exp(wt_to_V-lvsum)
  fit.set_loadings(V)
  # beta0 = H.mean(axis=0)
  # fit.ploc.assign(beta0,read_value=False)
  fit.qloc.assign(H,read_value=False)

def init_pnmf_with_nmf(fit, Y, X=None, sz=1, pseudocount=1e-2, factors=None,
                      loadings=None, shrinkage=0.2):
  L = fit.V.shape[1]
  kw = likelihoods.choose_nmf_pars(fit.lik)
  H,V = nnfu.regularized_nmf(Y, L, sz=sz, pseudocount=pseudocount, 
                             factors=factors, loadings=loadings, 
                             shrinkage=shrinkage, **kw)
  # eH = factors
  # V = loadings
  # N,L = fit.qloc.shape
  # if eH is None or V is None:
  #   kw = likelihoods.choose_nmf_pars(fit.lik)
  #   nmf = NMF(L,**kw)
  #   eH = nmf.fit_transform(Y)#/sz
  #   V = nmf.components_.T
  # # eH,V = postprocess.shrink_nmf(eH,V,shrinkage=shrinkage)
  # V = postprocess.shrink_loadings(V, shrinkage=shrinkage)
  # vsum = V.sum(axis=0)
  # eH = postprocess.shrink_factors(eH*vsum, shrinkage=shrinkage)
  # H = np.log(pseudocount+eH)-np.log(sz)
  # wt_to_V = H.mean(axis=0)-fit.ploc.numpy()
  # H-= wt_to_V
  # V*= np.exp(wt_to_V-np.log(vsum))
  fit.set_loadings(V)
  beta0 = H.mean(axis=0)
  fit.ploc.assign(beta0,read_value=False)
  fit.qloc.assign(H,read_value=False)
