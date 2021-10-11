#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 10:52:16 2021

@author: townesf
"""
from numpy import tile
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tv = tfp.util.TransformedVariable


class InvalidLikelihoodError(ValueError):
  pass

def init_lik(lik, J, disp="default", dtp="float32"):
  """
  Given a likelihood, number of features (J),
  and a dispersion parameter initialization value,
  Return a TransformedVariable representing a vector of trainable,
  feature-specific dispersion parameters.
  """
  if lik=="gau":
    #disp is the scale (st dev) of the normal distribution
    if disp=="default": disp=1.0
  elif lik=="nb":
    #var= mu + disp*mu^2, disp->0 is Poisson limit
    if disp=="default": disp=0.01
  elif lik=="poi": #lik="poi"
    disp = None
  else:
    raise InvalidLikelihoodError
  if disp is not None: #ie lik in ("gau","nb")
    disp = tv(tile(disp,J), tfb.Softplus(), dtype=dtp, name="dispersion")
  return disp

def lik_to_distr(lik, mu, disp):
  """
  Given a likelihood and a tensorflow TransformedVariable 'disp',
  Return a tensorflow probability distribution object with means 'pred_means'
  """
  if lik=="poi":
    return tfd.Poisson(mu)
  elif lik=="gau":
    return tfd.Normal(mu, disp)
  elif lik=="nb":
    return tfd.NegativeBinomial.experimental_from_mean_dispersion(mu, disp)
  else:
    raise InvalidLikelihoodError

def choose_nmf_pars(lik):
  if lik in ("poi","nb"):
    return {"beta_loss":"kullback-leibler", "solver":"mu", "init":"nndsvda"}
  elif lik=="gau":
    return {"beta_loss":"frobenius", "solver":"cd", "init":"nndsvd"}
  else:
    raise InvalidLikelihoodError
