#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 09:05:19 2022

@author: townesf
"""
import random
import numpy as np
from pandas import get_dummies
from anndata import AnnData
from scanpy import pp
from utils import misc,preprocess
dtp = "float32"
random.seed(101)

def squares():
  A = np.zeros([12,12])
  A[1:5,1:5] = 1
  A[7:11,1:5] = 1
  A[1:5,7:11] = 1
  A[7:11,7:11] = 1
  return A

def corners():
  B = np.zeros([6,6])
  for i in range(6):
    B[i,i:] = 1
  A = np.flip(B,axis=1)
  AB = np.hstack((A,B))
  CD = np.flip(AB,axis=0)
  return np.vstack((AB,CD))

def scotland():
  A = np.eye(12)
  for i in range(12):
    A[-i-1,i] = 1
  return A

def checkers():
  A = np.zeros([4,4])
  B = np.ones([4,4])
  AB = np.hstack((A,B,A))
  BA = np.hstack((B,A,B))
  return np.vstack((AB,BA,AB))

def quilt():
  A = np.zeros([4,144])
  A[0,:] = squares().flatten()
  A[1,:] = corners().flatten()
  A[2,:] = scotland().flatten()
  A[3,:] = checkers().flatten()
  return A #basic block size is 12x12

def ggblocks():
  A = np.zeros( [ 4 , 36 ] )
  A[0, [ 1 , 6 , 7 , 8 , 13 ] ] = 1
  A[1, [ 3 , 4 , 5 , 9 , 11 , 15 , 16 , 17  ] ] = 1
  A[2, [ 18 , 24 , 25 , 30 , 31 , 32 ] ] = 1
  A[3, [ 21 , 22 , 23 , 28 , 34 ] ] = 1
  return A #basic block size is 6x6

def sqrt_int(x):
  z = int(round(x**.5))
  if x==z**2:
    return z
  else:
    raise ValueError("x must be a square integer")

def gen_spatial_factors(scenario="quilt",nside=36):
  """
  Generate the factors matrix for either the 'quilt' or 'ggblocks' scenario
  There are 4 basic patterns (L=4)
  There are N=(nside^2) observations.
  Returns:
    factor values[Nx4] matrix
  """
  if scenario=="quilt":
    A = quilt()
  elif scenario=="ggblocks":
    A = ggblocks()
  else:
    raise ValueError("scenario must be 'quilt' or 'ggblocks'")
  unit = sqrt_int(A.shape[1]) #quilt: 12, ggblocks: 6
  assert nside%unit==0
  ncopy = nside//unit
  N = nside**2 #36x36=1296
  L = A.shape[0] #4
  A = A.reshape((L,unit,unit))
  A = np.kron(A,np.ones((1,ncopy,ncopy)))
  F = A.reshape((L,N)).T #NxL
  return F

def gen_spatial_coords(N): #N is number of observations
  X = misc.make_grid(N)
  X[:,1] = -X[:,1] #make the display the same
  return preprocess.rescale_spatial_coords(X)

def gen_nonspatial_factors(N,L=3,nzprob=0.2,seed=101):
  rng = np.random.default_rng(seed)
  return rng.binomial(1,nzprob,size=(N,L))

def gen_loadings(Lsp, Lns=3, Jsp=0, Jmix=500, Jns=0, expr_mean=20.0,
                 mix_frac_spat=0.55, seed=101, **kwargs):
  """
  generate a loadings matrix L=components, J=features
  kwargs currently ignored
  """
  rng = np.random.default_rng(seed)
  J = Jsp+Jmix+Jns #total number of features
  if Lsp>0:
    w = rng.choice(Lsp,J,replace=True) #spatial loadings
    W = get_dummies(w).to_numpy(dtype=dtp) #JxLsp indicator matrix
  else:
    W = np.zeros((J,0))
  if Lns>0:
    v = rng.choice(Lns,J,replace=True) #nonspatial loadings
    V = get_dummies(v).to_numpy(dtype=dtp) #JxLnsp indicator matrix
  else:
    V = np.zeros((J,0))
  #pure spatial features
  W[:Jsp,:]*=expr_mean
  V[:Jsp,:]=0
  #features with mixed assignment to spatial and nonspatial components
  W[Jsp:(Jsp+Jmix),:]*=(mix_frac_spat*expr_mean)
  V[Jsp:(Jsp+Jmix),:]*=((1-mix_frac_spat)*expr_mean)
  #pure nonspatial features
  W[(Jsp+Jmix):,:]=0
  V[(Jsp+Jmix):,:]*=expr_mean
  return W,V

def sim2anndata(locs, outcome, spfac, spload, nsfac=None, nsload=None):
  """
  d: a dict returned by sim_quilt or sim_ggblocks
  returns: an AnnData object with both raw counts and scanpy normalized counts
  obsm slots: spatial, spfac, nsfac (coordinates and factors)
  varm slots: spload, nsload (loadings)
  """
  obsm = {"spatial":locs, "spfac":spfac, "nsfac":nsfac}
  varm = {"spload":spload, "nsload":nsload}
  ad = AnnData(outcome, obsm=obsm, varm=varm)
  ad.layers = {"counts":ad.X.copy()} #store raw counts before normalization changes ad.X
  #here we do not normalize because the total counts are actually meaningful!
  # pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
  pp.log1p(ad)
  #shuffle indices to disperse validation data throughout the field of view
  idx = list(range(ad.shape[0]))
  random.shuffle(idx)
  ad = ad[idx,:]
  return ad

def sim(scenario, nside=36, nzprob_nsp=0.2, bkg_mean=0.2, nb_shape=10.0,
        seed=101, **kwargs):
  """
  scenario: either 'quilt'(L=4), 'ggblocks'(L=4), or 'both'(L=8)
  N=number of observations is nside**2
  nzprob_nsp: for nonspatial factors, the probability of a "one" (else zero)
  bkg_mean: negative binomial mean for observations that are "zero" in the factors
  nb_shape: shape parameter of negative binomial distribution
  seed: for random number generation reproducibility
  kwargs: passed to gen_loadings
  """
  if scenario=="both":
    F1 = gen_spatial_factors(nside=nside,scenario="ggblocks")
    F2 = gen_spatial_factors(nside=nside,scenario="quilt")
    F = np.hstack((F1,F2))
  else:
    F = gen_spatial_factors(scenario=scenario,nside=nside)
  rng = np.random.default_rng(seed)
  N = nside**2
  X = gen_spatial_coords(N)
  W,V = gen_loadings(F.shape[1],seed=seed, **kwargs)
  U = gen_nonspatial_factors(N,L=V.shape[1],nzprob=nzprob_nsp,seed=seed)
  Lambda = bkg_mean+F@W.T+U@V.T #NxJ
  r = nb_shape
  Y = rng.negative_binomial(r,r/(Lambda+r))
  return sim2anndata(X,Y,F,W,nsfac=U,nsload=V)
