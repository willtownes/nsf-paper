#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEFISTO wrapper functions

Created on Mon Jul 19 18:25:31 2021

@author: townesf
"""
#import json
import numpy as np
from os import path
from time import time,process_time
#from mofax import mofa_model
from mofapy2.run.entry_point import entry_point

from utils import misc
# from models.likelihoods import InvalidLikelihoodError


class MEFISTO(object):
  def __init__(self, Dtr, num_factors, inducing_pts=1000, quiet=False,
               pickle_path=None): #lik="gau"
    """Dtr should be normalized and log transformed data"""
    #https://nbviewer.jupyter.org/github/bioFAM/MEFISTO_tutorials/blob/master/MEFISTO_ST.ipynb
    # if lik=="gau": lik = "gaussian"
    # elif lik=="poi": lik = "poisson"
    # else: raise InvalidLikelihoodError
    lik = "gaussian"
    ent = entry_point()
    ent.set_data_options(use_float32=True, center_groups=True)
    ent.set_data_matrix([[Dtr["Y"]]],likelihoods=lik)
    #ent.set_data_from_anndata(adtr, likelihoods="gaussian")
    ent.set_model_options(factors=num_factors,ard_weights=False)
    ent.set_train_options(quiet=quiet)#iter=ne)
    ent.set_covariates([Dtr["X"]])
    #ent.set_covariates([adtr.obsm["spatial"]])
    Mfrac = float(inducing_pts)/Dtr["X"].shape[0]
    if Mfrac<1.0:
      ent.set_smooth_options(sparseGP=True, frac_inducing=Mfrac)
    else:
      ent.set_smooth_options(sparseGP=False)
    #ent.set_stochastic_options(learning_rate=0.5)
    ent.build()
    self.ent = ent
    self.ptime = 0.0
    self.wtime = 0.0
    self.elbos = np.array([])
    self.epoch = 0
    self.set_pickle_path(pickle_path)
    self.converged=False
    self.feature_means = Dtr["Y"].mean(axis=0)
    self.L = num_factors
    self.M = inducing_pts

  def init_loadings(self,*args,**kwargs):
    pass #only here for compatibility with PF, CF, etc

  def set_pickle_path(self,pickle_path):
    if pickle_path is not None:
      misc.mkdir_p(pickle_path)
    self.pickle_path=pickle_path

  def generate_pickle_path(self,base=None):
    pars = {"L":self.L,
            "lik":"gau",
            "model":"MEFISTO",
            "kernel":"ExponentiatedQuadratic",
            "M":self.M
            }
    pth = misc.params2key(pars)
    if base: pth = path.join(base,pth)
    return pth

  def pickle(self):
    """
    *args passed to update_times method
    """
    if self.converged:
      fname = "converged.pickle"
    else:
      fname = "epoch{}.pickle".format(self.epoch)
    misc.pickle_to_file(self, path.join(self.pickle_path,fname))

  @staticmethod
  def from_pickle(pth,epoch=None):
    if epoch:
      fname = "epoch{}.pickle".format(epoch)
    else:
      fname = "converged.pickle"
    return misc.unpickle_from_file(path.join(pth,fname))

  def train(self):
    ptic = process_time()
    wtic = time()
    self.ent.run()
    self.converged = self.ent.model.trained
    elbos = self.ent.model.getTrainingStats()["elbo"]
    self.elbos = elbos[~np.isnan(elbos)]
    self.epoch = max(len(self.elbos)-1, 0)
    self.ptime = process_time()-ptic
    self.wtime = time()-wtic
    if self.pickle_path: self.pickle()

  def get_loadings(self):
    return self.ent.model.nodes["W"].getExpectations()[0]["E"]

  def get_factors(self):
    return self.ent.model.nodes["Z"].getExpectations()["E"]

  def _reverse_normalization(self,Lambda,sz):
    return misc.reverse_normalization(Lambda, feature_means=self.feature_means,
                                      transform=np.expm1, sz=sz)

  def predict(self,Dtr,Dval=None,S=None):
    """
    Here Dtr,Dval should be raw counts (not normalized or log-transformed)

    returns the predicted training data mean and validation data mean
    on the original count scale

    S is not used, only here for compatibility with visualization functions
    """
    Wt = self.get_loadings().T
    Ftr = self.get_factors()
    sz_tr = Dtr["Y"].sum(axis=1)
    Mu_tr = self._reverse_normalization(Ftr@Wt, sz=sz_tr)
    if Dval:
      self.ent.predict_factor(new_covariates=Dval["X"])
      Fval = self.ent.Zpredictions['mean']
      sz_val = Dval["Y"].sum(axis=1)
      Mu_val = self._reverse_normalization(Fval@Wt, sz=sz_val)
    else:
      Mu_val = None
    return Mu_tr,Mu_val


# def predict(m,Dtr,Dtr_n,Dval=None):
#   """
#   m: a mofax mofa_model object

#   returns the predicted training data mean and validation data mean
#   on the original count scale
#   """
#   Wt = m.get_weights().T
#   Ftr = m.get_factors()
#   fmeans = Dtr_n["Y"].mean(axis=0)
#   sz_tr = Dtr["Y"].sum(axis=1)
#   Mu_tr = reverse_normalization(Ftr@Wt, feature_means=fmeans,
#                                 transform=np.expm1, sz=sz_tr)
#   if Dval:
#     Fval = m.get_interpolated_factors()
#     sz_val = Dval["Y"].sum(axis=1)
#     Mu_val = reverse_normalization(Fval@Wt, feature_means=fmeans,
#                                    transform=np.expm1, sz=sz_val)
#   else:
#     Mu_val = None
#   return Mu_tr,Mu_val


# def save(self,save_path):
#   stats = stats.copy()
#   stats["elbos"] = list(stats["elbos"])
#   if ent.model.trained:
#     fpath = path.join(save_path,"converged")
#   else:
#     epoch = len(stats["elbos"])-1
#     fpath = path.join(save_path,"epoch{}".format(epoch))
#   ent.save(fpath+".hdf5")#,save_data=False)
#   with open(fpath+".json","w") as ofile:
#     json.dump(stats,ofile)

# def load(pth,epoch=None):
#   if epoch:
#     fpath = path.join(pth,"epoch{}".format(epoch))
#   else:
#     fpath = path.join(pth,"converged")
#   with open(fpath+".json","r") as ifile:
#     stats = json.load(ifile)
#   stats["elbos"] = np.array(stats["elbos"])
#   return mofa_model(fpath+".hdf5"), stats
