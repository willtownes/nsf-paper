#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes and functions for training and saving models

Created on Wed May 19 09:54:16 2021

@author: townesf
"""

import numpy as np
import tensorflow as tf
from time import time,process_time
from contextlib import suppress
#from shutil import rmtree
from tempfile import TemporaryDirectory
#from scipy.linalg import norm
#from scipy.stats import pearsonr,linregress
from os import path

from utils.misc import mkdir_p, pickle_to_file, unpickle_from_file, rpad

class NumericalDivergenceError(ValueError):
  pass

def truncate_history(loss_history, epoch):
  with suppress(AttributeError):
    epoch = epoch.numpy()
  cutoff = epoch+1
  for i in loss_history:
    loss_history[i] = loss_history[i][:cutoff]
  return loss_history

class ConvergenceChecker(object):
  def __init__(self,span,dtp="float64"):
    x = np.arange(span,dtype=dtp)
    x-= x.mean()
    X = np.column_stack((np.ones(shape=x.shape),x,x**2,x**3))
    self.U = np.linalg.svd(X,full_matrices=False)[0]

  def smooth(self,y):
    return self.U@(self.U.T@y)

  def subset(self,y,idx=-1):
    span = self.U.shape[0]
    lo = idx-span+1
    if idx==-1:
      return y[lo:]
    else:
      return y[lo:(idx+1)]

  def relative_change(self,y,idx=-1,smooth=True):
    y = self.subset(y,idx=idx)
    if smooth:
      y = self.smooth(y)
    prev=y[-2]
    return (y[-1]-prev)/(0.1+abs(prev))

  def converged(self,y,tol=1e-4,**kwargs):
    return abs(self.relative_change(y,**kwargs)) < tol

  def relative_change_all(self,y,smooth=True):
    n = len(y)
    span = self.U.shape[0]
    cc = np.tile([np.nan],n)
    for i in range(span,n):
      cc[i] = self.relative_change(y,idx=i,smooth=smooth)
    return cc

  def converged_all(self,y,tol=1e-4,smooth=True):
    cc = self.relative_change_all(y,smooth=smooth)
    return np.abs(cc)<tol

class ModelTrainer(object): #goal to change this to tf.module?
  """
  Time keeping policy:
    * when object is first created, elapsed wtime and ptime are set to 0.0
    * when train_model is called a wtic, ptic baseline is set
    * whenever a checkpoint is saved or the object is pickled,
    the elapsed time is added to wtime and ptime and a new tic baseline is set
    * wtic and ptic are not stored, only the elapsed time is stored
    * checkpointing and pickling require the user to specify the additional
    elapsed time. If no time has elapsed the user can supply 0.0 values.
  """
  def __init__(self, model, lr=0.01, pickle_path=None, max_to_keep=3, **kwargs):
    #ckpt_path="/tmp/tf_ckpts", #use temporary directory instead
    """
    **kwargs are passed to tf.optimizers.[Optimizer] constructor
    """
    self.loss = {"train":np.array([np.nan]), "val":np.array([np.nan])}
    self.model = model
    #optimizer for all variables except kernel hyperparams
    self.optimizer = tf.optimizers.Adam(learning_rate=lr, **kwargs)
    #optimizer for kernel hyperparams, does nothing for nonspatial models
    self.optimizer_k = tf.optimizers.Adam(learning_rate=0.01*lr, **kwargs)
    self.epoch = tf.Variable(0,name="epoch")
    # self.tries = tf.Variable(0,name="number of tries")
    self.ptime = tf.Variable(0.0,name="elapsed process time")
    self.wtime = tf.Variable(0.0,name="elapsed wall time")
    # self.ckpt_counter = tf.Variable(0,name="checkpoint counter")
    self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer,
                                    optimizer_k=self.optimizer_k,
                                    epoch=self.epoch, ptime=self.ptime,
                                    wtime=self.wtime)#,counter=self.ckpt_counter)
    self.set_pickle_path(pickle_path)
    self.converged=False

  # def reset(self,lr_reduce=0.5):
  #   """
  #   Reset everything and decrease the learning rate
  #   """
  #   self.loss = {"train":[np.nan],"val":[np.nan]}
  #   self.model.reset()
  #   cfg = self.optimizer.get_config()
  #   cfg["learning_rate"]*=lr_reduce
  #   self.optimizer = tf.optimizers.Adam.from_config(cfg)
  #   cfg_k = self.optimizer_k.get_config()
  #   cfg_k["learning_rate"]*=lr_reduce
  #   self.optimizer_k = tf.optimizers.Adam.from_config(cfg_k)
  #   self.epoch.assign(0)
  #   self.ptime.assign(0.0)
  #   self.wtime.assign(0.0)
  #   self.converged=False

  def multiply_lr(self,factor):
    lr = self.optimizer.learning_rate
    lr_old = lr.numpy()
    lr_new = lr_old*factor
    lr.assign(lr_new)
    lr_k = self.optimizer_k.learning_rate
    lr_k.assign(lr_k.numpy()*factor)
    return lr_old,lr_new

  def set_pickle_path(self,pickle_path):
    if pickle_path is not None:
      mkdir_p(pickle_path)
    self.pickle_path=pickle_path

  def update_times(self,pchg,wchg):
    self.ptime.assign_add(pchg)
    self.wtime.assign_add(wchg)
    return process_time(),time() #for resetting the counter

  def checkpoint(self,mgr,*args):
    """
    *args passed to update_times method
    """
    p,w = self.update_times(*args)
    mgr.save(checkpoint_number=self.epoch)
    return p,w

  def restore(self,ckpt_id):
    #implicitly resets self.wtime,self.ptime to what was in the checkpoint
    self.ckpt.restore(ckpt_id)#self.manager.latest_checkpoint)
    return process_time(),time()

  def pickle(self,*args):
    """
    *args passed to update_times method
    """
    p,w = self.update_times(*args)
    if self.converged:
      fname = "converged.pickle"
    else:
      fname = "epoch{}.pickle".format(self.epoch.numpy())
    pickle_to_file(self, path.join(self.pickle_path,fname))
    return p,w

  @staticmethod
  def from_pickle(pth,epoch=None):
    if epoch:
      fname = "epoch{}.pickle".format(epoch)
    else:
      fname = "converged.pickle"
    return unpickle_from_file(path.join(pth,fname))

  def _train_model_fixed_lr(self, ckpt_mgr, Dtrain, Ntr, Dval=None, S=3,
                           verbose=True, num_epochs=5000,
                           ptic = process_time(), wtic = time(), ckpt_freq=50,
                           kernel_hp_update_freq=10, status_freq=10,
                           span=100, tol=5e-5, pickle_freq=None):
    """
    Dtrain, Dval : tensorflow Datasets produced by prepare_datasets_tf func
    ckpt_mgr must store at least 2 checkpoints (max_to_keep)
    Ntr: total number of training observations, needed to adjust KL term in ELBO
    S: number of samples to approximate the ELBO
    verbose: should status updates be printed
    num_epochs: maximum passes through the data after which optimization will be stopped
    ptic,wtic: process and wall time baselines
    kernel_hp_update_freq: how often to update the kernel hyperparameters (eg every 10 epochs)
      updating less than once per epoch improves speed but reduces numerical stability
    status_freq: how often to check for convergence and print updates
    ckpt_freq: how often to save tensorflow checkpoints to disk
    span: when checking for convergence, how many recent observations to consider
    tol: numerical (relative) change below which convergence is declared
    pickle_freq: how often to save the entire object to disk as a pickle file
    """
    ptic,wtic = self.checkpoint(ckpt_mgr, process_time()-ptic, time()-wtic)
    self.loss["train"] = rpad(self.loss["train"],num_epochs+1)
    if pickle_freq is None: #only pickle at the end
      pickle_freq = num_epochs
    msg = '{:04d} train: {:.3e}'
    if Dval:
      msg += ', val: {:.3e}'
      self.loss["val"] = rpad(self.loss["val"],num_epochs+1)
    msg2 = "" #modified later to include rel_chg
    #model, optimizer, S, Ntr do not change during optimization
    # @tf.function
    # def train_step_chol(D):
    #   return self.model.train_step(D, self.optimizer, self.optimizer_k,
    #                                S=S, Ntot=Ntr, chol=True)
    # @tf.function
    # def train_step_nochol(D):
    #   return self.model.train_step(D, self.optimizer, self.optimizer_k,
    #                                S=S, Ntot=Ntr, chol=False)
    @tf.function
    def train_step(D, chol=True):
      # if chol: return train_step_chol(D)
      # else: return train_step_nochol(D)
      return self.model.train_step(D, self.optimizer, self.optimizer_k,
                                   S=S, Ntot=Ntr, chol=chol)
    cvg = 0 #increment each time we think it has converged
    cc = ConvergenceChecker(span)
    while (not self.converged) and (self.epoch < num_epochs):
      # try:
      epoch_loss = tf.keras.metrics.Mean()
      chol=(self.epoch % kernel_hp_update_freq==0)
      for D in Dtrain: #iterate through each of the batches
        epoch_loss.update_state(train_step(D,chol=chol))
      trl = epoch_loss.result().numpy()
      self.epoch.assign_add(1)
      i = self.epoch.numpy()
      self.loss["train"][i] = trl
      if not np.isfinite(trl) or trl>self.loss["train"][1]:
        raise NumericalDivergenceError
      if i%status_freq==0 or i==num_epochs:
        if Dval:
          val_loss = self.model.validation_step(Dval, S=S, chol=False).numpy()
          self.loss["val"][i] = val_loss
        if i>span: #checking for convergence
          rel_chg = cc.relative_change(self.loss["train"],idx=i)
          msg2 = ", chg: {:.2e}".format(-rel_chg)
          if abs(rel_chg)<tol: cvg+=1
          else: cvg=0
          if cvg>=2: #ie convergence has been detected twice in a row
            self.converged=True
            pickle_freq = i #ensures final pickling will happen
            self.loss = truncate_history(self.loss, i)
        if verbose:
          if Dval: print(msg.format(i,trl,val_loss)+msg2)
          else: print(msg.format(i,trl)+msg2)
      if i%ckpt_freq==0:
        ptic,wtic = self.checkpoint(ckpt_mgr, process_time()-ptic, time()-wtic)
      if self.pickle_path and i%pickle_freq==0:
        ptic,wtic = self.pickle(process_time()-ptic, time()-wtic)
      # except tf.errors.InvalidArgumentError as err: #cholesky failed
      #   j = i.numpy() #save the current epoch value for printing
      #   ptic,wtic = self.restore()
      #   # self.ckpt.restore(self.manager.latest_checkpoint) #resets i to last checkpt
      #   if ng < 1.0: ng *= 10.0
      #   else: raise err #nugget has gotten too big so just give up
      #   try: self.model.set_nugget(ng) #spatial or integrated model
      #   except AttributeError: raise err #nonspatial model
      #   if verbose:
      #     print("Epoch: {:04d}, numerical error, reverting to epoch {:04d}, \
      #           increase nugget to {:.3E}".format(j, i.numpy(), ng))
      #   self.loss = truncate_history(self.loss,i)
      #   continue

  def find_checkpoint(self, ckpt_freq, back=1, epoch0=0):
    """
    If checkpoints are saved every [ckpt_freq] epochs, and we want to go back
    at least [back*ckpt_freq] epochs from the current epoch,
    returns the epoch number where the files can be found
    For example, if we are at epoch 201, ckpt_freq=50, and back=1, we want to
    go back to the checkpoint saved at epoch 150 (NOT 200).
    If back=2, we would go back to epoch 100.
    If this takes it below zero, we truncate at zero, since there should always
    be a checkpoint at epoch 0. Note this assumption may be violated for
    models loaded by from_pickle. For example if pickling happened at epoch
    201, and further model fitting proceeded to hit a numerical divergence at
    epoch 209, this function would try to go back to epoch 150 but the checkpoint
    would not exist because the temporary directory would have been cleaned up.
    This is considered acceptable since objects loaded from pickle are assumed
    to be either already numerically converged or to have at least run a large
    number of epochs without numerical problems.
    """
    return max(ckpt_freq*(self.epoch.numpy()//ckpt_freq - back), epoch0)

  def train_model(self, *args, lr_reduce=0.5, maxtry=10, verbose=True,
                  ckpt_freq=50, **kwargs):
    """
    See train_model_fixed_lr for args and kwargs. This method is a wrapper
    that automatically tries to adjust the learning rate
    """
    # assert self.tries<=maxtry
    ptic=process_time()
    wtic=time()
    tries=0
    #set the earliest epoch that could be returned to if numerical divergence
    epoch0=self.epoch.numpy() #usually zero, unless loaded from a pickle file
    with TemporaryDirectory() as ckpth:
      if verbose: print("Temporary checkpoint directory: {}".format(ckpth))
      mgr = tf.train.CheckpointManager(self.ckpt, directory=ckpth, max_to_keep=maxtry)
      while tries < maxtry:
        try:
          self._train_model_fixed_lr(mgr, *args, ptic=ptic, wtic=wtic,
                                     verbose=verbose, ckpt_freq=ckpt_freq,
                                     **kwargs)
          if self.epoch>=len(self.loss["train"])-1: break #finished training
        except (tf.errors.InvalidArgumentError,NumericalDivergenceError) as err: #cholesky failure
          tries+=1
          if tries==maxtry: raise err
          #else: #not yet reached the maximum number of tries
          if verbose:
            msg = "{:04d} numerical instability (try {:d})"
            print(msg.format(self.epoch.numpy(),tries))
          new_epoch = self.find_checkpoint(ckpt_freq, back=1, epoch0=epoch0) #int
          ckpt = path.join(ckpth,"ckpt-{}".format(new_epoch))
          # self.reset(lr_reduce=lr_reduce)
          ptic,wtic = self.restore(ckpt)
          lr_old,lr_new=self.multiply_lr(lr_reduce)
          if verbose:
            msg = "{:04d} learning rate: {:.2e}"
            print(msg.format(new_epoch,lr_new))
    if verbose:
      msg = "{:04d} training complete".format(self.epoch.numpy())
      if self.converged:
        print(msg+", converged.")
      else:
        print(msg+", reached maximum epochs.")


# def check_convergence(x,epoch,tol=1e-4):
#   """
#   x: a vector of loss function values
#   epoch: index of x with most recent loss
#   """
#   # with suppress(AttributeError):
#   #   epoch = epoch.numpy()
#   prev = x[epoch-1]
#   return abs(x[epoch]-prev)/(0.1+abs(prev)) < tol

# def check_convergence_linear(y,pval=.05):
#   n = len(y)
#   x = np.arange(n,dtype=y.dtype)
#   x -= x.mean()
#   yctr = y-y.mean()
#   return linregress(x,y).pvalue>pval

# def standardize(x,dtp="float64"):
#   xm = x.astype(dtp)-x.mean(dtype=dtp)
#   return xm/norm(xm)

# def check_convergence_linear(y,z=None,tol=0.1):
#   if z is None:
#     z = standardize(np.arange(len(y)))
#   # return abs(pearsonr(x,idx)[0])<tol
#   return np.abs(np.dot(standardize(y),z))<tol

# def check_convergence_posthoc(x,tol,method="linear",span=50):
#   if method=="simple":
#     start = 2
#     f = lambda i: check_convergence(x,i,tol)
#   elif method=="linear":
#     start = span
#     z=standardize(np.arange(span))
#     f = lambda i: check_convergence_linear(x[(i-span+1):(i+1)],z=z,tol=tol)
#   else:
#     raise ValueError("method must be linear or simple")
#   cc = np.tile([False],len(x))
#   for i in range(start,len(x)):
#     cc[i] = f(i)
#   return cc

# def update_time(t=None,chg=None):
#   try:
#     return t+chg
#   except TypeError: #either t or chg is None
#     return chg #note this may not be None if t is None but chg is not
