#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for visualization

Created on Wed May 19 09:58:59 2021

@author: townesf
"""

import numpy as np
import matplotlib.pyplot as plt
from contextlib import suppress

from utils.misc import poisson_deviance,dev2ss

def heatmap(X,y,figsize=(6,4),bgcol="gray",cmap="turbo",**kwargs):
  fig,ax=plt.subplots(figsize=figsize)
  ax.set_facecolor(bgcol)
  ax.scatter(X[:,0],X[:,1],c=y,cmap=cmap,**kwargs)
  # fig.show()

def hide_spines(ax):
  for side in ax.spines:
    ax.spines[side].set_visible(False)

def color_spines(ax,col="black"):
  # ax.spines['top'].set_color(col)
  # ax.spines['right'].set_color(col)
  # ax.spines['bottom'].set_color(col)
  # ax.spines['left'].set_color(col)
  for side in ax.spines:
    ax.spines[side].set_color(col)

def set_titles(fig,titles,**kwargs):
  for i in range(len(titles)):
    ax = fig.axes[i]
    ax.set_title(titles[i],**kwargs)

def hide_axes(ax):
  ax.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)

def multiheatmap(X, Y, grid, figsize=(6,4), cmap="turbo", bgcol="gray",
                 axhide=True, subplot_space=None, spinecolor=None,
                 savepath=None, **kwargs):
  if subplot_space is not None:
    gridspec_kw = {'wspace':subplot_space, 'hspace':subplot_space}
  else:
    gridspec_kw = {}
  fig, axgrid = plt.subplots(*grid, figsize=figsize, gridspec_kw=gridspec_kw)
  # if subplot_space is not None:
  #   plt.subplots_adjust(wspace=subplot_space, hspace=subplot_space)
  for i in range(Y.shape[1]):
    ax = fig.axes[i]
    ax.set_facecolor(bgcol)
    ax.scatter(X[:,0],X[:,1],c=Y[:,i],cmap=cmap,**kwargs)
    if spinecolor is not None: color_spines(ax,col=spinecolor)
    if axhide: hide_axes(ax)
    # with suppress(TypeError,IndexError):
    #   ax.set_title(titles[i],position=ttl_pos)
  # fig.tight_layout()
  if savepath:
    fig.savefig(savepath,bbox_inches='tight')
  return fig,axgrid

def plot_loss(loss_dict,title=None,ss=None,train_col="blue",val_col="red"):
  tr = np.array(loss_dict["train"])
  val = np.array(loss_dict["val"])
  if ss is None:
    plt.plot(tr,c=train_col,label="train")
    plt.plot(val,c=val_col,label="val")
  else:
    ss = list(ss)
    plt.plot(ss,tr[ss],c=train_col,label="train")
    plt.plot(ss,val[ss],c=val_col,label="val")
  if title is not None: plt.title(title)
  plt.xlabel("epoch")
  plt.ylabel("ELBO loss")
  plt.legend()
  plt.show()

def get_loadings(fit):
  with suppress(AttributeError): #MEFISTO
    return fit.get_loadings()
  with suppress(AttributeError): #CF
    return fit.V.numpy()
  with suppress(AttributeError): #PF
    return fit.W.numpy()
  with suppress(AttributeError): #PFH
    W = fit.spat.W.numpy()
    V = fit.nsp.V.numpy()
    return np.concatenate((W,V),axis=1)

def get_sparsity(fit,tol=1e-6):
  W = get_loadings(fit)
  return (np.abs(W)<tol).mean()

def plot_gof(Ytr, Mu_tr, Yval=None, Mu_val=None, title=None, loglog=False,
             lognorm=False):
  plt.scatter(Ytr.flatten(),Mu_tr.flatten(),c="blue",label="train")
  if Yval is not None and Mu_val is not None:
    plt.scatter(Yval.flatten(),Mu_val.flatten(),c="red",label="val")
  if loglog:
    plt.xscale("symlog")
    plt.yscale("symlog")
  plt.axline((0,0),(1,1),c="black",ls="--",lw=2)
  if lognorm:
    plt.xlabel("observed log-normalized counts")
  else:
    plt.xlabel("observed counts")
  plt.ylabel("predicted mean")
  if title is not None: plt.title(title)
  plt.legend()
  plt.show()

def gof(fit,Dtr,Dval=None,title=None,S=10,plot=True,**kwargs):
  """
  fit: an object with a predict method (eg ProcessFactorization, MEFISTO, etc)
  **kwargs passed to plot_gof
  """
  Mu_tr,Mu_val = fit.predict(Dtr,Dval=Dval,S=S)
  Ytr = Dtr["Y"]
  dev = {"tr":dev2ss(poisson_deviance(Ytr,Mu_tr,agg="mean",axis=0))}
  if Dval:
    Yval = Dval["Y"]
    dev["val"]=dev2ss(poisson_deviance(Yval,Mu_val,agg="mean",axis=0))
  else:
    Yval = None
  if plot: plot_gof(Ytr,Mu_tr,Yval,Mu_val,title=title,**kwargs)
  return dev
