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
from scipy.interpolate import interp1d
from scipy.spatial import Delaunay

from utils.misc import poisson_deviance,dev2ss,rmse

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
  for i in range(len(fig.axes)):
    ax = fig.axes[i]
    ax.set_facecolor(bgcol)
    if i<Y.shape[1]:
      ax.scatter(X[:,0],X[:,1],c=Y[:,i],cmap=cmap,**kwargs)
    if spinecolor is not None: color_spines(ax,col=spinecolor)
    if axhide: hide_axes(ax)
    # with suppress(TypeError,IndexError):
    #   ax.set_title(titles[i],position=ttl_pos)
  # fig.tight_layout()
  if savepath:
    fig.savefig(savepath,bbox_inches='tight')
  return fig,axgrid

def interpolate_loss(x):
  """Modify the 1D numpy array 'x' to fill NA values with 
  linear interpolation."""
  x = np.array(x)
  t = np.arange(len(x))
  bad = np.isnan(x)
  f = interp1d(t[~bad], x[~bad], kind='linear', fill_value="extrapolate", 
               assume_sorted=True)
  return f(t)

# def interpolate_loss_dict(loss):
#   """
#   Modify the validation loss in-place to fill in the missing epochs.
#   Makes it easier to visualize the loss trace plot
#   """
#   loss["val"] = interpolate_loss(loss["val"])
#   return loss

def plot_loss(loss_dict,title=None,ss=None,figsize=(6,4),
              train_col="blue",val_col="red",**kwargs):
  tr = np.array(loss_dict["train"])
  val = np.array(loss_dict["val"])
  if np.any(np.isnan(val)):
    val = interpolate_loss(val)
  fig,ax = plt.subplots(figsize=figsize)
  if ss is None:
    ax.plot(tr,c=train_col,label="train",**kwargs)
    ax.plot(val,c=val_col,label="val",**kwargs)
  else:
    ss = list(ss)
    ax.plot(ss,tr[ss],c=train_col,label="train",**kwargs)
    ax.plot(ss,val[ss],c=val_col,label="val",**kwargs)
  ax.set_xlabel("epoch")
  ax.set_ylabel("ELBO loss")
  ax.legend()
  if title is not None: ax.set_title(title)
  # if savepath:
  #   fig.savefig(savepath,bbox_inches='tight')
  return fig,ax
  # if ss is None:
  #   plt.plot(tr,c=train_col,label="train")
  #   plt.plot(val,c=val_col,label="val")
  # else:
  #   ss = list(ss)
  #   plt.plot(ss,tr[ss],c=train_col,label="train")
  #   plt.plot(ss,val[ss],c=val_col,label="val")
  # if title is not None: plt.title(title)
  # plt.xlabel("epoch")
  # plt.ylabel("ELBO loss")
  # plt.legend()
  # plt.show()

def get_loadings(fit):
  with suppress(AttributeError): #MEFISTO
    return fit.get_loadings()
  with suppress(AttributeError): #CF
    return fit.V.numpy()
  with suppress(AttributeError): #SF
    return fit.W.numpy()
  with suppress(AttributeError): #SFH
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
  dev["rmse_tr"] = rmse(Ytr,Mu_tr)
  if Dval:
    Yval = Dval["Y"]
    dev["val"]=dev2ss(poisson_deviance(Yval,Mu_val,agg="mean",axis=0))
    dev["rmse_val"] = rmse(Yval,Mu_val)
  else:
    Yval = None
  if plot: plot_gof(Ytr,Mu_tr,Yval,Mu_val,title=title,**kwargs)
  return dev

def bounding_box_tile(Z, N):
  """
  Z is a Mx2 matrix representing a set of unique data points spatial coords.
  Returns an approximately Nx2 matrix of evenly spaced points that form a grid 
  on the bounding box of Z with same aspect ratio as Z.
  """
  xl,yl = Z.min(axis=0)
  xh,yh = Z.max(axis=0)
  ar = (yh-yl)/(xh-xl)
  nx = np.sqrt(N/ar)
  ny = ar*nx
  xg = np.linspace(xl,xh,int(np.ceil(nx)))
  yg = np.linspace(yl,yh,int(np.ceil(ny)))
  return np.array(np.meshgrid(xg,yg)).reshape((2,len(xg)*len(yg))).T

def in_hull(p, hull):
  """
  Copied from https://stackoverflow.com/a/16898636
  Test if points in `p` are in `hull`

  `p` should be a `NxK` coordinates of `N` points in `K` dimensions
  `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
  coordinates of `M` points in `K`dimensions for which Delaunay triangulation
  will be computed
  """
  if not isinstance(hull,Delaunay):
    hull = Delaunay(hull)
  return hull.find_simplex(p)>=0

def hull_tile(Z, N):
  Zg = bounding_box_tile(Z,N)
  return Zg[in_hull(Zg,Z)]
  
