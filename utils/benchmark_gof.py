#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 08:50:51 2021

@author: townesf
"""

from argparse import ArgumentParser
from utils import benchmark

def arghandler(args=None):
    """parses a list of arguments (default is sys.argv[1:])"""
    parser = ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help="location of scanpy H5AD data file")
    parser.add_argument("val_pct", type=int,
                        help="percentage of data to be used as validation set (0-100)")
    args = parser.parse_args(args) #if args is None, this will automatically parse sys.argv[1:]
    return args

if __name__=="__main__":
  args = arghandler()
  # dat = "scrna/sshippo/data/sshippo_J2000.h5ad"
  res = benchmark.update_results(args.dataset, val_pct=args.val_pct, todisk=True)
