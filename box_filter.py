#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
__all__ = ['movsum', 'movmean', 'movmean2d', 'movmin', 'movmin2d', 'movmax', 'movmax2d']
from numpy import *
#%%
def movsum(im, r, *, axis=0):
    assert im.shape[axis] > 2*r+1
    a = list(range(im.ndim))
    a[0] = axis
    a[axis] = 0
    imv = im.transpose(a)
    tmp = cumsum(imv, axis=0)
    im1 = zeros(imv.shape)
    im1[r+1:-r] = tmp[2*r+1:]-tmp[:-2*r-1]
    im1[:r+1] = tmp[r:2*r+1]
    im1[-r:] = tmp[-1]-tmp[-2*r-1:-r-1]
    return im1.transpose(a)
def movmean(im, r, *, axis=0):
    ms = movsum(im, r, axis=axis)
    l = arange(r+1, 2*r+1)
    # broadcast scale
    s = ms.shape[axis]
    bs = hstack((1/l, 1/(2*r+1)*ones(s-2*r),1/l[::-1]))
    rs = list(ones(im.ndim, dtype=int64)) # shape for reshape
    rs[axis] = -1
    bs = bs.reshape(rs)
    return ms*bs
def movmean2d(im, r):
    im = movmean(im, r, axis=0)
    im = movmean(im, r, axis=1)
    return im

def movextremum(im, r, extremfunc, *, axis=0):
    """
    extremfunc = numpy.minimum or numpy.maximum
    """
    a = list(range(im.ndim))
    a[0] = axis
    a[axis] = 0
    imv = im.transpose(a) # then we can treat as axis=0
    im1 = imv.copy()
    for i in range(1, r+1):
        im1[:-i] = extremfunc(imv[i:], im1[:-i])
    for i in range(1, r+1):
        im1[i:] = extremfunc(imv[:-i], im1[i:])
    return im1.transpose(a)
def movmin(im, r, axis=0):
    return movextremum(im, r, minimum, axis=0)
def movmin2d(im, r):
    im1 = movmin(im, r, axis=0)
    im1 = movmin(im1, r, axis=1)
    return im1
def movmax(im, r, axis=0):
    return movextremum(im, r, maximum, axis=0)
def movmax2d(im, r):
    im1 = movmax(im, r, axis=0)
    im1 = movmax(im1, r, axis=1)
    return im1

#%%

#%%
#%%
#%%
#%%
#%%
#%%
