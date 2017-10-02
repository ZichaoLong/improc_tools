#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
__all__ = ['guided_filter_plan', 'imguidedfilter']
from numpy import *
from numpy.linalg import inv,norm
from scipy.interpolate import interp2d
from .box_filter import *
#%%
def guided_filter_plan(I, r, *, eps_ratio=1e-3, stride=1, **kw):
    eps = I.max()*eps_ratio
    # set I.ndim=4 for array manipulation broadcast
    assert squeeze(I).ndim in (2,3)
    if squeeze(I).ndim == 2:
        I = I[:,:,newaxis,newaxis]
    elif squeeze(I).ndim == 3:
        I = I[:,:,:,newaxis]
    n,m,channeln,_ = I.shape
    # downsample I for speeding up
    I1 = I[::stride,::stride]
    n1,m1,channeln,_ = I1.shape
    I1_T = I1.reshape((n1, m1, 1, channeln))
    A1 = movmean2d(I1*I1_T, r)
    I1_m = movmean2d(I1, r)
    A1 -= I1_m*I1_m.reshape((n1, m1, 1, channeln))
    A1 += eps*eye(channeln, channeln).reshape(1, 1, channeln, channeln)
    A1 = inv(A1) # pixel wise matrix inverse, note that A1.ndim=4 here
    def f(p):
        pshape = p.shape
        p = squeeze(p)
        if p.ndim == 3:
            q = zeros(p.shape)
            for i in range(p.shape[2]):
                q[:,:,i] = f(p[:,:,i])
            return q
        elif p.ndim == 2:
            assert pshape[0] == n and pshape[1] == m
            p1 = p[::stride,::stride].reshape(n1, m1, 1, 1) # set p1.ndim=4 for unified treatment
        p1_m = movmean2d(p1, r)
        a1 = zeros((n1, m1, channeln, 1))
        Corrp1I1 = movmean2d(p1*I1, r)-p1_m*I1_m
        for i in range(A1.shape[2]):
            a1[:,:,i,0] = (A1[:,:,i,:]*Corrp1I1[:,:,:,0]).sum(2)
        b1 = p1_m-(a1*I1_m).sum(2).reshape((n1, m1, 1, 1))
        # print(norm(p1.sum((2,3))-(a1*I1).sum((2,3))-b1.sum((2,3))))
        a1_bar = movmean2d(a1, r)
        b1_bar = movmean2d(b1, r)
        x = arange(0, m) # x = cols
        y = arange(0, n) # y = rows
        a_bar = zeros((n,m,channeln,1))
        b_bar = zeros((n,m,1,1))
        for i in range(channeln):
            # notice that when we call interp2d(x, y, z) on a regular grid 
            # with 1d-array x and y, and z.ndim==2, then z.shape=(y.size, x.size) 
            # but not (x.size, y.size)
            upsample_interp = interp2d(x[::stride], y[::stride], a1_bar[:,:,i,0])
            a_bar[:,:,i,0] = upsample_interp(x,y)
        upsample_interp = interp2d(x[::stride], y[::stride], b1_bar[:,:,0,0])
        b_bar[:,:,0,0] = upsample_interp(x,y)
        q = (a_bar*I).sum((2,3))+b_bar[:,:,0,0]
        return q.reshape(pshape)
    return f
def imguidedfilter(im, I, r, *, eps_ratio=1e-3, stride=1, **kw):
    F = guided_filter_plan(I, r, eps_ratio=eps_ratio, stride=stride)
    return F(im)









#%%
#%%
#%%
#%%
#%%
#%%
#%%
