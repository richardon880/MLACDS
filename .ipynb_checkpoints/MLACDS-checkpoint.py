import numpy as np
import glob
import pandas as pd
import matplotlib.pylab as plt

#### Calculates the distance between two points when using Periodic Boundaries Conditions
def min_distance(x0, x1, dimensions):
    delta = np.absolute(x0 - x1)
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
    return np.sqrt((delta**2).sum(axis=-1))

#### Similar to min_distance
def vec_r(x0, x1, dimensions):
    r = []
    for i in range(0,3):
        dx = x1[i] - x0[i]
        if (dx >   dimensions[i] * 0.5):
            dx = dx -dimensions[i]
        if (dx <= -dimensions[i] * 0.5):
            dx = dx + dimensions[i]
        r.append(dx)
    return np.array(r)

#### Volume of a truncated sphere (this is used to calculate the local volume)
##https://mathworld.wolfram.com/Sphere-SphereIntersection.html
def partial_v(R,r,d):
    return np.pi*(R+r-d)**2*(d**2+2*d*r-3*r**2+2*d*R+6*r*R-3*R**2)/12./d

#### This functions returns the local volume fraction around all particles
#### x,y,z and a are the arrays with coordinates and radii of all particles
#### Np is the number of particles
#### r_cut is the cut-off
def cal_v_local(x,y,z,r_cut,Np,a):
    v_local = np.zeros(Np)
    for i in range(Np):
        ri = np.array([x[i],y[i],z[i]])
        vp = 0.
        for j in range(Np):
            rj = np.array([x[j],y[j],z[j]])
            d = min_distance(ri, rj, Ls)
            ### full in
            if d + a[i] < r_cut[i]:#r_cut[i] - a[i]:
                #print(i,j,d)
                vp = vp + 4.*np.pi*a[i]**3/3.
            ### partially in
            elif d - a[i] < r_cut[i]:# + a[i]:
                vp = vp + partial_v(r_cut[i],a[j],d)
                #print(partial_v(a[i],a[j],d))
                #print(i,j,d,'**')
        v_cut = (4./3.)*np.pi*r_cut[i]**3
        v_local[i] = vp/v_cut
    
    return v_local

##### Calculates D at infinity dilution (D0)
##### This is just used to normalize all D's
def cal_D0():
    ### Calc Do
    KT = 4.11e-21
    R = 0.5e-10 ##Ams
    mu = 0.00089 ##Pa.s
    #mu = 0.001 ##Pa.s
    return (KT/(6.*np.pi*mu*R))