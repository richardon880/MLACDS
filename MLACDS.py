import numpy as np
import glob
from scipy.special import sph_harm
from scipy.interpolate import CubicSpline
import pandas as pd

import matplotlib.pyplot as plt #might get rid of automatic plotting later

from tqdm.notebook import tqdm_notebook

###These are the aevrage results for the "exact" calculation of Diffusion parameter. Will be used for interpolation to give the "theory curve" which can be used as a competitor model for the machine learning
D_snap = np.array([0.6549626681427589,
 0.5415773484728538,
 0.45611420397539226,
 0.3899386738814587,
 0.32989963726772464,
 0.27904761273590456])

p_snap= np.array([0.05,0.1,0.15,0.2,0.25,0.3])

stdD_snap = np.array([0.03135978350995926,
 0.0369455245372216,
 0.03509771542582867,
 0.02964564689529907,
 0.0264751917347454,
 0.02262952233632323])

### Competitor Model - Volume Correction ###
#takes Local volume as input param
VolumeCorrection = CubicSpline(p_snap, D_snap) # returns diffusion parameter as output


### Dist Calculation Functions ###
def min_distance(x0, x1, dimensions):
    """
    Function Calculates Distance Between 2 points
    when using periodic boundary conditions.
    x0 = array like len 3 (x,y,z position)
    x1 = as above, dist will be calculated between this and x0
    dimensions = array like len 3 (x,y,z len of box)
    """
    delta = np.absolute(x0 - x1) # diff between points
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta) #if diff between points is greater than half of the box dimensions then get distance through  
    return np.sqrt((delta**2).sum(axis=-1))


def vec_r(x0, x1, dimensions):
    """
    Similar to min distance
    """
    r = []
    for i in range(0,3):
        dx = x1[i] - x0[i]
        if (dx >   dimensions[i] * 0.5):
            dx = dx -dimensions[i]
        if (dx <= -dimensions[i] * 0.5):
            dx = dx + dimensions[i]
        r.append(dx)
    return np.array(r)


#### This functions gets the distance to neighbours that are within a r_cut distance.
def cal_dis(x,y,z,r_cut,ls,Np,a,N_N):
    n_neighbours = np.zeros(Np)
    ql = np.zeros((Np,Np))#len(ls)))
    ######### Getting the neighbours list
    ii = []
    #r_cut = 6*a
    for i in range(Np):
        ri = np.array([x[i],y[i],z[i]])
        tmp = []
        for j in range(Np):
            rj = np.array([x[j],y[j],z[j]])
            d = min_distance(ri, rj, Ls)
            ### full in
            if d + a[i] < r_cut[i] and i != j:#r_cut[i] - a[i]:
                #print(i,j,d)
                ql[i,j] = 1/d
                tmp.append(j)
        ii.append(tmp)
        n_neighbours[i] = len(tmp)
            
    #print(np.sort(ql)[:,-2:])
    ql = np.sort(ql)[:,-N_N:]
    return ql,n_neighbours

### Volume Calculation Functions ###

def partial_v(R, r, d):
    #calculate the volume of a truncated sphere
    #more info: https://mathworld.wolfram.com/Sphere-SphereIntersection.html
    return np.pi*(R+r-d)**2*(d**2+2*d*r-3*r**2+2*d*R+6*r*R-3*R**2)/12./d

#### This functions returns the local volume fraction around all particles
#### x,y,z and a are the arrays with coordinates and radii of all particles
#### Np is the number of particles
#### r_cut is the cut-off
def cal_v_local(x,y,z,r_cut,Np,a):
    """
    Function to return local volume fraction around all particles.
    x, y, z = Array of coords of all particles
    a = array of radii of particles
    r_cut = cutoff
    Np = number of particles
    """
    v_local = np.zeros(Np) #arr to store values
    for i in range(Np): #over all points
        ri = np.array([x[i],y[i],z[i]])
        vp = 0.
        for j in range(Np):
            rj = np.array([x[j],y[j],z[j]])
            d = min_distance(ri, rj, Ls)
            ### circles fully encompassed by cut-off radius
            if d + a[i] < r_cut[i]:#r_cut[i] - a[i]:
                #print(i,j,d)
                vp = vp + 4.*np.pi*a[i]**3/3.
            ### partially enclosed
            elif d - a[i] < r_cut[i]:# + a[i]:
                vp = vp + partial_v(r_cut[i],a[j],d) #truncated sphere v
                #print(partial_v(a[i],a[j],d))
                #print(i,j,d,'**')
        v_cut = (4./3.)*np.pi*r_cut[i]**3
        v_local[i] = vp/v_cut
    
    return v_local

### Diffusion Calculations ###
def cal_D0():
    ### Calculate D0, used to normalise all Ds
    KT = 4.11e-21
    R = 0.5e-10 ##Ams
    mu = 0.00089 ##Pa.s
    return (KT/(6.*np.pi*mu*R))


### Data Gathering ###


### Feature Extraction Functions###


def cal_sp(x,y,z,r_cut,ls,Np,a):
    """
    Function to calculate bond order of particles
    """
    n_neighbours = np.zeros(Np)
######### Getting the neighbours list
    ii = []
    #r_cut = 6*a
    for i in range(Np):
        ri = np.array([x[i],y[i],z[i]])
        tmp = []
        for j in range(Np):
            rj = np.array([x[j],y[j],z[j]])
            d = min_distance(ri, rj, Ls)
            ### full in
            if d + a[i] < r_cut[i] and i != j:#r_cut[i] - a[i]:
                #print(i,j,d)
                tmp.append(j)
        ii.append(tmp)
        n_neighbours[i] = len(tmp)
    #ii
    ##### Calculating theta and phi for all pairs of neighbours
    thetas = []
    phis = []

    i = 0
    for jj in ii:
        ri = np.array([x[i],y[i],z[i]])
        tmp_p = []
        tmp_t = []
        for j in jj:
            rj = np.array([x[j],y[j],z[j]])
            r_vec = vec_r(ri, rj, Ls)
            r = min_distance(ri, rj, Ls)
            theta = np.arctan2(r_vec[1],r_vec[0])+np.pi
            phi = np.arccos(r_vec[2]/r)
            tmp_t.append(theta)
            tmp_p.append(phi)
    #             #print(theta*180/np.pi,phi*180/np.pi)
    #             sph_harm(m, l, theta, phi)
        thetas.append(tmp_t)
        phis.append(tmp_p)
        i+=1
        

    #ls = [1,2,3,4,5,7,8]
    ql = np.zeros((Np,len(ls)))

    ll=0
    ##### loop over l
    for l in ls:
        # print(len(ls))
        # print(l)
        tmp_qlm = []
        ###### getting q_{lm}(i)
        for i, jj in enumerate(ii):
            tmp_qlm_i = []
            for m in range(-l,l+1):
                # print(l,m)
                tmp_q = 0
                #print(jj)
                for j, k in enumerate(jj):
                    tmp_q += sph_harm(m, l, thetas[i][j], phis[i][j])
                    #print(jj[j])
                tmp_qlm_i.append(tmp_q/len(jj))
            tmp_qlm.append(tmp_qlm_i)

        ##### getting the q_{lm}(i) average (this includes q_i and the neighbours of i) 
        tmp_qlm_av = []
        for i, jj in enumerate(ii):
            tmp_qlm_i = []
            for m in range(2*l+1):
                tmp_q = 0
                for j, k in enumerate(jj):
                    #print(i,k,m,jj)
                    tmp_q += tmp_qlm[k][m]
                tmp_q+= tmp_qlm[i][m]#### adding q_lm of itself
                tmp_qlm_i.append(tmp_q/(len(jj)+1))
                #print(i,m)
            tmp_qlm_av.append(tmp_qlm_i)

        ###### calculating q_l(i) 
        for i in range(Np):
            tmp_q = 0
            for m in range(2*l+1):
                tmp_q += np.absolute(tmp_qlm_av[i][m])

            #print(i,ll)
            ql[i,ll] = 4.*np.pi*tmp_q/(2*l+1)
        ll+=1  

    return ql,n_neighbours

def get_data2(files,f_r_cut1,f_r_cut2,l_s,l_s_names, bo, bo_names, N_N):

    #### creating a dataframe with all the data
    df = pd.DataFrame([], columns=l_s_names+bo_names)
    df['vol'] = []
    df['n_neighbours'] = []
    df['Ds'] = []

    dire = "data/"+files[0][5:14] #### THIS IS A QUICK FIX

    for f in tqdm_notebook(files):
        ######## Getting the seed used for the file name
        ######## File names start contain a long number and this is
        ######## the seed used for simulating that snap-shot
        #print(f)
        
        seed_tmp = f.split("Ds_")
        seed = seed_tmp[1].split(".dat")[0]
        fop_str = dire+seed+'.str'###File with positions
        fop_Ds = dire+'Ds_'+seed+'.dat'####File with the Ds
        #print(fop_str,fop_Ds)

        ######## Reading the x,y,z file and getting the local volume
        data = np.genfromtxt(fop_str,skip_header=0)        
        x = data[:,3]
        y = data[:,4]
        z = data[:,5]
        a = data[:,6] ###
        Rs = data[:,1] ###
        Np = len(x)
        #print(Np)
        #r_cut = 4*a
        r_cut = f_r_cut1*a
        v_local = cal_v_local(x,y,z,r_cut,Np,a)
        #print(v_local[:10])
        # print('seed=',seed,'Mean local vol:',np.mean(v_local))

        ######## Reading the Ds file and getting the distances
        data = np.genfromtxt(fop_Ds,skip_header=0,skip_footer=0)
        #data.shape
        #r_cut = 6*a
        r_cut = f_r_cut2*a
        bond_order, _ = cal_sp(x,y,z,r_cut,bo,Np,a)##cal_sp gets bond order 
        ql,ns = cal_dis(x,y,z,r_cut,l_s,Np,a,N_N)##cal_dis gets the distance to the l_s 
                                             ## closest neighbours.
                                             ## In the function check if you are getting d or 1/d

        ######## Calculating the average of Dx,Dy and Dz
        D_av = np.zeros(Np)#data[0::3,0]
        for j in range(0,3):
            D_av += data[j::3,0]
        D_av = D_av/3.

        D_av = np.array(D_av)
        D_av = D_av/cal_D0()

        ########
        df2 = pd.DataFrame(np.concatenate((ql, bond_order), axis=1), columns=l_s_names+bo_names)
        # df2 = pd.DataFrame(ql, columns=l_s_names)
        df2['vol'] = v_local
        df2['n_neighbours'] = ns
        df2['Ds'] = D_av
        
        # df3 = pd.DataFrame(bond_order, columns=bo_names)
        
        df = pd.concat([df,df2],ignore_index=True)



        ####### Plotting to check the local volume calculation
        # plt.plot(v_local,D_av,'.')

    # plt.plot(p_snap,D_snap,'-o')
    # plt.errorbar(p_snap,D_snap,stdD_snap)
    print(np.shape(ql))
    print(np.shape(bond_order))

    return df

def load_data(dires, l_s, l_s_names, bo, bo_names, f_r_cut_vol, f_r_cut_sp, N_neighbours):
    path = "data/"

    df = pd.DataFrame()
    
    for dire in dires:
        #### getting the size of the simulation box
        data = np.genfromtxt(path+dire+'nums.dat',skip_header=0,max_rows=1)
        L = data
        # print(L)
        global Ls #declare as global variable right now as functions need to use but not taken as arg
        Ls = np.array([L,L,L])
    
        ####### Getting the list of files names (each file contains a snapshot)
        files = glob.glob(path+dire+"Ds_*")
        # print(files)
    
        ### creating the dataframe with all the data
        # print(bo)
        tmp_df = get_data2(files,f_r_cut_vol,f_r_cut_sp,l_s,l_s_names, bo, bo_names, N_neighbours)
        # df = df.append(tmp_df, ignore_index = True, sort=False)
        df = pd.concat([df, tmp_df], ignore_index=True, sort=False)

        return tmp_df