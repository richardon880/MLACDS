import numpy as np
import glob
from scipy.special import sph_harm
from scipy.interpolate import CubicSpline
import pandas as pd

from tqdm.notebook import tqdm_notebook

np.seterr(divide='ignore')

#volume fraction formula
def getvolfrac(sim_diameter, N, L):
    r = (sim_diameter/20)**(1/3)
    NperV = N/(L**3)
    phi = 4/3 * np.pi * r**3 * NperV
    return phi

def findparams(volfrac, N=None, L=None):
    sim_diameter = 2.5
    r3 = (sim_diameter/20)
    
    if N == None and L == None:
        return print("Please provide at least one of N or L.")
    elif N == None and L != None:
        V = L**3
        return ((volfrac*V)/r3) * (3/(4*np.pi))
    elif L == None and N != None:
        return (((4*np.pi)/3) * r3 * (N/volfrac)) ** (1/3)


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
# def min_distance(x0, x1, dimensions):
#     """
#     Function Calculates Distance Between 2 points
#     when using periodic boundary conditions.
#     x0 = array like len 3 (x,y,z position)
#     x1 = as above, dist will be calculated between this and x0
#     dimensions = array like len 3 (x,y,z len of box)
#     """
#     delta = np.absolute(x0 - x1) # diff between points
#     delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta) #if diff between points is greater than half of the box dimensions then get distance through  
#     return np.sqrt((delta**2).sum(axis=-1))

def min_distance(x0, x1, dimensions, angular_info=False):
    """
    Function Calculates Distance Between 2 points
    when using periodic boundary conditions.
    x0 = array like len 3 (x,y,z position)
    x1 = as above, dist will be calculated between this and x0
    dimensions = array like len 3 (x,y,z len of box)
    """
    delta = np.absolute(x0 - x1) # diff between points
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta) #if diff between points is greater than half of the box dimensions then get distance through
    R = np.sqrt((delta**2).sum(axis=-1))
    if angular_info == False:
        return R
    else:
        angx = delta[0]/(R**2)
        angy = delta[1]/(R**2)
        angz = delta[2]/(R**2)
        return R, angx, angy, angz


def vec_r(x0, x1, dimensions):
    """
    Similar to min distance, used in bond order calculations
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
def cal_dis(x,y,z,r_cut,ls,Np,a,N_N, inverse, angular_info=False):
    """
    x,y,z = float, x y and z position of particle.
    r_cut = radius of sphere to search for nearest neighbours in
    ls = arraylike, shape of 3d box which bounds simulation
    Np = Number of neighbours to stop search at (may be less than in r_cut sphere)
    a = array of radii of particles
    N_N = Number nearest neighbours to return ?
    """
    n_neighbours = np.zeros(Np) #array to store number of neighbours in for each particle 
    ql = np.zeros((Np,Np)) #array to store the dist to each neighbour for each particle
    ######### Getting the neighbours list
    ii = []
    #loop through the particles
    if angular_info == False:
        for i in range(Np): #the ith particle
            ri = np.array([x[i],y[i],z[i]]) #position of ith particle
            tmp = [] #temp list to store neighbours of current particle
            for j in range(Np): #the jth particle
                rj = np.array([x[j],y[j],z[j]]) #position of the jth particle
                #get the min distance between ith and jth particle taking into account periodic boundary conditions
                d = min_distance(ri, rj, Ls)
    
                if d + a[i] < r_cut[i] and i != j: #if dist from i to j is within r cut and are 2 diff particles
        
                    ql[i,j] = d
                    tmp.append(j) #add j to tmp list of neighbours within r cut
            ii.append(tmp) #add the list of neighbours to the list ii
            n_neighbours[i] = len(tmp) #add the number of neighbours within r cut to n_neighbours
        if inverse == True:
            ql = -np.sort(-1/ql)[:,:N_N]
        elif inverse == False:
            ql = np.sort(ql)[:,:N_N]
            
    elif angular_info == True:
        for i in range(Np): #the ith particle
            ri = np.array([x[i],y[i],z[i]]) #position of ith particle
            tmp = [] #temp list to store neighbours of current particle
            for j in range(Np): #the jth particle
                rj = np.array([x[j],y[j],z[j]]) #position of the jth particle
                #get the min distance between ith and jth particle taking into account periodic boundary conditions
                d, ang_info = min_distance(ri, rj, Ls, angular_info=True)
    
                if d + a[i] < r_cut[i] and i != j: #if dist from i to j is within r cut and are 2 diff particles
        
                    ql[i,j] = (d, ang_info)
                    tmp.append(j) #add j to tmp list of neighbours within r cut
            ii.append(tmp) #add the list of neighbours to the list ii
            n_neighbours[i] = len(tmp) #add the number of neighbours within r cut to n_neighbours
        if inverse == True:
            ql = -np.sort(-1/ql[0])[:,:N_N]
        elif inverse == False:
            ql = np.sort(ql[0])[:,:N_N]
            
    return ql,n_neighbours

### Volume Calculation Functions ###

def partial_v(R, r, d):
    """
    R: float, radius of sphere centered at 0,0,0
    r: float, radius of sphere centered at d,0,0
    d: x coord of radius sphere r.
    returns: volume of the intersection of sphere with radius R and r
    """
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
    r_cut = cutoff
    Np = number of particles
    a = array of radii of particles
    returns: the local volume fraction phi, of all particles
    """
    v_local = np.zeros(Np) #arr to store values
    for i in range(Np): #over all points
        ri = np.array([x[i],y[i],z[i]]) #get position of ith point
        vp = 0. #init volume fraction phi to 0
        for j in range(Np): 
            rj = np.array([x[j],y[j],z[j]]) #position of jth particle
            d = min_distance(ri, rj, Ls) #get min distance between i and jth particle with periodic boundary
            ### circles fully encompassed by cut-off radius
            if d + a[i] < r_cut[i]: #if distance and radius is less than cutoff radius the particle is inside by rcut
                #print(i,j,d)
                vp = vp + 4.*np.pi*a[i]**3/3. #calculate and update vp
            ### partially enclosed
            elif d - a[i] < r_cut[i]: #if distance less particle radius is less than r_cut then is partially indside
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

    Returns: Bond order and nearest neighbours for each particle
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

def get_data2(files,dire,f_r_cut1,f_r_cut2,l_s,l_s_names, bo, bo_names, N_N, angular_info=False):
    N_N += 1
    l_s_names_copy = list(l_s_names)
    l_s_names_copy.insert(0, "drop1")
    l_s_names_copy.insert(N_N, "drop2")
    
    #### creating a dataframe with all the data
    df = pd.DataFrame([], columns=l_s_names_copy+bo_names)
    # print(l_s_names)
    df['vol'] = []
    df['n_neighbours'] = []
    df['Ds'] = []

    dire = "data/"+dire #[0][5:15] #### THIS IS A QUICK FIX

    for f in tqdm_notebook(files):
        ######## Getting the seed used for the file name
        ######## File names start contain a long number and this is
        ######## the seed used for simulating that snap-shot
        #print(f)
        
        seed_tmp = f.split("Ds_")
        seed = seed_tmp[1].split(".dat")[0]
        fop_str = dire+seed+'.str'###File with positions
        fop_Ds = dire+'Ds_'+seed+'.dat'####File with the Ds
        # print(fop_str,fop_Ds)

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
        if angular_info == False:
            ql,ns = cal_dis(x,y,z,r_cut,l_s,Np,a,N_N, inverse=True)##cal_dis gets the distance to the l_s closest neighbours.
        elif angular_info == True:
            ql,ns = cal_dis(x,y,z,r_cut,l_s,Np,a,N_N, inverse=True)##cal_dis gets the distance to the l_s closest neighbours.
            
        ql_1,ns = cal_dis(x,y,z,r_cut,l_s,Np,a,N_N, inverse=False)##cal_dis gets the distance to the l_s 


        ######## Calculating the average of Dx,Dy and Dz
        D_av = np.zeros(Np)#data[0::3,0]
        for j in range(0,3):
            D_av += data[j::3,0]
        D_av = D_av/3.

        D_av = np.array(D_av)
        D_av = D_av/cal_D0()

        ########
        df2 = pd.DataFrame(np.concatenate((ql, ql_1, bond_order), axis=1), columns=l_s_names_copy+bo_names)
        # df2 = pd.DataFrame(ql, columns=l_s_names)
        df2['vol'] = v_local
        df2['n_neighbours'] = ns
        df2['Ds'] = D_av
        
        # df3 = pd.DataFrame(bond_order, columns=bo_names)
        
        df = pd.concat([df,df2],ignore_index=True)
    print(df.head())
    # df = df.drop(["drop1", "drop2"], axis=1)


        ####### Plotting to check the local volume calculation
        # plt.plot(v_local,D_av,'.')

    # plt.plot(p_snap,D_snap,'-o')
    # plt.errorbar(p_snap,D_snap,stdD_snap)
    # print(np.shape(ql))
    # print(np.shape(bond_order))

    return df

def load_data(dires, l_s, l_s_names, bo, bo_names, f_r_cut_vol, f_r_cut_sp, N_neighbours, use_bond_order=True):
    # reset_l_s_names = list(l_s_names)
    # reset_N_neighbours = N_neighbours
    path = "data/"

    # df = pd.DataFrame(columns=l_s_names+bo_names+["vol", "n_neighbours", "Ds"])
    df = pd.DataFrame()
    
    for dire in dires:
        print(l_s_names)

        # print("altered :",l_s_names)
        # print("original:",reset_l_s_names)
        # print("altered NN :",N_neighbours)
        # print("original NN:",reset_N_neighbours)
        # l_s_names = reset_l_s_names
        # N_neighbours = reset_N_neighbours
        print(dire)
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
        if use_bond_order==True:
            tmp_df = get_data2(files,dire,f_r_cut_vol,f_r_cut_sp,l_s,l_s_names, bo, bo_names, N_neighbours)
        else: 
            tmp_df = get_data2_no_bond_order_or_dist(files,dire,f_r_cut_vol,f_r_cut_sp,l_s,l_s_names, bo, bo_names, N_neighbours)
        # df = df.append(tmp_df, ignore_index = True, sort=False)
        df = pd.concat([df, tmp_df], ignore_index=True, sort=False)

    print(df)
    if use_bond_order == True:
        df = df.drop(["drop1", "drop2"], axis=1)
    else:
        df = df.drop(["drop1"], axis=1)
    return df

"""
def get_data2_no_bond_order_or_dist(files,dire,f_r_cut1,f_r_cut2,l_s,l_s_names, bo, bo_names, N_N):
    
    #### creating a dataframe with all the data
    df = pd.DataFrame([], columns=l_s_names)
    df['vol'] = []
    df['n_neighbours'] = []
    df['Ds'] = []
    
    dire = "data/"+dire #[0][5:15] #### THIS IS A QUICK FIX
    
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

        ql,ns = cal_dis(x,y,z,r_cut,l_s,Np,a,N_N, inverse=True)##cal_dis gets the distance to the l_s closest neighbours.

        ######## Calculating the average of Dx,Dy and Dz
        D_av = np.zeros(Np)#data[0::3,0]
        for j in range(0,3):
            D_av += data[j::3,0]
        D_av = D_av/3.

        D_av = np.array(D_av)
        D_av = D_av/cal_D0()

        ########
        df2 = pd.DataFrame(ql, columns=l_s_names)
        df2['vol'] = v_local
        df2['n_neighbours'] = ns
        df2['Ds'] = D_av
        df = pd.concat([df,df2],ignore_index=True)

    return df
"""

def get_data2_no_bond_order_or_dist(files,dire,f_r_cut1,f_r_cut2,l_s,l_s_names, bo, bo_names, N_N):
    N_N += 1
    l_s_names_copy = list(l_s_names)
    l_s_names_copy.insert(0, "drop1")
    
    #### creating a dataframe with all the data
    df = pd.DataFrame([], columns=l_s_names)
    df['vol'] = []
    df['n_neighbours'] = []
    df['Ds'] = []

    dire = "data/"+dire #[0][5:15] #### THIS IS A QUICK FIX

    for f in tqdm_notebook(files):
        ######## Getting the seed used for the file name
        ######## File names start contain a long number and this is
        ######## the seed used for simulating that snap-shot
        #print(f)
        
        seed_tmp = f.split("Ds_")
        seed = seed_tmp[1].split(".dat")[0]
        fop_str = dire+seed+'.str'###File with positions
        fop_Ds = dire+'Ds_'+seed+'.dat'####File with the Ds
        # print(fop_str,fop_Ds)

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
        ql,ns = cal_dis(x,y,z,r_cut,l_s,Np,a,N_N, inverse=True)##cal_dis gets the distance to the l_s closest neighbours.

        ######## Calculating the average of Dx,Dy and Dz
        D_av = np.zeros(Np)#data[0::3,0]
        for j in range(0,3):
            D_av += data[j::3,0]
        D_av = D_av/3.

        D_av = np.array(D_av)
        D_av = D_av/cal_D0()

        ########
        # print(ql)
        # print(l_s_names_copy)
        df2 = pd.DataFrame(ql, columns=l_s_names_copy)
        # df2 = pd.DataFrame(ql, columns=l_s_names)
        df2['vol'] = v_local
        df2['n_neighbours'] = ns
        df2['Ds'] = D_av
        
        # df3 = pd.DataFrame(bond_order, columns=bo_names)
        
        df = pd.concat([df,df2],ignore_index=True)
    # df = df.drop(["drop1", "drop2"], axis=1)


        ####### Plotting to check the local volume calculation
        # plt.plot(v_local,D_av,'.')

    # plt.plot(p_snap,D_snap,'-o')
    # plt.errorbar(p_snap,D_snap,stdD_snap)
    # print(np.shape(ql))
    # print(np.shape(bond_order))

    return df

def make_dict(num_neighbours, dist, dist_inv, bond_order):
    if dist == True:
        dist_names = [f"l_{i}" for i in range(0,num_neighbours)]
    if dist_inv == True:
        dist_inv_names = [f"l_{i}_inv" for i in range(0,num_neighbours)]
    if bond_order == True:
        bo_names = [f"bo_{i}" for i in range(0,num_neighbours)]
    keys = dist_names + dist_inv_names + bo_names + ["vol", "n_neighbours", "Ds"]
    values = [[] for i in range(len(keys))]
    colname_dict = dict(zip(keys, values))
    return colname_dict

def get_data(path, dires, r_cut1, r_cut2, num_neighbours, dist=True, dist_inv=True, angular=True, bond_order=True):
    dires = [path+dire for dire in dires]
    neighbour_index = list(range(0,num_neighbours))
    dataframe_dict = make_dict(num_neighbours, dist, dist_inv, bond_order)
    colnames = list(dataframe_dict.keys())
    for dire in dires:
        #call function to get filenames from dires and box size for later
        files = get_filenames(dire)
        global Ls
        Ls = getboxsize(dire)
    #call function to iterate over files get position info, ds info
        for f in tqdm_notebook(files):
            data_tmp = np.array([])
            seed_tmp = f.split("Ds_")
            seed = seed_tmp[1].split(".dat")[0]
            fop_str = dire+seed+'.str'
            fop_Ds = dire+'Ds_'+seed+'.dat'
            position_data = np.genfromtxt(fop_str, skip_header=0)  
            diffusion_data = np.genfromtxt(fop_Ds, skip_header=0, skip_footer=0)
            x = position_data[:,3]
            y = position_data[:,4]
            z = position_data[:,5]
            a = position_data[:,6]
            Np = len(x)
            #call function to calculate vlocal + other features
            D_av = avg_diffusion_data(diffusion_data, Np)
            local_vol_cut = r_cut1*a
            feature_cut = r_cut2*a
            v_local = cal_v_local(x,y,z,local_vol_cut,Np,a)
            if bond_order == True:
                bond_order_data, _ = cal_sp(x,y,z,feature_cut,neighbour_index,Np,a)
            else:
                bond_order_data = np.array([])
    
            if dist == True:
                nearest_neighbour_dists, num_nearby = cal_dis(x,y,z,feature_cut,neighbour_index,Np,a,num_neighbours+1, inverse=False)
            else:
                nearest_neighbour_dists = np.array([])
    
            if dist_inv == True:
                nearest_neighbour_inv_dists, num_nearby = cal_dis(x,y,z,feature_cut,neighbour_index,Np,a,num_neighbours+1, inverse=True)
            else:
                nearest_neighbour_inv_dists = np.array([])
            data_tmp = np.vstack((nearest_neighbour_dists.T[1:],
                                 nearest_neighbour_inv_dists.T[1:], bond_order_data.T, 
                                 v_local, np.full(len(v_local), num_nearby), D_av))
            for i in range(len(colnames)):
                dataframe_dict[colnames[i]] = np.append(dataframe_dict[colnames[i]], data_tmp[i])

    df = pd.DataFrame(dataframe_dict)
    return dataframe_dict



def get_filenames(dire):
    files = glob.glob(dire+"Ds_*")
    return files

def getboxsize(dire):
    L = np.genfromtxt(dire+'nums.dat',skip_header=0,max_rows=1)
    box_dims = np.array([L,L,L])
    return box_dims

def get_simulation_data(dire, files):
    for f in tqdm_notebook(files):
        seed_tmp = f.split("Ds_")
        seed = seed_tmp[1].split(".dat")[0]
        fop_str = dire+seed+'.str'
        fop_Ds = dire+'Ds_'+seed+'.dat'
        position_data = np.genfromtxt(fop_str, skip_header=0)  
        diffusion_data = np.genfromtxt(fop_Ds, skip_header=0, skip_footer=0)
        return position_data, diffusion_data
        
def avg_diffusion_data(data, Np):
    D_av = np.zeros(Np)
    for j in range(0,3):
        D_av += data[j::3,0]
    D_av = D_av/3.
    D_av = np.array(D_av)
    D_av = D_av/cal_D0()
    return D_av


