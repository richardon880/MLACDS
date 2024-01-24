import numpy as np

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
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta) #if diff between points is greater than half of the box dimensions then get distance through boundary
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





