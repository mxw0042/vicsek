import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import MinMaxScaler
 
#size of box
L = 32.0
rho = 3.0
#number of particles (rho per unit of the box)
N = int(rho*L**2)
print("N",N)
 
#radius of influence
r0 = 1.0
#velocity
v0 = 1.0
#number of steps
iterations = 10000
#noise
eta = 0.5
 
#initialize positions and orientations
#(N,2)
pos = np.random.uniform(0,L,size=(N,2))
#(N)
orient = np.random.uniform(-np.pi, np.pi,size=N)
 
fig, ax= plt.subplots(figsize=(6,6))
 
#draw quivers
qv = ax.quiver(pos[:,0], pos[:,1], np.cos(orient), np.sin(orient), orient, clim=[-np.pi, np.pi])
 
def animate(i):
    # print(i)
    global orient
    tree = cKDTree(pos,boxsize=[L,L])
    # dist.col and dist.row contain indices of where there is non-zero data
    #(N, N)
    dist = tree.sparse_distance_matrix(tree, max_distance=r0,output_type='coo_matrix')
    # we evaluate a quantity for every column j
    # (# of non-zero values,)
    data = np.exp(orient[dist.col]*1j)
    # construct  a new sparse marix with entries in the same places ij of the dist matrix
    #(N, N)
    neigh = sparse.coo_matrix((data,(dist.row,dist.col)), shape=dist.get_shape())
    # and sum along the columns (sum over j)
    #(N,)
    S = np.squeeze(np.asarray(neigh.tocsr().sum(axis=1)))

    orient = np.angle(S)+eta*np.random.uniform(-np.pi, np.pi, size=N)

    scaled_orient=np.interp(orient, (orient.min(), orient.max()), (0, +1))
    # print(abs(sum(scaled_orient)/(N)))
    order=np.var(orient)
    print(abs(order-7.33438)/7.33438)

    # if i==499:
    #     return
 
    cos, sin= np.cos(orient), np.sin(orient)
    pos[:,0] += cos*v0
    pos[:,1] += sin*v0
 
    pos[pos>L] -= L
    pos[pos<0] += L
 
    qv.set_offsets(pos)
    qv.set_UVC(cos, sin,orient)
    return qv,
 
FuncAnimation(fig,animate,np.arange(0, 500),interval=1, blit=True)
plt.show()
