import numpy as np
import matplotlib
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

data=np.load('/users/nathd/pyGITRM/test_167196_bgp_mesh.npy', allow_pickle=True).tolist()
key_list=[k for k in data.keys() if k!='mesh']

#Mesh
mesh_array=np.array(data['mesh'])

plt.close('all')

#Fields
for k in key_list:
    
    if data.get(k) is None:
        data[k] = 0*data['KTEBS']
        print("This field has no data:",k)
        
    field_array=np.array(data[k])
    field_array[np.isnan(field_array)] = 0
    
    #Figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    patches = [matplotlib.patches.Polygon(v, closed=False) for v in mesh_array]
    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=1, edgecolor=None)
    p.set_array(field_array)
    ax.add_collection(p)
    ax.set_xlim([1, 2.5])  
    ax.set_ylim([-1.5, 1.5])
    plt.title(k)

    fig.colorbar(p)
    fig.show()
