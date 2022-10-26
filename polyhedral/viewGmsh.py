import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
plt.rcParams.update({'font.size':22})

mcc = cm.coolwarm(np.linspace(0,1,10))

# vis point
def visPoint(ps,lc=mcc[2],m='o',mfc='none',ms=3):
    plt.plot([ps[0]],[ps[1]],m,c=lc,mfc=mfc,markersize=ms)

# vis points
def visPoints(ps,lc=mcc[2],m='o',mfc='none',ms=3):
    plt.plot([ps[:,0]],[ps[:,1]],m,c=lc,mfc=mfc,markersize=ms)

# vis polygon element
def visPoly(vs,lc=mcc[2],lw=2,ml='-'):
    plt.plot([*vs[:,0],vs[0,0]],
             [*vs[:,1],vs[0,1]],ml,c=lc)

# vis 3-d patch
def visPatch(ax, x, y, z, v=0, vmin=0, vmax=100, cmap_name='coolwarm'):
    v = z[0]
    # Get colormap by name
    cmap = cm.get_cmap(cmap_name)
    # Normalize value and get color
    c = cmap(colors.Normalize(vmin, vmax)(v))
    # Create PolyCollection from coords
    pc = Poly3DCollection([list(zip(x,y,z))])
    # Set facecolor to mapped value
    pc.set_facecolor(c)
    # Set edgecolor to black
    pc.set_edgecolor('k')
    # Adr PolyCollection to axes
    ax.add_collection3d(pc)
    return pc

def loadGmsh(fname,readBoundaryTags=False):
    f = open(fname,'r')

    # reading dimension
    line = f.readline().strip()
    if len(line) < 6:
        print('error: expected to read \'NDIME=\'')
        return -1

    if line[:6] != 'NDIME=':
        print('error: expected to read \'NDIME=\'')
        return -1

    sline = line.split(' ')
    dim = int(sline[-1])

    # reading number of elements
    line = f.readline().strip()
    if len(line) < 6:
        print('error: expected to read \'NELEM=\'')
        return -1

    if line[:6] != 'NELEM=':
        print('error: expected to read \'NELEM=\'')
        return -1

    sline = line.split(' ')
    nElems = int(sline[-1])

    if dim == 2:
        # assuming only quad, tri for 2d
        connectivity = np.zeros((5,nElems),dtype=np.int64)
    elif dim == 3:
        # assuming only tetrahedral, hexahedral for 2d
        connectivity = np.zeros((9,nElems),dtype=np.int64)
    else:
        print('error: only 2-d and 3-d supported')
        return -1

    idx = 0
    maxVerts = 4
    while not ('=' in (line := f.readline())):
        if idx >= nElems:
            print('error: read more elements than expected')
            return -1

        sline = line.strip().split(' ')
        n = len(sline)
        while n-2 > maxVerts:
            connectivity = np.vstack((connectivity,
                    np.zeros((1,nElems),dtype=np.int64)))
            maxVerts += 1
            print('extended')
        connectivity[:n-1,idx] = [int(v) for v in sline[:n-1]]
        idx += 1

    elemTypes = np.unique(connectivity[0,:])
    if elemTypes.shape[0] == 1:
        # if we read only triangles, trim connectivity
        if dim == 2 and elemTypes[0] == 5:
            # trim from (5,nElems) to (4,nElems)
            connectivity = connectivity[:-1,:]
        # if we read only tetrahedrals, trim connectivity
        if dim == 3 and elemTypes[0] == 10:
            # trim from (9,nElems) to (5,nElems)
            connectivity = connectivity[:-4,:]

    connectivity = connectivity.T

    # now read in the vertices
    # have already read the next line
    if len(line) < 6:
        print('error: expected to read \'NPOIN=\'')
        return -1

    if line[:6] != 'NPOIN=':
        print('error: expected to read \'NPOIN=\'')
        return -1

    sline = line.split(' ')
    nPoints = int(sline[-1])

    points = np.zeros((nPoints,dim))

    idx = 0
    while not ('=' in (line := f.readline())):
        sline = line.strip().split(' ')
        points[idx,:] = [float(v) for v in sline[:-1]]
        idx += 1

    if not readBoundaryTags:
        return dim,nElems,nPoints,connectivity,points,0,[],[],[]

    # need to make sure there is boundary condition data if requested
    # line already read
    if not line:
        print('error: tried to read boundary tags but no data found')
        return -1

    # now read in the number of boundary tags
    if len(line) < 6:
        print('error: expected to read \'NMARK=\'')
        return -1

    if line[:6] != 'NMARK=':
        print('error: expected to read \'NMARK=\'')
        return -1

    sline = line.split(' ')
    nMarks = int(sline[-1])

    marks = []
    markNames = []
    markNumFaces = []

    axis1 = 2+1   # all 2-d boundaries will be lines
    if dim == 3:
        axis1 = 4+1

    for imark in range(nMarks):
        # now read in the boundary tag name
        line = f.readline()
        if len(line) < 11:
            print('error: expected to read \'MARKER_TAG=\'')
            return -1

        if line[:11] != 'MARKER_TAG=':
            print('error: expected to read \'MARKER_TAG=\'')
            return -1

        sline = line.strip().split(' ')
        markNames.append(sline[-1])

        # now read in the number of boundary faces
        line = f.readline()
        if len(line) < 13:
            print('error: expected to read \'MARKER_ELEMS=\'')
            return -1

        if line[:13] != 'MARKER_ELEMS=':
            print('error: expected to read \'MARKER_ELEMS=\'')
            return -1

        sline = line.strip().split(' ')
        numBFaces = int(sline[-1])
        markNumFaces.append(numBFaces)

        mark = np.zeros((numBFaces,axis1),dtype=np.int32)

        # now read the faces
        for iface in range(numBFaces):
            sline = f.readline().strip().split(' ')
            if dim == 2:
                if sline[0] != '3':
                    print('error: boundary wasn\'t a line in 2-d')
                    return -1
            if dim == 3:
                if sline[0] not in ['5','9']:
                    print('error: only quad and tri boundary faces '+
                            'supported in 3-d for now')
                    return -1

            mark[iface,:] = [int(v) for v in sline]

        marks.append(mark)

    # mask end of each connectivity array if invalid data
    # n = connectivity.shape[1]
    # for i in range(nElems):
        # elem = connectivity[i,:]
        # if elem[0] == 7:
            # for m in range(n-1,0,-1):
                # if connectivity[i][m] == 0:
                    # connectivity[i][m] = np.NaN
                # else:
                    # break

    return dim,nElems,nPoints,connectivity,points,nMarks,marks,markNames,markNumFaces





# fname = 'poly.su2'
fname = '../tdisk_mid_bc.su2'
# fname = 'tdisk_poly.su2'

res = loadGmsh(fname,True)

if res == -1:
    print('error: load Gmsh failed')
    exit()

dim,nElems,nPoints,connectivity,points,nMarks,marks,markNames,markNumFaces = res

fig = plt.figure()
if dim == 3:
    ax = plt.axes(projection='3d')
    print('error: 3d plotting not supported')
    exit()

for i in range(nElems):
    elem = connectivity[i,:]
    n = elem.shape[0] - 1
    #
    if elem[0] == 5:
        n = 3
    elif elem[0] == 9:
        n = 4
    elif elem[0] == 3:
        n = 2
    elif elem[0] == 7:
        n = connectivity.shape[1]
        for m in range(n-1,0,-1):
            if connectivity[i][m] != 0:
                n = m
                break
    #
    visPoly(points[elem[1:n+1]])

plt.show(block=False)
