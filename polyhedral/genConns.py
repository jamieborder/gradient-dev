import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
plt.rcParams.update({'font.size':22})

from halfedge import *

mcc = cm.coolwarm(np.linspace(0,1,10))

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
            connectivity = np.hstack((connectivity,
                    np.zeros((1,nElems))))
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


    return dim,nElems,nPoints,connectivity,points,nMarks,marks,markNames,markNumFaces





# if __name__ == "__main__":

# fname = 'tdisk.su2'
# fname = 'qdisk.su2'
# fname = 'tdisk_med.su2'
# fname = 'tdisk_med_bc.su2'
# fname = 'qdisk_med.su2'
# fname = 'tdisk_fine.su2'
# fname = 'qdisk_fine.su2'

# fname = 'tdisk_coarse_bc.su2'
# fname = 'tdisk_med_bc.su2'
# fname = 'tdisk_fine_bc.su2'
# fname = '../tdisk_mid_bc.su2'        # <--
# fname = '../tdisk_verymid_bc.su2'
# fname = 'tdisk_veryfine_bc.su2'

# fname = 'qdisk_coarse_bc.su2'
# fname = 'qdisk_med_bc.su2'
# fname = 'qdisk_fine_bc.su2'
# fname = 'qdisk_veryfine_bc.su2'

fname = 'sTetMesh.su2'
# fname = '../qdisk_veryfine_bc.su2'

res = loadGmsh(fname,True)

if res == -1:
    print('error: load Gmsh failed')
    exit()

dim,nElems,nPoints,connectivity,points,nMarks,marks,markNames,markNumFaces = res


## convert to format as read from vtu file
offsets = np.arange(1,connectivity.shape[0]+1)*3
conn = connectivity[:,1:].flatten()
pts = np.hstack((points,np.zeros((points.shape[0],1))))
ncells = nElems

# loop over offsets and access connectivity
hes = []

a = 0
for b in offsets:
    # offsets give a range of connectivity values (point ids)
    #  for a face (assumed ordered)
    for i,pid in enumerate(conn[a:b]):
        hes.append(HalfEdge(pts[pid], pid,
                conn[a+(i+1)%(b-a)]))

    # we know it is a triangle, so make connections
    hes[-3].next = hes[-2]
    hes[-2].next = hes[-1]
    hes[-1].next = hes[-3]

    a = b

# making twin connection
# dups = 0
for i,hei in enumerate(hes):
    for j,hej in enumerate(hes):
        if (i == j):
            continue

        # if hei.p0 == hej.p0 and hei.p1 == hej.p1:
            # dups += 1
        if hej.twin is not None:
            continue

        if hei.p0 == hej.p1 and hei.p1 == hej.p0:
            hes[i].twin = hes[j]
            hes[j].twin = hes[i]
            break

# making faces
faces = []
nextId = 0
for i,he in enumerate(hes):
    if he.face is None:
        faces.append(Face(nextId))
        nextId += 1
        hi = he
        faces[-1].hes.append(hi)
        hi.face = faces[-1]
        hi = hi.next
        while hi != he:
            faces[-1].hes.append(hi)
            hi.face = faces[-1]
            hi = hi.next

dups = 0
for i,hei in enumerate(hes):
    for j,hej in enumerate(hes):
        if (i == j):
            continue
        if hei.p0 == hej.p0 and hei.p1 == hej.p1:
            dups += 1
        if hej.twin is not None:
            continue
        if hei.p0 == hej.p1 and hei.p1 == hej.p0:
            hes[i].twin = hes[j]
            hes[j].twin = hes[i]
            break


rows = []
cols = []
for i in range(len(faces)):
    rows.append(faces[i].id)
    cols.append(faces[i].id)
    for he in faces[i].hes:
        if he.twin is None:
            continue
        else:
            rows.append(he.face.id)
            cols.append(he.twin.face.id)

f = open('mat2.dat','w')
for i in range(len(rows)):
    f.write('{:d} {:d} 1\n'.format(rows[i],cols[i]))

f.close()

