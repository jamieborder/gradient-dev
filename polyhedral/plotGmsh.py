import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
plt.rcParams.update({'font.size':22})

from parula import *

mcc = cm.coolwarm(np.linspace(0,1,10))

visualise = False

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
            # print('extended')
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

fname = '../tdisk_mid_bc.su2'        # <--
# fname = 'tdisk_poly.su2'        # <--

# fname = '../tdisk_verymid_bc.su2'        # <--
# fname = 'tdisk_poly2.su2'        # <--

# fname = 'tdisk_verymid_bc.su2'
# fname = 'tdisk_veryfine_bc.su2'

# fname = 'qdisk_coarse_bc.su2'
# fname = 'qdisk_med_bc.su2'
# fname = 'qdisk_fine_bc.su2'
# fname = 'qdisk_veryfine_bc.su2'

# fname = 'sTetMesh.su2'
# fname = 'poly.su2'

res = loadGmsh(fname,True)

if res == -1:
    print('error: load Gmsh failed')
    exit()

dim,nElems,nPoints,connectivity,points,nMarks,marks,markNames,markNumFaces = res

if np.all(connectivity[:,0] == 3):
    tris = True
else:
    tris = False

'''
fig = plt.figure()
if dim == 3:
    ax = plt.axes(projection='3d')
    print('error: 3d plotting not supported')
    exit()

for i in range(nElems):
    elem = connectivity[i,:]
    n = elem.shape[0] - 1

    if elem[0] == 5:
        n = 3
    elif elem[0] == 9:
        n = 4
    elif elem[0] == 3:
        n = 2

    visPoly(points[elem[1:n+1]])
'''

faceWeightedCentroids = True
# faceWeightedCentroids = False

centres = np.zeros((nElems,2))
for i in range(nElems):
    elem = connectivity[i,:]
    n = elem.shape[0] - 1

    if elem[0] == 5:
        n = 3
    elif elem[0] == 9:
        n = 4
    elif elem[0] == 3:
        n = 2
    elif elem[0] == 7:
        for m in range(n,0,-1):
            if connectivity[i][m] != 0:
                n = m
                break
        elem = connectivity[i,:n+1]

    if faceWeightedCentroids:
        areaPxSum = 0.0
        areaPySum = 0.0
        areaSum = 0.0
        ps = points[elem[1:n+1]]
        for j in range(n):
            area = np.sum((ps[(j+1)%n] - ps[j])**2.)**0.5
            areaSum += area
            areaPxSum += (ps[j][0] + ps[(j+1)%n][0])/2.0 * area
            areaPySum += (ps[j][1] + ps[(j+1)%n][1])/2.0 * area

        centres[i,:] = areaPxSum / areaSum, areaPySum / areaSum
    else:
        centres[i,:] = np.sum(points[elem[1:n+1]],0) / n

'''
visPoints(centres,lc=mcc[0],m='x')

for i in range(nMarks):
    for j in range(markNumFaces[i]):
        visPoly(points[marks[i][j,1:]],lc='r')
'''


nFields = 3
fields = np.zeros((nElems,nFields))

# f = lambda x,y: np.sin(0.1 * np.abs(x * y)) * 4*np.exp(np.abs(x * y))
# dfdx = lambda x,y: (x*y**2*np.exp(np.abs(x*y))*(4*np.sin(0.1*np.abs(x*y))
        # + 0.4*np.cos(0.1*np.abs(x*y))))/np.abs(x*y)
# dfdy = lambda x,y: (y*x**2*np.exp(np.abs(x*y))*(4*np.sin(0.1*np.abs(x*y))
        # + 0.4*np.cos(0.1*np.abs(x*y))))/np.abs(x*y)

func=lambda x,y: 4.0*x + 5.0*x*x + 6.0*x**3 + 4.0*y + 5.0*y*y + 6.0*y**3 + x*x*y*y
dfdx=lambda x,y: 4.0 + 10.0*x + 18.0*x**2 + 2*x*y*y
dfdy=lambda x,y: 4.0 + 10.0*y + 18.0*y**2 + 2*x*x*y

# func = lambda x,y: 4.0*x + 5.0*x*x + 4.0*y + 5.0*y*y + 2*x*y
# dfdx = lambda x,y: 4.0 + 10.0*x + 2*y
# dfdy = lambda x,y: 4.0 + 10.0*y + 2*x

# func = lambda x,y: 4.0*x + 4.0*y + 8.0*x*y
# dfdx = lambda x,y: 4.0 + 8.0*y
# dfdy = lambda x,y: 4.0 + 8.0*x

for i in range(nElems):
    fields[i,0] = i
    fields[i,1] = i/100.
    fields[i,2] = func(*centres[i,:])

'''
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig = plt.figure(2, clear=True)
ax = fig.add_subplot(111, projection='3d')

vmin = np.min(fields[:,0])
vmax = np.max(fields[:,0])

for i in range(nElems):
    elem = connectivity[i,:]
    n = elem.shape[0] - 1

    if elem[0] == 5:
        n = 3
    elif elem[0] == 9:
        n = 4
    elif elem[0] == 3:
        n = 2

    ps = points[elem[1:n+1]]
    fs = [fields[i,1] for j in range(n)]
    visPatch(ax, ps[:,0], ps[:,1], fs, vmin=vmin, vmax=vmax)
    # plotField(points[elem[1:n+1]],fields[i,1])

ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
plt.show(block=False)
'''

# if still going down this route, need:
# - hashmap to identify which elements share faces
# - element - face - element connectivity

def polygonArea(ps):
    area = 0.0
    for i in range(len(ps)):
        j = (i+1) % len(ps)
        area += ps[i][0]*ps[j][1] - ps[i][1]*ps[j][0]
    return abs(area) / 2.0


class Node:
    def __init__(self,idx):
        self.idx = idx
        self.vert = points[idx]
        self.faces = []
        self.cells = []

class Cell:
    def __init__(self,idx,verts,centre=None):
        self.idx = idx
        self.verts = verts
        self.faces = []
        self.nodes = []
        self.area = polygonArea(points[verts])
        self.centre = centre
        self.ring = []
        self.LHS = None
        self.RHS = None
        self.A = None
        self.AT = None
        self.b = None
        self.x = None
        self.C = None
        self.d = None
        self.w = None

class Face:
    def __init__(self,idx,verts):
        self.idx = idx
        self.verts = verts
        self.cells = []
        self.nodes = []
        self.isBoundary = True
        self.boundaryIdx = None
        self.centre = None
        self.norm = None
        self.area = 0.0

cells = []
faces = []

hm = {}
ps = []
for i in range(1,7):
    ps.append(10**(i*5))

if dim != 2:
    print('error: cells / faces not supported in 2d')

nodes = []
for i in range(nPoints):
    nodes.append(Node(i))

iface = 0
for i in range(nElems):

    elem = connectivity[i,:]
    n = elem.shape[0] - 1

    if elem[0] == 5:
        n = 3
    elif elem[0] == 9:
        n = 4
    elif elem[0] == 7:
        for m in range(n,0,-1):
            if connectivity[i][m] != 0:
                n = m
                break
        elem = connectivity[i,:n+1]

    elem = elem[1:n+1]

    cells.append(Cell(i,elem,centres[i]))
    for j in elem:
        nodes[j].cells.append(cells[-1])

    for k in range(n):

        v1 = elem[k]
        v2 = elem[(k+1)%n]

        v1,v2 = sorted((v1,v2))

        key = v1 + v2 * ps[1]

        val = hm.get(key)

        if val is None:
            faces.append(Face(iface,[v1,v2]))
            nodes[v1].faces.append(faces[-1])
            nodes[v2].faces.append(faces[-1])
            hm[key] = iface
            iface += 1
            faces[-1].cells.append(cells[i])
            cells[i].faces.append(faces[-1])
        else:
            faces[val].cells.append(cells[i])
            faces[val].isBoundary = False
            cells[i].faces.append(faces[val])
            hm.pop(key)

for i in range(nPoints):
    for c in nodes[i].cells:
        c.nodes.append(nodes[i])
    for f in nodes[i].faces:
        f.nodes.append(nodes[i])

'''
plt.figure(3)

i = 8

ch = [cells[i].idx]
for f in cells[i].faces:
    for c in f.cells:
        if c.idx not in ch:
            visPoly(points[c.verts])
            ch.append(c.idx)
            # print('1-ring:',c.idx)
            for f2 in c.faces:
                for c2 in f2.cells:
                    if c2.idx not in ch:
                        visPoly(points[c2.verts],lc='g')
                        # print('2-ring:',c2.idx)
                        ch.append(c2.idx)
                        for f3 in c2.faces:
                            for c3 in f3.cells:
                                if c3.idx not in ch:
                                    visPoly(points[c3.verts],lc='g')
                                    # print('3-ring:',c3.idx)
                                    ch.append(c3.idx)
                                    # for f4 in c3.faces:
                                        # for c4 in f4.cells:
                                            # if c4.idx not in ch:
                                                # visPoly(points[c4.verts],lc='g')
                                                # print('4-ring:',c4.idx)
                                                # ch.append(c4.idx)

'''

for cell in cells:
    cell.ring = []

#### for symmetric stencil, need:
## 1-ring (these are all in sym, and used for next cell search)
## 2-ring (these are searched for sym cells)

symmetricStencil = True

visualise2 = False

if symmetricStencil:
    adaptiveRingSize = False
    if adaptiveRingSize:
        minimumRingSize = 9     # for tri
        minimumRingSize = 6     # for poly
    numRings = 1
    # numRings = 2
    for cell in cells:
        ring1 = []
        ring2 = []
        sring = []
        #
        ringsDone = 0
        visitThisRing = [cell.idx]
        visitNextRing = []
        cellsVisited = []
        localNumRings = numRings
        # loop over rings to do
        while ringsDone <= localNumRings:
            # loop over cells in this ring yet to do
            while len(visitThisRing) > 0:
                c = cells[visitThisRing[0]]
                if c.idx not in cellsVisited:
                    cellsVisited.append(c.idx)
                    if c.idx != cell.idx:
                        if ringsDone == 1:
                            ring1.append(c)
                        # else:
                            # ring2.append(c)
                # loop over faces for this cell
                for f in c.faces:
                    # loop over cells connected to those faces
                    for oc in f.cells:
                        if oc.idx not in cellsVisited:
                            visitNextRing.append(oc.idx)
                _ = visitThisRing.pop(0)
            ringsDone += 1
            visitThisRing = visitNextRing
            visitNextRing = []
            # if adaptiveRingSize:
                # if (len(ring2) < minimumRingSize) and ringsDone > localNumRings:
                    # localNumRings += 1
        # for search ring, go to node neighbours
        ring1i = [oc.idx for oc in ring1]
        ring2i = []
        for n in cell.nodes:
            for oc in n.cells:
                if oc.idx not in ring1i and oc.idx not in ring2i:
                    ring2.append(oc)
                    ring2i.append(oc.idx)
                if visualise2:
                    visPoly(points[oc.verts],lc=mcc[8])
                    visPoint(oc.centre,lc=mcc[8],ms=10)
        # now have 1-ring and 2-ring
        # adding 1-ring cells to sym-ring
        for oc1 in ring1:
            sring.append(oc1)
            if visualise2:
                visPoly(points[oc1.verts],lc=mcc[2],lw=4)
                visPoint(oc1.centre,lc=mcc[2],ms=10)
        # print('1-ring: ', [c.idx for c in ring1])
        # print('2-ring: ', [c.idx for c in ring2])
        # loop 1-ring and add that cell and a symmetric cell from 2-ring
        rgb = 'rgb'
        for i1,oc1 in enumerate(ring1):
            ri = rgb[i1]
            # print('---', oc1.centre)
            ejk = (oc1.centre - cell.centre) / np.linalg.norm(oc1.centre - cell.centre)
            if visualise2:
                plt.plot([cell.centre[0],cell.centre[0]+0.1*ejk[0]],
                         [cell.centre[1],cell.centre[1]+0.1*ejk[1]],ri+'-')
                alpha = 3./4. * np.pi
                ejk1 = ( ejk[0]*np.cos(alpha) + ejk[1]*np.sin(alpha),
                        -ejk[0]*np.sin(alpha) + ejk[1]*np.cos(alpha))
                alpha = 5./4. * np.pi
                ejk2 = ( ejk[0]*np.cos(alpha) + ejk[1]*np.sin(alpha),
                        -ejk[0]*np.sin(alpha) + ejk[1]*np.cos(alpha))
                plt.plot([cell.centre[0],cell.centre[0]+0.1*ejk1[0]],
                         [cell.centre[1],cell.centre[1]+0.1*ejk1[1]],ri+'--')
                plt.plot([cell.centre[0],cell.centre[0]+0.1*ejk2[0]],
                         [cell.centre[1],cell.centre[1]+0.1*ejk2[1]],ri+':')
            #
            poc2 = None  # possible 2-ring other cells
            mindv = 10.0
            for oc2 in ring2:
                dv = np.dot((oc2.centre - cell.centre) / np.linalg.norm(oc2.centre - cell.centre), ejk)
                # print(oc2.centre, dv)
                if dv < np.cos(3./4. * np.pi):
                    if dv < mindv:
                        poc2 = oc2
                        mindv = dv
            if poc2 is None:
                # failed to find cell that satisfied requirements, continue
                pass
            else:
                # print('-->', poc2.centre, mindv)
                sring.append(poc2)
                ring2.remove(poc2)
        # print('s-ring: ', [c.idx for c in sring])
        cell.ring = sring;
        if visualise2:
            for c in cell.ring:
                visPoly(points[c.verts],lc='g')
                visPoint(c.centre,lc=mcc[2],ms=10)
else:
    adaptiveRingSize = False
    if adaptiveRingSize:
        minimumRingSize = 9     # for tri
        minimumRingSize = 6     # for poly
    # numRings = 1            # for poly
    numRings = 2            # for tri
    # numRings = 3
    for cell in cells:
        ringsDone = 0
        visitThisRing = [cell.idx]
        visitNextRing = []
        cellsVisited = []
        localNumRings = numRings
        # loop over rings to do
        while ringsDone <= localNumRings:
            # loop over cells in this ring yet to do
            while len(visitThisRing) > 0:
                c = cells[visitThisRing[0]]
                if c.idx not in cellsVisited:
                    cellsVisited.append(c.idx)
                    if c.idx != cell.idx:
                        # cell.ring.append(c.idx)
                        cell.ring.append(c)
                # loop over faces for this cell
                for f in c.faces:
                    # loop over cells connected to those faces
                    for oc in f.cells:
                        if oc.idx not in cellsVisited:
                            visitNextRing.append(oc.idx)
                _ = visitThisRing.pop(0)
            ringsDone += 1
            visitThisRing = visitNextRing
            visitNextRing = []
            if adaptiveRingSize:
                if (len(cell.ring) < minimumRingSize) and ringsDone > localNumRings:
                    localNumRings += 1


# for n in cells[1000].nodes:
    # for oc in n.cells:
        # visPoly(points[oc.verts],lc=mcc[8])
        # visPoint(oc.centre,lc=mcc[8],ms=10)

plt.show(block=False)

# cells with large rings
stens = np.array([len(cell.ring) for cell in cells])
bigRings = np.arange(stens.shape[0])[stens>9]

if visualise:
    if bigRings.shape[0] > 0:
        ci = 17
        plt.figure(1)
        s = 10
        for ci in bigRings[::s]:
            for c in cells[ci].ring:
                visPoly(points[c.verts],lc=mcc[2])
        #
        for ci in bigRings[::s]:
            visPoly(points[cells[ci].verts],lc=mcc[8])
        #
        plt.show(block=False)


if visualise:
    ci = 237
    for c in cells[ci].ring:
        visPoly(points[c.verts],lc=mcc[2])
    #
    visPoly(points[cells[ci].verts],lc=mcc[8])
    plt.show(block=False)

'''
for c in ch:
    visPoint(cells[c].centre)

for f in cells[i].faces:
    visPoly(points[f.verts],lc=mcc[8])

plt.show(block=False)
'''

if not tris:
    npolyts = 0
    for c in cells:
        npolyts += len(c.faces)

    tpoints = points.copy()
    for c in cells:
        tpoints = np.vstack((tpoints,c.centre))

    tfields = np.zeros((npolyts,fields.shape[1]))
    tconn = np.zeros((npolyts,4),dtype=np.int64)
    j = 0
    for i,c in enumerate(cells):
        for f in c.faces:
            tconn[j,0] = 3
            tconn[j,1] = points.shape[0] + i
            tconn[j,2] = f.verts[0]
            tconn[j,3] = f.verts[1]
            tfields[j,:] = fields[i,:]
            j += 1


# figuring out which are some edge triangles
# visPoly(points[cells[ 8].verts],lc='k')
# visPoly(points[cells[ 9].verts],lc='r')
# visPoly(points[cells[11].verts],lc='b')#
# visPoly(points[cells[12].verts],lc='m')
# plt.show(block=False)


if tris:
    # TRIANGLES ONLY!
    triang = mtri.Triangulation(points[:,0], points[:,1], connectivity[:,1:])

    if visualise:
        plt.tripcolor(points[:,0],points[:,1],connectivity[:,1:],facecolors=fields[:,2],cmap=cm.coolwarm)
        plt.triplot(triang,'ko-',markersize=2)
        plt.show(block=False)

        for c in ch:
            plt.tripcolor(points[connectivity[c,1:],0],
                          points[connectivity[c,1:],1],
                          connectivity[c,1:],facecolors=fields[c,1:2],
                          cmap=cm.coolwarm
                          ,vmin=np.min(fields[ch,1:2])
                          ,vmax=np.max(fields[ch,1:2]))
            visPoly(points[connectivity[c,1:]])

        plt.show(block=False)


        for c in [a.idx for a in cells[8].ring]:
            plt.tripcolor(points[connectivity[c,1:],0],
                          points[connectivity[c,1:],1],
                          connectivity[c,1:],facecolors=fields[c,1:2]*0,
                          cmap=cm.coolwarm
                          ,vmin=np.min(fields[ch,1:2])
                          ,vmax=np.max(fields[ch,1:2]))
            visPoly(points[connectivity[c,1:]])

        plt.show(block=False)

boundaryFaces = []
i = 0
for f in faces:
    f.centre = np.sum(points[f.verts],0)/2.0
    p1,p2 = points[f.verts[0]], points[f.verts[1]]
    nf = np.array([p1[1]-p2[1], p2[0]-p1[0]])
    f.norm = nf / np.dot(nf,nf)**0.5
    df = p2 - p1
    f.area = np.dot(df,df)**0.5
    if f.isBoundary:
        boundaryFaces.append(f)
        f.boundaryIdx = i
        i += 1

nBoundaryFaces = len(boundaryFaces)

fFields = np.zeros((nBoundaryFaces,1))
for f in boundaryFaces:
    fFields[f.boundaryIdx,0] = func(*f.centre)




fig = plt.figure(48,figsize=(20,20))
for f in faces:
    plt.plot(points[f.verts][:,0],points[f.verts][:,1], 'k-',lw=0.3)

c = cells[10]
for n in c.nodes:
    for oc in n.cells:
        visPoly(points[oc.verts],lc=mcc[8])
        visPoint(oc.centre,lc=mcc[8],ms=10)

for oc in c.ring:
    visPoly(points[oc.verts],lc=mcc[2],)
    visPoint(oc.centre,lc=mcc[2],ms=10)

visPoly(points[c.verts],lc=mcc[0],ml='-.')
visPoint(c.centre,lc=mcc[0],ms=10)
plt.show(block=False)




dim = 2
plotS = True
plt.figure(49)
ci = 113
##### LEAST SQUARES STUFF #####
for c in cells:
    c.A = np.zeros((len(c.ring),dim))
    c.b = np.zeros((len(c.ring),1))
    c.w = np.eye(len(c.ring))
    c.x = np.zeros((dim,1))
    for i,oc in enumerate(c.ring):
        for j in range(dim):
            c.A[i,j] = oc.centre[j] - c.centre[j]
        # p = 0
        # c.w[i,i] = 1.0
        # p = 1
        # c.w[i,i] = 1.0 / np.sum(c.A[i,:]**2.)**0.5
        # p = 2
        # c.w[i,i] = 1.0 / np.sum(c.A[i,:]**2.)
        # p = 2/3
        # c.w[i,i] = 1.0 / (np.sum(c.A[i,:]**2.)**0.5)**(2/3.)
        # p = 3/2
        # c.w[i,i] = 1.0 / (np.sum(c.A[i,:]**2.)**0.5)**(3/2.)
        # p = -3/2
        c.w[i,i] = (np.sum(c.A[i,:]**2.)**0.5)**(-3/2.)
        # c.w[i,i] = 1.0 / (np.sum(c.A[i,:]**2.)**0.5)**(-3/2.)
        c.b[i,0] = fields[oc.idx,2] - fields[c.idx,2]
        if plotS and c.idx == ci:
            visPoly(points[oc.verts],lc=mcc[2])
            visPoint(oc.centre,lc=mcc[2],ms=10)

    if plotS and c.idx == ci:
        visPoly(points[c.verts],lc=mcc[0],ml='-.')
        visPoint(c.centre,lc=mcc[0],ms=10)

    # soft constraint
    if False:
        for f in c.faces:
            if f.isBoundary:
                c.A = np.vstack((c.A,np.zeros((1,dim))))
                for j in range(dim):
                    c.A[-1,j] = f.centre[j] - c.centre[j]
                c.b = np.vstack((c.b,np.zeros((1,1))))
                c.b[-1,0] = fFields[f.boundaryIdx,0] - fields[c.idx,2]

    weighting = True
    # weighting = False

    c.AT = c.A.T
    if weighting:
        c.LHS = np.dot(np.dot(c.AT,c.w),c.A)
        c.RHS = np.dot(np.dot(c.AT,c.w),c.b)
    else:
        c.LHS = np.dot(c.AT,c.A)
        c.RHS = np.dot(c.AT,c.b)

    # hard constraints
    if True:
        locBoundaryFaces = 0
        for f in c.faces:
            if f.isBoundary:
                locBoundaryFaces += 1

        includeNeighbourBFs = False

        if includeNeighbourBFs:
            # may want to include neighbouring cells boundary faces
            for f in c.faces:
                for cn in f.cells:
                    if cn.idx == c.idx:
                        continue
                    for fn in cn.faces:
                        if fn.isBoundary:
                            locBoundaryFaces += 1

        if locBoundaryFaces > 0:
            c.LHS *= 2.0
            c.RHS *= 2.0
            c.C = np.zeros((locBoundaryFaces,dim))
            c.d = np.zeros((locBoundaryFaces,1))
            # c.z = np.zeros((locBoundaryFaces,1))

            # hard constraint; Dirichlet
            # grad_F0 . (xf - x0) = (ff - f0)
            if False:
                i = 0
                for f in c.faces:
                    if f.isBoundary:
                        for j in range(dim):
                            c.C[i,j] = f.centre[j] - c.centre[j]
                        c.d[i,0] = fFields[f.boundaryIdx,0] - fields[c.idx,2]
                        i += 1

            # hard constraint; Neumann
            # grad_F0 . (xf - x0) = (ff - f0)  <-- at cell
            # grad_Ff . (x0 - xf) = (f0 - ff)  <-- at face
            # ff = f0 - grad_Ff . (x0 - xf)
            # therefore
            # grad_F0 . (xf - x0) = -grad_Ff . (x0 - xf)
            if False:
                i = 0
                for f in c.faces:
                    if f.isBoundary:
                        for j in range(dim):
                            c.C[i,j] = f.centre[j] - c.centre[j]
                        # only RHS changes
                        gradFf = np.array([
                            dfdx(*f.centre),
                            dfdy(*f.centre)])
                        dx = c.centre - f.centre
                        c.d[i,0] = -gradFf[0]*dx[0] - gradFf[1]*dx[1]
                        i += 1

            # hard constraint; correct Neumann (grad_f . n is set)
            if True:
                i = 0
                for f in c.faces:
                    if f.isBoundary:
                        # calculate the normal
                        p1,p2 = points[f.verts[0]], points[f.verts[1]]
                        nx,ny = p1[1]-p2[1], p2[0]-p1[0]
                        cvec = f.centre - c.centre
                        if np.dot(cvec,np.array([nx,ny])) < 0.0:
                            nx *= -1.0
                            ny *= -1.0
                        #
                        mag = (nx*nx + ny*ny)**0.5
                        nx /= mag
                        ny /= mag
                        #
                        gradfnx = dfdx(*f.centre) * nx
                        gradfny = dfdy(*f.centre) * ny
                        c.C[i,0] = nx
                        c.C[i,1] = ny
                        c.d[i,0] = gradfnx + gradfny
                        i += 1
                        if plotS and c.idx == ci:
                            visPoly(points[f.verts],lc=mcc[7],ml='--')
                            visPoint(f.centre,lc=mcc[7],m='x',ms=10)


                if includeNeighbourBFs:
                    # include neighbouring cells boundary faces
                    for f in c.faces:
                        for cn in f.cells:
                            if cn.idx == c.idx:
                                continue

                            for fn in cn.faces:
                                if fn.isBoundary:
                                    # calculate the normal
                                    p1,p2 = points[fn.verts[0]], points[fn.verts[1]]
                                    nx,ny = p1[1]-p2[1], p2[0]-p1[0]
                                    cvec = fn.centre - c.centre
                                    if np.dot(cvec,np.array([nx,ny])) < 0.0:
                                        nx *= -1.0
                                        ny *= -1.0
                                    #
                                    mag = (nx*nx + ny*ny)**0.5
                                    nx /= mag
                                    ny /= mag
                                    #
                                    gradfnx = dfdx(*fn.centre) * nx
                                    gradfny = dfdy(*fn.centre) * ny
                                    c.C[i,0] = nx
                                    c.C[i,1] = ny
                                    c.d[i,0] = gradfnx + gradfny
                                    i += 1
                                    if plotS and c.idx == ci:
                                        visPoly(points[fn.verts],lc=mcc[9],ml='--')
                                        visPoint(fn.centre,lc=mcc[9],m='*',ms=10)


            c.LHS = np.vstack((
                np.hstack((c.LHS, c.C.T)),
                np.hstack((  c.C, np.zeros((locBoundaryFaces,locBoundaryFaces))))
                ))
            c.RHS = np.vstack((
                c.RHS,
                c.d
                ))


plt.show(block=False)

# for c in cells:
    # c.x = np.dot(
            # np.dot(
                # np.linalg.inv(
                    # np.dot(c.AT,c.A)),
                # c.AT),
            # c.b)

for c in cells:
    c.x = np.dot(np.linalg.inv(c.LHS),c.RHS)


# alternative approach for calculating gradients -> modified Green-Gauss
##### MODIFIED GREEN GAUSS STUFF #####
grads = np.zeros((len(cells),2,2))
j1 = -1
j2 = 0
gradsave = np.zeros((len(cells),2,10))
for j in range(10):  # number of times to iterate
    j1 = (j1 + 1) % 2
    j2 = (j2 + 1) % 2
    for c in cells:
        grads[c.idx,:,j2] = 0.0
        for f in c.faces:
            if len(f.cells) == 1:
                continue
            oc = f.cells[0] if f.cells[0].idx != c.idx else f.cells[1]
            nf = f.norm
            rf = oc.centre - c.centre
            dr = np.dot(rf,rf)**0.5
            rf = rf / dr
            alpha = np.dot(nf,rf)
            dphidn = (alpha / dr * (fields[oc.idx,2] - fields[c.idx,2])
                    + 0.5 * np.dot(grads[c.idx,:,j1] + grads[oc.idx,:,j1], nf - alpha * rf))
            grads[c.idx,:,j2] += dphidn * (f.centre - c.centre) * f.area
        grads[c.idx,:,j2] /= c.area
    gradsave[:,:,j] = grads[:,:,j2]

# res = 0.0
# res2 = 0.0
# for f in c.faces:
    # if len(f.cells) == 1:
        # continue
    # oc = f.cells[0] if f.cells[0].idx != c.idx else f.cells[1]
    # nf = f.norm
    # rf = oc.centre - c.centre
    # dr = np.dot(rf,rf)**0.5
    # rf = rf / dr
    # alpha = np.dot(nf,rf)
    # print(alpha)
    # dphidn = alpha / dr * (fields[oc.idx,2] - fields[c.idx,2])
    # res += dphidn * (f.centre - c.centre) * f.area
    # res2 += func(*f.centre) * nf * f.area

# res /= c.area
# res2 /= c.area

# fres = 0.0
# fakefs = np.zeros((5,2))
# fakefs[1] = [-0.5,0]
# fakefs[2] = [0,0.5]
# fakefs[3] = [0.5,0]
# fakefs[4] = [0,-0.5]
# for oc in range(1,5):
    # rf = fakeps[oc] - fakeps[0]
    # dr = np.dot(rf,rf)**0.5
    # dphidn = (func(*fakeps[oc]) - func(*fakeps[0])) / dr
    # fres += dphidn * (fakefs[oc] - fakeps[0]) * 1.0

# fres /= 1.0

# c = cells[0]
# plt.plot(c.centre[0],c.centre[1],'gs')
# for oc in c.ring:
    # plt.plot(oc.centre[0],oc.centre[1],'gs')
    # for f in oc.faces:
        # plt.plot(points[f.verts][:,0],points[f.verts][:,1],'b-')

# for f in c.faces:
    # plt.plot(points[f.verts][:,0],points[f.verts][:,1],'k-')
    # plt.plot(f.centre[0],f.centre[1],'ko')
    # nf = f.norm
    # rf = f.centre - c.centre
    # if np.dot(nf,rf) < 0.0:
        # nf *= -1.0
    # plt.plot([f.centre[0],f.centre[0]+nf[0]],
             # [f.centre[1],f.centre[1]+nf[1]],'k-')
    # plt.plot([f.centre[0],f.centre[0]+rf[0]],
             # [f.centre[1],f.centre[1]+rf[1]],'r--')

# plt.show(block=False)

errGradF = np.zeros((len(cells),2))
for c in cells:
    errGradF[c.idx,0] = c.x[0] - dfdx(*c.centre)
    errGradF[c.idx,1] = c.x[1] - dfdy(*c.centre)

errL2dfdx = (np.sum(errGradF[:,0]**2.) / nElems)**0.5
errL2dfdy = (np.sum(errGradF[:,1]**2.) / nElems)**0.5

errGradF_MGG = np.zeros((len(cells),2))
for c in cells:
    errGradF_MGG[c.idx,0] = grads[c.idx,0,j2] - dfdx(*c.centre)
    errGradF_MGG[c.idx,1] = grads[c.idx,1,j2] - dfdy(*c.centre)

errL2dfdx_MGG = (np.sum(errGradF_MGG[:,0]**2.) / nElems)**0.5
errL2dfdy_MGG = (np.sum(errGradF_MGG[:,1]**2.) / nElems)**0.5

gradF_f = np.zeros((len(faces),2))
errGradF_f = np.zeros((len(faces),2))
method = 3
for f in faces:
    if len(f.cells) == 2:
        if method == 0: # averaging
            gradF_f[f.idx,0] = 0.5 * (f.cells[0].x[0] + f.cells[1].x[0])
            gradF_f[f.idx,1] = 0.5 * (f.cells[0].x[1] + f.cells[1].x[1])
        elif method == 1: # Nishikawa method
            rjF = f.centre - f.cells[0].centre
            riF = f.centre - f.cells[1].centre
            phij = fields[f.cells[0].idx,2]
            phii = fields[f.cells[1].idx,2]
            phiFp = phij + (rjF[0]*f.cells[0].x[0] + rjF[1]*f.cells[0].x[1])
            phiFm = phii + (riF[0]*f.cells[1].x[0] + riF[1]*f.cells[1].x[1])
            rij = f.cells[0].centre - f.cells[1].centre
            eij = rij / (rij[0]**2. + rij[1]**2)**0.5
            alpha = (abs(f.norm[0] * eij[0] + f.norm[1] * eij[1])
                  / (f.norm[0] * eij[0] + f.norm[1] * eij[1]))
            gradF_f[f.idx,0] = (0.5 * (f.cells[0].x[0] + f.cells[1].x[0])
                    + alpha / abs(rij[0] * f.norm[0] + rij[1] * f.norm[1])
                      * (phiFp - phiFm) * f.norm[0])
            gradF_f[f.idx,1] = (0.5 * (f.cells[0].x[1] + f.cells[1].x[1])
                    + alpha / abs(rij[0] * f.norm[0] + rij[1] * f.norm[1])
                      * (phiFp - phiFm) * f.norm[1])
        elif method == 2:
            # Hasselbacher method: edge-normal augmented cell-face gradient
            rij = f.cells[0].centre - f.cells[1].centre
            eij = rij / (rij[0]**2. + rij[1]**2)**0.5
            eijN = (eij[0]**2. + eij[1]**2.)**0.5
            phij = fields[f.cells[0].idx,2]
            phii = fields[f.cells[1].idx,2]
            gradFavg = 0.5 * (f.cells[0].x[:2] + f.cells[1].x[:2])
            gradF_f[f.idx,0] = gradFavg[0] - (
                    gradFavg[0]*eij[0] + gradFavg[1]*eij[1]
                  - (phij - phii) / eijN) * eij[0]
            gradF_f[f.idx,1] = gradFavg[1] - (
                    gradFavg[0]*eij[0] + gradFavg[1]*eij[1]
                  - (phij - phii) / eijN) * eij[1]
        elif method == 3:
            # Hasselbacher method: face-tangent augmented cell-face gradient
            rij = f.cells[0].centre - f.cells[1].centre
            eij = rij / (rij[0]**2. + rij[1]**2)**0.5
            nf = f.norm
            eijN = (eij[0]**2. + eij[1]**2.)**0.5
            phij = fields[f.cells[0].idx,2]
            phii = fields[f.cells[1].idx,2]
            gradFavg = 0.5 * (f.cells[0].x[:2] + f.cells[1].x[:2])
            gradF_f[f.idx,0] = gradFavg[0] - (
                    (gradFavg[0]*eij[0] + gradFavg[1]*eij[1])
                  - (phij - phii) / eijN) * (nf[0]/(nf[0]*eij[0] + nf[1]*eij[1]))
            gradF_f[f.idx,1] = gradFavg[1] - (
                    (gradFavg[0]*eij[0] + gradFavg[1]*eij[1])
                  - (phij - phii) / eijN) * (nf[1]/(nf[0]*eij[0] + nf[1]*eij[1]))

    else:
        # all one-sided?
        gradF_f[f.idx,0] = f.cells[0].x[0]
        gradF_f[f.idx,1] = f.cells[0].x[1]

    errGradF_f[f.idx,0] = gradF_f[f.idx,0] - dfdx(*f.centre)
    errGradF_f[f.idx,1] = gradF_f[f.idx,1] - dfdy(*f.centre)

errL2dfdx_f = (np.sum(errGradF_f[:,0]**2.) / len(faces))**0.5
errL2dfdy_f = (np.sum(errGradF_f[:,1]**2.) / len(faces))**0.5

if False:
    # fp = open('res.dat', 'a')
    # fp = open('tmp.dat', 'a')
    fp = open('tmp2.dat', 'a')
    fp.write('number of elements: {}\n'.format(nElems))
    fp.write('type of elements: {}\n'.format(np.unique(connectivity[:,0])))
    fp.write('number of face rings: {}\n'.format(numRings))
    fp.write('cells: min, max, avg of dfdx: {:.10f}, {:.10f}, {:.10f}\n'.format(
            np.min(errGradF[:,0]),
            np.max(errGradF[:,0]),
            np.sum(errGradF[:,0]) / nElems))
    fp.write('cells: min, max, avg of dfdy: {:.10f}, {:.10f}, {:.10f}\n'.format(
            np.min(errGradF[:,1]),
            np.max(errGradF[:,1]),
            np.sum(errGradF[:,1]) / nElems))
    fp.write('cells: L2 dfdx, dfdy: {:.10f}, {:.10f}\n\n'.format(
        errL2dfdx,errL2dfdy))
    fp.write('faces: min, max, avg of dfdx: {:.10f}, {:.10f}, {:.10f}\n'.format(
            np.min(errGradF_f[:,0]),
            np.max(errGradF_f[:,0]),
            np.sum(errGradF_f[:,0]) / nElems))
    fp.write('faces: min, max, avg of dfdy: {:.10f}, {:.10f}, {:.10f}\n'.format(
            np.min(errGradF_f[:,1]),
            np.max(errGradF_f[:,1]),
            np.sum(errGradF_f[:,1]) / nElems))
    fp.write('faces: L2 dfdx, dfdy: {:.10f}, {:.10f}\n\n'.format(
        errL2dfdx_f,errL2dfdy_f))
    fp.close()

else:
    print('number of elements: {}'.format(nElems))
    print('type of elements: {}'.format(np.unique(connectivity[:,0])))
    print('number of face rings: {}'.format(numRings))
    print('cells: min, max, amin, avg of dfdx: {:.10f}, {:.10f}, {:.10f}, {:.10f}'.format(
            np.min(errGradF[:,0]),
            np.max(errGradF[:,0]),
            np.min(np.abs(errGradF[:,0])),
            np.sum(errGradF[:,0]) / nElems))
    print('cells: min, max, amin, avg of dfdy: {:.10f}, {:.10f}, {:.10f}, {:.10f}'.format(
            np.min(errGradF[:,1]),
            np.max(errGradF[:,1]),
            np.min(np.abs(errGradF[:,1])),
            np.sum(errGradF[:,1]) / nElems))
    print('cells: L2 dfdx, dfdy: {:.10f}, {:.10f}'.format(
        errL2dfdx,errL2dfdy))
    print('faces: min, max, avg of dfdx: {:.10f}, {:.10f}, {:.10f}'.format(
            np.min(errGradF_f[:,0]),
            np.max(errGradF_f[:,0]),
            np.sum(errGradF_f[:,0]) / nElems))
    print('faces: min, max, avg of dfdy: {:.10f}, {:.10f}, {:.10f}'.format(
            np.min(errGradF_f[:,1]),
            np.max(errGradF_f[:,1]),
            np.sum(errGradF_f[:,1]) / nElems))
    print('faces: L2 dfdx, dfdy: {:.10f}, {:.10f}'.format(
        errL2dfdx_f,errL2dfdy_f))
    print('cells MGG: min, max, amin, avg of dfdx: {:.10f}, {:.10f}, {:.10f}, {:.10f}'.format(
            np.min(errGradF_MGG[:,0]),
            np.max(errGradF_MGG[:,0]),
            np.min(np.abs(errGradF_MGG[:,0])),
            np.sum(errGradF_MGG[:,0]) / nElems))
    print('cells MGG: min, max, amin, avg of dfdy: {:.10f}, {:.10f}, {:.10f}, {:.10f}'.format(
            np.min(errGradF_MGG[:,1]),
            np.max(errGradF_MGG[:,1]),
            np.min(np.abs(errGradF_MGG[:,1])),
            np.sum(errGradF_MGG[:,1]) / nElems))
    print('cells: L2 dfdx, dfdy: {:.10f}, {:.10f}'.format(
        errL2dfdx_MGG,errL2dfdy_MGG))

if tris:
    # TRIANGLES ONLY!
    triang = mtri.Triangulation(points[:,0], points[:,1], connectivity[:,1:])

cmap = cm.coolwarm
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap = colors.LinearSegmentedColormap.from_list(
        'ccmap',cmaplist,cmap.N)
# bounds = np.linspace(-0.5,0.5,13)
bounds = np.linspace(-10,1,100)
# bounds = np.linspace(-5,1,100)
norm = colors.BoundaryNorm(bounds,cmap.N)

vertErrs = np.zeros((nPoints,2))
numAccum = np.zeros((nPoints))
for c in cells:
    for v in c.verts:
        vertErrs[v,0] += errGradF[c.idx,0]
        vertErrs[v,1] += errGradF[c.idx,1]
        numAccum[v] += 1

vertErrs[:,0] /= numAccum
vertErrs[:,1] /= numAccum

vi = -0.5; va = 0.5

cmap = cm.coolwarm
# cmap = cm.coolwarm_r
# cmap = parula_r_map
# cmap = cm.viridis

plotEdges = True

# TRIANGLES ONLY
if tris:
    fig = plt.figure(111,figsize=(20,20))
    if plotEdges:
        for f in faces:
            plt.plot(points[f.verts][:,0],points[f.verts][:,1], 'w-',lw=0.3)
    plt.title(r'$log$ of error of $\partial f / \partial x$',y=1.0,pad=-14)
    ax = plt.gca()
    # tpc = ax.tripcolor(points[:,0],points[:,1],connectivity[:,1:],
            # facecolors=errGradF[:,0],cmap=cm.coolwarm,vmin=vi,vmax=va)
    # tpc = ax.tripcolor(points[:,0],points[:,1],connectivity[:,1:],
            # facecolors=errGradF[:,0],cmap=cmap)#,norm=norm)
    tpc = ax.tripcolor(points[:,0],points[:,1],connectivity[:,1:],
            facecolors=np.log(np.abs(errGradF[:,0])),cmap=cmap,norm=norm)
    # tpc = ax.tripcolor(triang,vertErrs[:,0],cmap=cmap,#vmin=vi,vmax=va,
            # norm=norm,shading='flat')
    # tpc = ax.tripcolor(triang,vertErrs[:,0],cmap=cmap,#vmin=vi,vmax=va,
            # norm=norm,shading='gouraud')
    # cb = plt.colorbar.ColorbarBase(ax,cmap=cmap,norm=norm,ticks=bounds,
            # boundaries=bounds)
    # cb = plt.colorbar.ColorbarBase(ax,cmap=cmap,norm=norm,ticks=bounds,
            # boundaries=bounds)
    # plt.triplot(triang,'w-',lw=0.1)
    # fig.colorbar(tpc)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    # plt.savefig('err_dfdx_4.png',dpi=400)
    plt.show(block=False)

    # fig = plt.figure(111)
    # ax = plt.gca()
    # tpc = ax.tripcolor(points[:,0],points[:,1],connectivity[:,1:],
            # facecolors=errGradF[:,1],cmap=cm.coolwarm,vmin=vi,vmax=va)
    # # plt.triplot(triang,'w-',lw=0.1)
    # fig.colorbar(tpc)
    # plt.show(block=False)



    fig = plt.figure(112,figsize=(20,20))
    if plotEdges:
        for f in faces:
            plt.plot(points[f.verts][:,0],points[f.verts][:,1], 'w-',lw=0.3)
    plt.title(r'$log$ of error of $\partial f / \partial y$',y=1.0,pad=-14)
    # plt.title(r'$\frac{\partial f}{\partial y}$',y=1.0,pad=-14)
    ax = plt.gca()
    # tpc = ax.tripcolor(points[:,0],points[:,1],connectivity[:,1:],
            # facecolors=errGradF[:,1],cmap=cm.coolwarm,vmin=vi,vmax=va)
    # tpc = ax.tripcolor(points[:,0],points[:,1],connectivity[:,1:],
            # facecolors=errGradF[:,1],cmap=cmap)#,norm=norm)
    tpc = ax.tripcolor(points[:,0],points[:,1],connectivity[:,1:],
            facecolors=np.log(np.abs(errGradF[:,1])),cmap=cmap,norm=norm)
    # cb = plt.colorbar.ColorbarBase(ax,cmap=cmap,norm=norm,ticks=bounds,
            # boundaries=bounds)
    # plt.triplot(triang,'w-',lw=0.1)
    # fig.colorbar(tpc)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    # plt.savefig('err_dfdy_4.png',dpi=400)
    plt.show(block=False)

else:

    terrGradF = np.zeros((npolyts,2))
    terrGradF_MGG = np.zeros((npolyts,2))
    j = 0
    for i,c in enumerate(cells):
        for f in c.faces:
            terrGradF[j,:] = errGradF[i,:]
            terrGradF_MGG[j,:] = errGradF_MGG[i,:]
            j += 1

    # fig = plt.figure(111,figsize=(20,20))   # tri, 1-ring
    # fig = plt.figure(121,figsize=(20,20))   # tri, 2-ring
    fig = plt.figure(131,figsize=(20,20))   # tri, s-ring
    if plotEdges:
        for f in faces:
            plt.plot(points[f.verts][:,0],points[f.verts][:,1], 'w-',lw=0.3)
    plt.title(r'$log$ of error of $\partial f / \partial x$, CWLSQ',y=1.0,pad=-14)
    ax = plt.gca()
    tpc = ax.tripcolor(tpoints[:,0],tpoints[:,1],tconn[:,1:],
            facecolors=np.log(np.abs(terrGradF[:,0])),cmap=cmap,norm=norm)
    # tpc = ax.tripcolor(tpoints[:,0],tpoints[:,1],tconn[:,1:],
            # facecolors=np.abs(terrGradF[:,0]),cmap=cmap)#,norm=norm)
    fig.colorbar(tpc)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    # plt.savefig('err_dfdx_poly.png',dpi=400)
    plt.show(block=False)

    # fig = plt.figure(112,figsize=(20,20))   # tri, 1-ring
    # fig = plt.figure(122,figsize=(20,20))   # tri, 2-ring
    fig = plt.figure(132,figsize=(20,20))   # tri, s-ring
    if plotEdges:
        for f in faces:
            plt.plot(points[f.verts][:,0],points[f.verts][:,1], 'w-',lw=0.3)
    plt.title(r'$log$ of error of $\partial f / \partial y$, CWLSQ',y=1.0,pad=-14)
    ax = plt.gca()
    tpc = ax.tripcolor(tpoints[:,0],tpoints[:,1],tconn[:,1:],
            facecolors=np.log(np.abs(terrGradF[:,1])),cmap=cmap,norm=norm)
    # tpc = ax.tripcolor(tpoints[:,0],tpoints[:,1],tconn[:,1:],
            # facecolors=np.abs(terrGradF[:,1]),cmap=cmap)#,norm=norm)
    fig.colorbar(tpc)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    # plt.savefig('err_dfdy_poly.png',dpi=400)
    plt.show(block=False)

    fig = plt.figure(113,figsize=(20,20))
    if plotEdges:
        for f in faces:
            plt.plot(points[f.verts][:,0],points[f.verts][:,1], 'w-',lw=0.3)
    plt.title(r'$log$ of error of $\partial f / \partial x$, MGG',y=1.0,pad=-14)
    ax = plt.gca()
    tpc = ax.tripcolor(tpoints[:,0],tpoints[:,1],tconn[:,1:],
            facecolors=np.log(np.abs(terrGradF_MGG[:,0])),cmap=cmap,norm=norm)
    # tpc = ax.tripcolor(tpoints[:,0],tpoints[:,1],tconn[:,1:],
            # facecolors=np.abs(terrGradF[:,0]),cmap=cmap)#,norm=norm)
    fig.colorbar(tpc)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    # plt.savefig('err_dfdx_poly.png',dpi=400)
    plt.show(block=False)

    fig = plt.figure(114,figsize=(20,20))
    if plotEdges:
        for f in faces:
            plt.plot(points[f.verts][:,0],points[f.verts][:,1], 'w-',lw=0.3)
    plt.title(r'$log$ of error of $\partial f / \partial y$, MGG',y=1.0,pad=-14)
    ax = plt.gca()
    # tpc = ax.tripcolor(tpoints[:,0],tpoints[:,1],tconn[:,1:],
            # facecolors=np.log(np.abs(terrGradF[:,1])),cmap=cmap,norm=norm)
    tpc = ax.tripcolor(tpoints[:,0],tpoints[:,1],tconn[:,1:],
            facecolors=np.log(np.abs(terrGradF_MGG[:,1])),cmap=cmap,norm=norm)
    # tpc = ax.tripcolor(tpoints[:,0],tpoints[:,1],tconn[:,1:],
            # facecolors=np.abs(terrGradF[:,1]),cmap=cmap)#,norm=norm)
    fig.colorbar(tpc)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    # plt.savefig('err_dfdy_poly.png',dpi=400)
    plt.show(block=False)

# visualise normals
if False:
    fig = plt.figure(113)
    for c in cells:
        for f in c.faces:
            if f.isBoundary:
                p1,p2 = points[f.verts[0]], points[f.verts[1]]
                nx,ny = p1[1]-p2[1], p2[0]-p1[0]
                cvec = f.centre - c.centre
                if np.dot(cvec,np.array([nx,ny])) < 0.0:
                    nx *= -1.0
                    ny *= -1.0
                #
                mag = (nx*nx + ny*ny)**0.5
                nx /= mag
                ny /= mag
                # visualise normals
                plt.plot([f.centre[0],f.centre[0]+nx],
                         [f.centre[1],f.centre[1]+ny],'b-')



# TRIANGLES ONLY
if False:
    ax = plt.gca()
    tpc = ax.tripcolor(points[:,0],points[:,1],connectivity[:,1:],
            facecolors=fields[:,-1],cmap=cm.coolwarm)
    # plt.triplot(triang,'ko-',markersize=2)
    fig.colorbar(tpc)
    plt.show(block=False)

# recasting polyhedral as tris for plotting

else:

    fig = plt.figure()
    ax = plt.gca()
    tpc = ax.tripcolor(tpoints[:,0],tpoints[:,1],tconn[:,1:],
            facecolors=tfields[:,-1],cmap=cm.coolwarm)
    # plt.triplot(triang,'ko-',markersize=2)
    fig.colorbar(tpc)
    plt.show(block=False)


dps = []
for c in cells:
    for f in c.faces:
        if f.isBoundary:
            dps.append(abs(np.dot(f.centre - c.centre, f.norm)))
        else:
            oc = f.cells[0]
            if f.cells[0].idx == c.idx:
                oc = f.cells[1]
            dps.append(abs(np.dot(oc.centre - c.centre, f.norm)))

# from scipy import stats
# dps = np.array(dps)
# print(stats.describe(dps))


# plt.figure(777)
# pa = np.arange(psdps.shape[0]) / psdps.shape[0]
# ta = np.arange(tsdps.shape[0]) / tsdps.shape[0]
# plt.plot(pa,psdps,'b-',label='polyhedral, face-norm . f-c line')
# plt.plot(ta,tsdps,'r-',label='triangular, face-norm . f-c line')
# plt.plot(pa,p2sdps,'g-',label='polyhedral, face-norm . c-c line')
# plt.plot(ta,t2sdps,'y-',label='triangular, face-norm . c-c line')
# plt.legend(loc='best')
# plt.xlabel('normalised number of points')
# plt.ylabel('|u.v|')
# plt.show(block=False)



