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
             [*vs[:,1],vs[0,1]],ml,c=lc,lw=lw)

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

# fname = '../tdisk_mid_bc.su2'        # <--
# fname = 'tdisk_poly.su2'        # <--

# fname = '../tdisk_verymid_bc.su2'        # <--
fname = 'tdisk_poly2.su2'        # <--

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

class Cell:
    def __init__(self,idx,verts,centre=None):
        self.idx = idx
        self.verts = verts
        self.faces = []
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
        self.isBoundary = True
        self.boundaryIdx = None
        self.centre = None
        self.norm = None

cells = []
faces = []

hm = {}
ps = []
for i in range(1,7):
    ps.append(10**(i*5))

if dim != 2:
    print('error: cells / faces not supported in 2d')

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

    for k in range(n):

        v1 = elem[k]
        v2 = elem[(k+1)%n]

        v1,v2 = sorted((v1,v2))

        key = v1 + v2 * ps[1]

        val = hm.get(key)

        if val is None:
            faces.append(Face(iface,[v1,v2]))
            hm[key] = iface
            iface += 1
            faces[-1].cells.append(cells[i])
            cells[i].faces.append(faces[-1])
        else:
            faces[val].cells.append(cells[i])
            faces[val].isBoundary = False
            cells[i].faces.append(faces[val])
            hm.pop(key)


for cell in cells:
    cell.ring = []

adaptiveRingSize = False
if adaptiveRingSize:
    minimumRingSize = 9     # for tri
    minimumRingSize = 6     # for poly
numRings = 1            # for poly
# numRings = 2            # for tri
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
            visitThisRing.pop(0)
        ringsDone += 1
        visitThisRing = visitNextRing
        visitNextRing = []
        if adaptiveRingSize:
            if (len(cell.ring) < minimumRingSize) and ringsDone > localNumRings:
                localNumRings += 1

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

    visPoly(points[cells[ci].verts],lc=mcc[8])
    plt.show(block=False)


values = np.zeros((len(cells),2))

if not tris:
    npolyts = 0
    for c in cells:
        npolyts += len(c.faces)

    tpoints = points.copy()
    for c in cells:
        tpoints = np.vstack((tpoints,c.centre))

    tvalues = np.zeros((npolyts,values.shape[1]))
    tconn = np.zeros((npolyts,4),dtype=np.int64)
    j = 0
    for i,c in enumerate(cells):
        for f in c.faces:
            tconn[j,0] = 3
            tconn[j,1] = points.shape[0] + i
            tconn[j,2] = f.verts[0]
            tconn[j,3] = f.verts[1]
            tvalues[j,:] = values[i,:]
            j += 1

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
    f.norm = np.array([p1[1]-p2[1], p2[0]-p1[0]])
    if f.isBoundary:
        boundaryFaces.append(f)
        f.boundaryIdx = i
        i += 1

nBoundaryFaces = len(boundaryFaces)


if tris:
    # TRIANGLES ONLY!
    triang = mtri.Triangulation(points[:,0], points[:,1], connectivity[:,1:])

cmap = cm.coolwarm
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap = colors.LinearSegmentedColormap.from_list(
        'ccmap',cmaplist,cmap.N)
# bounds = np.linspace(-0.5,0.5,13)
bounds = np.linspace(-10,1,100)
norm = colors.BoundaryNorm(bounds,cmap.N)

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
    tpc = ax.tripcolor(points[:,0],points[:,1],connectivity[:,1:],
            facecolors=values[:,0],cmap=cmap,norm=norm)
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
    tpc = ax.tripcolor(points[:,0],points[:,1],connectivity[:,1:],
            facecolors=values[:,1],cmap=cmap,norm=norm)
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

    tvalues = np.zeros((npolyts,2))
    j = 0
    for i,c in enumerate(cells):
        for f in c.faces:
            tvalues[j,:] = values[i,:]
            j += 1

    fig = plt.figure(111,figsize=(20,20))
    if plotEdges:
        for f in faces:
            plt.plot(points[f.verts][:,0],points[f.verts][:,1], 'w-',lw=0.3)
    plt.title(r'$log$ of error of $\partial f / \partial x$',y=1.0,pad=-14)
    ax = plt.gca()
    tpc = ax.tripcolor(tpoints[:,0],tpoints[:,1],tconn[:,1:],
            facecolors=tvalues[:,0],cmap=cmap,norm=norm)
    # fig.colorbar(tpc)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    # plt.savefig('err_dfdx_poly.png',dpi=400)
    plt.show(block=False)

    fig = plt.figure(112,figsize=(20,20))
    if plotEdges:
        for f in faces:
            plt.plot(points[f.verts][:,0],points[f.verts][:,1], 'w-',lw=0.3)
    plt.title(r'$log$ of error of $\partial f / \partial y$',y=1.0,pad=-14)
    ax = plt.gca()
    tpc = ax.tripcolor(tpoints[:,0],tpoints[:,1],tconn[:,1:],
            facecolors=tvalues[:,1],cmap=cmap,norm=norm)
    # fig.colorbar(tpc)
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
            facecolors=values[:,-1],cmap=cm.coolwarm)
    # plt.triplot(triang,'ko-',markersize=2)
    fig.colorbar(tpc)
    plt.show(block=False)

# recasting polyhedral as tris for plotting

else:

    fig = plt.figure()
    ax = plt.gca()
    tpc = ax.tripcolor(tpoints[:,0],tpoints[:,1],tconn[:,1:],
            facecolors=tvalues[:,-1],cmap=cm.coolwarm)
    # plt.triplot(triang,'ko-',markersize=2)
    fig.colorbar(tpc)
    plt.show(block=False)


########################################################################
# Agglomeration
########################################################################

# try colouring

# get neighbour list for each cell
for c in cells:
    c.neighbours = []
    for f in c.faces:
        for nc in f.cells:
            if nc.idx != c.idx:
                c.neighbours.append(nc)

# exclude bc cells from this
intCells = []
bcCells = []
for c in cells:
    c.bcCell = False
    for f in c.faces:
        if len(f.cells) == 1:
            c.bcCell = True
    if c.bcCell:
        bcCells.append(c)
    else:
        intCells.append(c)

colours = np.zeros((nElems,1),dtype=np.int32)
validColours = np.zeros((10),dtype=np.int32)
# colouring
for c in cells:
    if c.bcCell:
        continue
    validColours[:] = 1
    for nc in c.neighbours:
        # exclude bc cells
        if nc.bcCell:
            continue
        # if neighbouring cell uses a colour, not a valid colour
        if colours[nc.idx]: # i.e. non-zero, is coloured
            validColours[colours[nc.idx]] = 0
    for i in range(1,validColours.shape[0]):
        if validColours[i]:
            colours[c.idx] = i
            break


if not tris:
    tcolours = np.zeros((npolyts,values.shape[1]))
    j = 0
    for i,c in enumerate(cells):
        for f in c.faces:
            tcolours[j,:] = colours[i,:]
            j += 1
    #
    fig = plt.figure()
    ax = plt.gca()
    tpc = ax.tripcolor(tpoints[:,0],tpoints[:,1],tconn[:,1:],
            facecolors=tcolours[:,-1],cmap=cm.tab20c)
    # plt.triplot(triang,'ko-',markersize=2)
    fig.colorbar(tpc)
    plt.show(block=False)

ncolors = np.unique(colours).shape[0]

# agglomerate cells starting with lowest colour
agglomerated = np.zeros((nElems,1),dtype=np.int32) - 1
# target number to be joined
targetN = 2
for ic in range(1,ncolors): # ignoring boundaries, coloured 0
    for c in cells:
        # skip for now if not current colour
        if colours[c.idx] != ic:
            continue
        # ignore if boundary cell (shouldn't happen)
        if c.bcCell:
            continue
        # cell has already been agglomerated
        if agglomerated[c.idx] >= 0:
            continue
        # not processed yet - this cell is main, others join
        agglomerated[c.idx] = c.idx
        # look for available neighbours and record their colours
        avail = []
        acols = []
        for nc in c.neighbours:
            if nc.bcCell:
                continue
            if agglomerated[nc.idx] < 0:
                avail.append(nc.idx)
                acols.append(colours[nc.idx,0])
        # if none found, nothing to do for now
        if len(avail) == 0:
            continue
        # sort by colours as want to add highest colour first
        acols,avail = list(zip(*sorted(list(zip(acols,avail)))))
        numConnected = 1
        for nci in avail[::-1]:
            agglomerated[nci] = c.idx
            numConnected += 1
            if numConnected >= targetN:
                break

# count number of cells per new group
aggFreqs = np.zeros((nElems,1),dtype=np.int32)
for i in range(aggFreqs.shape[0]):
    aggFreqs[i] = agglomerated[agglomerated==agglomerated[i]].shape[0]

tooSmall = np.min(aggFreqs) < targetN

# aggCpy = agglomerated.copy()
# agglomerated = aggCpy.copy()
reps = 0
while tooSmall and reps < 4:
    # increase those that are less than target by going above target
    for i in range(aggFreqs.shape[0]):
        c = cells[i]
        ac = cells[agglomerated[c.idx,0]]
        if aggFreqs[ac.idx] < targetN:
            avail = []
            aNums = []
            for nc in ac.neighbours:
                # don't add boundary cells
                if nc.bcCell:
                    continue
                # don't re-add cells already added
                if agglomerated[nc.idx] == ac.idx:
                    continue
                avail.append(nc.idx)
                aNums.append(agglomerated[nc.idx,0])
            aNums,avail = list(zip(*sorted(list(zip(aNums,avail)))))
            connected = np.arange(agglomerated.shape[0])[
                    agglomerated[:,0]==ac.idx].tolist()
            # add the next smallest
            for ic in connected:
                agglomerated[ic] = avail[0]

    # count number of cells per new group
    aggFreqs = np.zeros((nElems,1),dtype=np.int32)
    for i in range(aggFreqs.shape[0]):
        aggFreqs[i] = agglomerated[agglomerated==agglomerated[i]].shape[0]

    tooSmall = np.min(aggFreqs) < targetN
    print('number of cells no agglomerated to target size: ', 
            np.count_nonzero(aggFreqs < targetN))

    reps += 1

if not tris:
    tagglomerated = np.zeros((npolyts,values.shape[1]))
    j = 0
    for i,c in enumerate(cells):
        for f in c.faces:
            tagglomerated[j,:] = agglomerated[i,:]
            j += 1
    # ordered
    taggFreqs = np.zeros((npolyts,values.shape[1]))
    j = 0
    for i,c in enumerate(cells):
        for f in c.faces:
            taggFreqs[j,:] = aggFreqs[i,:]
            j += 1
    taggFreqs[taggFreqs>20] = 0
    #
    fig = plt.figure(3)
    ax = plt.gca()
    # tpc = ax.tripcolor(tpoints[:,0],tpoints[:,1],tconn[:,1:],
            # facecolors=tagglomerated[:,-1],cmap=cm.tab20c)
    # tpc = ax.tripcolor(tpoints[:,0],tpoints[:,1],tconn[:,1:],
            # facecolors=tagglomerated[:,-1],cmap=cm.coolwarm)
    tpc = ax.tripcolor(tpoints[:,0],tpoints[:,1],tconn[:,1:],
            facecolors=taggFreqs[:,-1],cmap=cm.coolwarm)
    # plt.triplot(triang,'ko-',markersize=2)
    fig.colorbar(tpc)
    plt.show(block=False)

# uniqueF,countsF = np.unique(agglomeratedF,return_counts=True)
# uniqueB,countsB = np.unique(agglomeratedB,return_counts=True)

# print('forward:')
# for i in range(1,10):
    # print('number with {} cells: {}'.format(i,np.count_nonzero(
        # countsF[countsF==i])))

# print('backward:')
# for i in range(1,10):
    # print('number with {} cells: {}'.format(i,np.count_nonzero(
        # countsB[countsB==i])))

# check normals to see if all poly verts are clockwise / counterclockwise
pnorms = np.zeros((nElems,1))   # only have z-component
for c in cells:
    v1 = points[c.verts[1]] - points[c.verts[0]]
    v2 = points[c.verts[2]] - points[c.verts[0]]
    pnorms[c.idx] = v1[0]*v2[1] - v1[1]*v2[0]

# all are positive, so all poly are ordered counter-clockwise
# eg. (1,2,3,4,5,6), (8,9,10,4,3,7)
# with joining edge being 3-4, which is backwards in the other poly
# so, loop first poly edges forwards and second poly edges backwards,
# and if any matches split each and join
# split first:  (1,2,3),(4,5,6)
# split second: (8,9,10,4),(3,7)
# join: (1,2,3),(3,7),(8,9,10,4),(4,5,6)    # dupes removed after
#
# edge cases:
#
# eg. (1,2,3,4,5,6), (7,8,9,10,4,3)
# split first:  (1,2,3),(4,5,6)
# split second: (7,8,9,10,4),(3)
# join: (1,2,3),(3),(7,8,9,10,4),(4,5,6)      # dupes removed after
#
# eg. (1,2,3,4,5,6), (3,7,8,9,10,4)
# split first:  (1,2,3),(4,5,6)
# split second: (3,7,8,9,10,4)
# join: (1,2,3),(3,7,8,9,10,4),(4,5,6)      # dupes removed after

def genConn(lcells):
    if len(lcells) == 0:
        print('error, failed')
        return
    if len(lcells) == 1:
        return lcells[0].verts
    # vert indices of first cells
    lverts = lcells[0].verts.tolist()
    # now loop other cells and insert indices where appropriate
    for c in lcells[1:]:
        mpair = None
        cverts = c.verts.tolist()
        # all vertices are ordered counter-clockwise, so check one backwards
        for i in range(len(lverts)-1):
            for j in range(len(cverts)-1):
                if (cverts[j+1],cverts[j]) == (lverts[i],lverts[i+1]):
                    if mpair is None:
                        mpair = i,j
                    else:
                        print('genConn failed: more than one matched edge!')
                        print('1: ',mpair[0],',',mpair[1])
                        print('2: ',i,',',j)
                        return []
        # check if end points of lverts are a match
        if mpair is None:
            for j in range(len(cverts)-1):
                if (cverts[j+1],cverts[j]) == (lverts[-1],lverts[0]):
                    if mpair is None:
                        mpair = len(lverts)-1,j
                    else:
                        print('genConn failed: more than one matched edge!')
                        print('1: ',mpair[0],',',mpair[1])
                        print('2: ',len(lverts)-1,j)
                        return []
        # check if end points of cverts are a match
        if mpair is None:
            for i in range(len(lverts)-1):
                if (cverts[0],cverts[-1]) == (lverts[i],lverts[i+1]):
                    if mpair is None:
                        mpair = i,len(cverts)-1
                    else:
                        print('genConn failed: more than one matched edge!')
                        print('1: ',mpair[0],',',mpair[1])
                        print('2: ',i,len(cverts)-1)
                        return []
        #
        if mpair is None:
            print('genConn failed: no matched edge found')
            print('lverts: ', lverts)
            print('cverts: ', cverts)
            return lverts
        i,j = mpair
        # print(i,j)
        # print(lverts)
        # now split the first list
        l1 = lverts[:i+1]
        l2 = lverts[i+1:]
        c1 = cverts[:j+1]
        c2 = cverts[j+1:]
        # print('l1: ', l1)
        # print('l2: ', l2)
        # print('c1: ', c1)
        # print('c2: ', c2)
        # print("NOT SPLITTING PROPERLY!!!! DRAW IT")
    res = l1
    res.extend(c2)
    res.extend(c1)
    res.extend(l2)
    for i in range(len(res)-1,0,-1):
        if res[i] == res[i-1]:
            res.pop(i)
    # print('res:',res)
    # return l1,l2,c1,c2,res
    return res

'''
a = np.array([1,2,3,4,5,6])
b = np.array([8,9,10,4,3,7])
Ca = Cell(0,a)
Cb = Cell(0,b)
print(a)
print(b)
res = genConn([Ca,Cb])
print()

a = np.array([1,2,3,4,5,6])
b = np.array([7,8,9,10,4,3])
Ca = Cell(0,a)
Cb = Cell(0,b)
print(a)
print(b)
res = genConn([Ca,Cb])
print()

a = np.array([1,2,3,4,5,6])
b = np.array([3,7,8,9,10,4])
Ca = Cell(0,a)
Cb = Cell(0,b)
print(a)
print(b)
res = genConn([Ca,Cb])
print()
'''

aggCells = []
j = 0
for i in range(agglomerated.shape[0]):
    c = cells[i]
    if agglomerated[i,0] != i:
        continue
    connected = np.arange(agglomerated.shape[0])[
            agglomerated[:,0]==c.idx].tolist()
    newConn = genConn([cells[k] for k in connected])
    aggCells.append(Cell(j,newConn))

plt.figure(1)

if not tris:
    tagglomerated = np.zeros((npolyts,values.shape[1]))
    j = 0
    for i,c in enumerate(cells):
        for f in c.faces:
            tagglomerated[j,:] = agglomerated[i,:]
            j += 1
    #
    ax = plt.gca()
    # tpc = ax.tripcolor(tpoints[:,0],tpoints[:,1],tconn[:,1:],
            # facecolors=tagglomerated[:,-1],cmap=cm.tab20c)
    tpc = ax.tripcolor(tpoints[:,0],tpoints[:,1],tconn[:,1:],
            facecolors=tagglomerated[:,-1],cmap=cm.coolwarm)
    # plt.triplot(triang,'ko-',markersize=2)
    fig.colorbar(tpc)
    # plt.show(block=False)

for ac in aggCells:
    visPoly(points[ac.verts],lc='w',lw=0.3)

plt.show(block=False)
