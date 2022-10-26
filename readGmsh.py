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
# fname = 'tdisk_mid_bc.su2'        # <--
# fname = 'tdisk_verymid_bc.su2'
# fname = 'tdisk_veryfine_bc.su2'

# fname = 'qdisk_coarse_bc.su2'
# fname = 'qdisk_med_bc.su2'
# fname = 'qdisk_fine_bc.su2'
# fname = 'qdisk_veryfine_bc.su2'

fname = 'sTetMesh.su2'

res = loadGmsh(fname,True)

if res == -1:
    print('error: load Gmsh failed')
    exit()

dim,nElems,nPoints,connectivity,points,nMarks,marks,markNames,markNumFaces = res

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

# f = lambda x,y: np.sin(0.1 * abs(x * y)) * 4*np.exp(abs(x * y))
# dfdx = lambda x,y: (x*y**2*np.exp(abs(x*y))*(4*np.sin(0.1*abs(x*y))
        # + 0.4*np.cos(0.1*abs(x*y))))/abs(x*y)
# dfdy = lambda x,y: (y*x**2*np.exp(abs(x*y))*(4*np.sin(0.1*abs(x*y))
        # + 0.4*np.cos(0.1*abs(x*y))))/abs(x*y)

func = lambda x,y: 4.0*x + 5.0*x*x + 4.0*y + 5.0*y*y + 2*x*y
dfdx = lambda x,y: 4.0 + 10.0*x + 2*y
dfdy = lambda x,y: 4.0 + 10.0*y + 2*x

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

adaptiveRingSize = False
minimumRingSize = 9
numRings = 2
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

if bigRings.shape[0] > 0:
    ci = 17
    plt.figure(1)
    s = 10
    for ci in bigRings[::s]:
        for c in cells[ci].ring:
            visPoly(points[c.verts],lc=mcc[2])

    for ci in bigRings[::s]:
        visPoly(points[cells[ci].verts],lc=mcc[8])

    plt.show()



'''
for c in ch:
    visPoint(cells[c].centre)

for f in cells[i].faces:
    visPoly(points[f.verts],lc=mcc[8])

plt.show(block=False)
'''


# figuring out which are some edge triangles
# visPoly(points[cells[ 8].verts],lc='k')
# visPoly(points[cells[ 9].verts],lc='r')
# visPoly(points[cells[11].verts],lc='b')#
# visPoly(points[cells[12].verts],lc='m')
# plt.show(block=False)


if False:
    # TRIANGLES ONLY!
    triang = mtri.Triangulation(points[:,0], points[:,1], connectivity[:,1:])

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
    if f.isBoundary:
        boundaryFaces.append(f)
        f.boundaryIdx = i
        f.centre = np.sum(points[f.verts],0)/2.0
        i += 1

nBoundaryFaces = len(boundaryFaces)

fFields = np.zeros((nBoundaryFaces,1))
for f in boundaryFaces:
    fFields[f.boundaryIdx,0] = func(*f.centre)

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


errGradF = np.zeros((len(cells),2))
for c in cells:
    errGradF[c.idx,0] = c.x[0] - dfdx(*c.centre)
    errGradF[c.idx,1] = c.x[1] - dfdy(*c.centre)

errL2dfdx = (np.sum(errGradF[:,0]**2.) / nElems)**0.5
errL2dfdy = (np.sum(errGradF[:,1]**2.) / nElems)**0.5

if False:
    # fp = open('res.dat', 'a')
    # fp = open('tmp.dat', 'a')
    fp = open('tmp2.dat', 'a')
    fp.write('number of elements: {}\n'.format(nElems))
    fp.write('type of elements: {}\n'.format(np.unique(connectivity[:,0])))
    fp.write('number of face rings: {}\n'.format(numRings))
    fp.write('min, max, avg of dfdx: {:.10f}, {:.10f}, {:.10f}\n'.format(
            np.min(errGradF[:,0]),
            np.max(errGradF[:,0]),
            np.sum(errGradF[:,0]) / nElems))
    fp.write('min, max, avg of dfdy: {:.10f}, {:.10f}, {:.10f}\n'.format(
            np.min(errGradF[:,1]),
            np.max(errGradF[:,1]),
            np.sum(errGradF[:,1]) / nElems))
    fp.write('L2 dfdx, dfdy: {:.10f}, {:.10f}\n\n'.format(
        errL2dfdx,errL2dfdy))
    fp.close()

else:
    print('number of elements: {}'.format(nElems))
    print('type of elements: {}'.format(np.unique(connectivity[:,0])))
    print('number of face rings: {}'.format(numRings))
    print('min, max, avg of dfdx: {:.10f}, {:.10f}, {:.10f}'.format(
            np.min(errGradF[:,0]),
            np.max(errGradF[:,0]),
            np.sum(errGradF[:,0]) / nElems))
    print('min, max, avg of dfdy: {:.10f}, {:.10f}, {:.10f}'.format(
            np.min(errGradF[:,1]),
            np.max(errGradF[:,1]),
            np.sum(errGradF[:,1]) / nElems))
    print('L2 dfdx, dfdy: {:.10f}, {:.10f}'.format(
        errL2dfdx,errL2dfdy))

# TRIANGLES ONLY!
triang = mtri.Triangulation(points[:,0], points[:,1], connectivity[:,1:])

cmap = cm.coolwarm
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap = colors.LinearSegmentedColormap.from_list(
        'ccmap',cmaplist,cmap.N)
bounds = np.linspace(-0.5,0.5,13)
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

fig = plt.figure(111,figsize=(20,20))
plt.title(r'$log$ of error of $\partial f / \partial x$',y=1.0,pad=-14)
ax = plt.gca()
# tpc = ax.tripcolor(points[:,0],points[:,1],connectivity[:,1:],
        # facecolors=errGradF[:,0],cmap=cm.coolwarm,vmin=vi,vmax=va)
# tpc = ax.tripcolor(points[:,0],points[:,1],connectivity[:,1:],
        # facecolors=errGradF[:,0],cmap=cmap)#,norm=norm)
tpc = ax.tripcolor(points[:,0],points[:,1],connectivity[:,1:],
        facecolors=np.log(np.abs(errGradF[:,0])),cmap=cmap)#,norm=norm)
# tpc = ax.tripcolor(triang,vertErrs[:,0],cmap=cmap,#vmin=vi,vmax=va,
        # norm=norm,shading='flat')
# tpc = ax.tripcolor(triang,vertErrs[:,0],cmap=cmap,#vmin=vi,vmax=va,
        # norm=norm,shading='gouraud')
# cb = plt.colorbar.ColorbarBase(ax,cmap=cmap,norm=norm,ticks=bounds,
        # boundaries=bounds)
# cb = plt.colorbar.ColorbarBase(ax,cmap=cmap,norm=norm,ticks=bounds,
        # boundaries=bounds)
# plt.triplot(triang,'w-',lw=0.1)
fig.colorbar(tpc)
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
plt.title(r'$log$ of error of $\partial f / \partial y$',y=1.0,pad=-14)
# plt.title(r'$\frac{\partial f}{\partial y}$',y=1.0,pad=-14)
ax = plt.gca()
# tpc = ax.tripcolor(points[:,0],points[:,1],connectivity[:,1:],
        # facecolors=errGradF[:,1],cmap=cm.coolwarm,vmin=vi,vmax=va)
# tpc = ax.tripcolor(points[:,0],points[:,1],connectivity[:,1:],
        # facecolors=errGradF[:,1],cmap=cmap)#,norm=norm)
tpc = ax.tripcolor(points[:,0],points[:,1],connectivity[:,1:],
        facecolors=np.log(np.abs(errGradF[:,1])),cmap=cmap)#,norm=norm)
# cb = plt.colorbar.ColorbarBase(ax,cmap=cmap,norm=norm,ticks=bounds,
        # boundaries=bounds)
# plt.triplot(triang,'w-',lw=0.1)
fig.colorbar(tpc)
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
# plt.savefig('err_dfdy_4.png',dpi=400)
plt.show(block=False)

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
            #
            gradfnx = dfdx(*f.centre) * nx
            gradfny = dfdy(*f.centre) * ny


ax = plt.gca()
tpc = ax.tripcolor(points[:,0],points[:,1],connectivity[:,1:],
        facecolors=fields[:,-1],cmap=cm.coolwarm)
# plt.triplot(triang,'ko-',markersize=2)
fig.colorbar(tpc)
plt.show(block=False)





tNs = [117,468,1872,7488]
qNs = [66,234,935,3743]

# [2-ring x, 2-ring y, 3-ring x, 3-ring y]
tL2s = np.array([
    [0.0881270506, 0.0884061194, 0.1147105438, 0.1146847626],
    [0.0509759085, 0.0564668092, 0.0639101668, 0.0687018920],
    [0.0338681162, 0.0364066731, 0.0417259482, 0.0435182344],
    [0.0234117126, 0.0249487888, 0.0282671254, 0.0294458627],
        ])
# [2-ring x, 2-ring y, 3-ring x, 3-ring y]
qL2s = np.array([
    [0.1300284042, 0.1374803489, 0.1806276262, 0.1795262481],
    [0.0742162578, 0.0766044630, 0.0974671968, 0.0967429109],
    [0.0497103603, 0.0513003805, 0.0608095905, 0.0623536455],
    [0.0336120199, 0.0342614834, 0.0404498287, 0.0410566144],
        ])

'''
plt.figure(13)
plt.loglog(tNs,tL2s[:,0],'o-',c=mcc[0],mfc='none',label=' tri,dfdx,2-ring')
plt.loglog(tNs,tL2s[:,1],'s-',c=mcc[1],mfc='none',label=' tri,dfdy,2-ring')
plt.loglog(qNs,qL2s[:,0],'o-',c=mcc[8],mfc='none',label='quad,dfdx,2-ring')
plt.loglog(qNs,qL2s[:,1],'s-',c=mcc[9],mfc='none',label='quad,dfdy,2-ring')
plt.loglog(tNs,tL2s[:,2],'o-',c=mcc[0],label=' tri,dfdx,3-ring')
plt.loglog(tNs,tL2s[:,3],'s-',c=mcc[1],label=' tri,dfdy,3-ring')
plt.loglog(qNs,qL2s[:,2],'o-',c=mcc[8],label='quad,dfdx,3-ring')
plt.loglog(qNs,qL2s[:,3],'s-',c=mcc[9],label='quad,dfdy,3-ring')
plt.legend(loc='best')
plt.show(block=False)

tooas = np.zeros((len(tNs)-1,4))
qooas = np.zeros((len(qNs)-1,4))
for i in range(len(tNs)-1):
    tr = (tNs[i+1]/tNs[i])**(1.0 / dim)
    tooas[i,0] = np.log(tL2s[i+1,0]/tL2s[i,0]) / np.log(tr)
    tooas[i,1] = np.log(tL2s[i+1,1]/tL2s[i,1]) / np.log(tr)
    tooas[i,2] = np.log(tL2s[i+1,2]/tL2s[i,2]) / np.log(tr)
    tooas[i,3] = np.log(tL2s[i+1,3]/tL2s[i,3]) / np.log(tr)

for i in range(len(qNs)-1):
    qr = (qNs[i+1]/qNs[i])**(1.0 / dim)
    qooas[i,0] = np.log(qL2s[i+1,0]/qL2s[i,0]) / np.log(qr)
    qooas[i,1] = np.log(qL2s[i+1,1]/qL2s[i,1]) / np.log(qr)
    qooas[i,2] = np.log(qL2s[i+1,2]/qL2s[i,2]) / np.log(qr)
    qooas[i,3] = np.log(qL2s[i+1,3]/qL2s[i,3]) / np.log(qr)

plt.figure(14)
plt.plot(tNs[1:],tooas[:,0],'o-',c=mcc[0],mfc='none',label=' tri,dfdx,2-ring')
plt.plot(tNs[1:],tooas[:,1],'s-',c=mcc[1],mfc='none',label=' tri,dfdy,2-ring')
plt.plot(qNs[1:],qooas[:,0],'o-',c=mcc[8],mfc='none',label='quad,dfdx,2-ring')
plt.plot(qNs[1:],qooas[:,1],'s-',c=mcc[9],mfc='none',label='quad,dfdy,2-ring')
plt.plot(tNs[1:],tooas[:,2],'o-',c=mcc[0],label=' tri,dfdx,3-ring')
plt.plot(tNs[1:],tooas[:,3],'s-',c=mcc[1],label=' tri,dfdy,3-ring')
plt.plot(qNs[1:],qooas[:,2],'o-',c=mcc[8],label='quad,dfdx,3-ring')
plt.plot(qNs[1:],qooas[:,3],'s-',c=mcc[9],label='quad,dfdy,3-ring')
plt.legend(loc='best')
plt.show(block=False)
'''

# ======================================================

# with hard constraint

tNs = [
     117,
     468,
    1872,
    2836,
    6028,
    7488,
    ]

tL2s = np.array([
    [0.3985302546, 0.4035986315],
    [0.1503884617, 0.1516778791],
    [0.0587889428, 0.0592583017],
    [0.0800550351, 0.0776482036],
    [0.0404355719, 0.0402355751],
    [0.0217968515, 0.0219391942],
    ])

'''
plt.figure(15)
plt.loglog(tNs,tL2s[:,0],'o-',c=mcc[0],mfc='none',label=' tri,dfdx,2-ring')
plt.loglog(tNs,tL2s[:,1],'s-',c=mcc[1],mfc='none',label=' tri,dfdy,2-ring')
plt.legend(loc='best')
plt.show(block=False)

tooas = np.zeros((len(tNs)-1,4))
for i in range(len(tNs)-1):
    tr = (tNs[i+1]/tNs[i])**(1.0 / dim)
    tooas[i,0] = np.log(tL2s[i+1,0]/tL2s[i,0]) / np.log(tr)
    tooas[i,1] = np.log(tL2s[i+1,1]/tL2s[i,1]) / np.log(tr)

plt.figure(16)
plt.plot(tNs[1:],tooas[:,0],'o-',c=mcc[0],mfc='none',label=' tri,dfdx,2-ring')
plt.plot(tNs[1:],tooas[:,1],'s-',c=mcc[1],mfc='none',label=' tri,dfdy,2-ring')
plt.legend(loc='best')
plt.show(block=False)
'''
