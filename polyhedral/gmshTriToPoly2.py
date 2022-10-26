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
# fname = '../tdisk_mid_bc.su2'        # <--
fname = '../tdisk_verymid_bc.su2'
# fname = 'tdisk_veryfine_bc.su2'

# fname = 'qdisk_coarse_bc.su2'
# fname = 'qdisk_med_bc.su2'
# fname = 'qdisk_fine_bc.su2'
# fname = 'qdisk_veryfine_bc.su2'

# fname = 'sTetMesh.su2'

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

# calculating face centroids and areas
faceAreas = np.zeros((ncells,1))
faceCentroids = np.zeros((ncells,3))
for i,face in enumerate(faces):
    faceAreas[i] = area(face)
    faceCentroids[i] = centroid(face)

print("building dual mesh")
pctrs = np.copy(faceCentroids)
pes = []
for i,face in enumerate(faces):
    for j in range(len(face.hes)):
        he = face.hes[j]
        if he.dual is None:
            if he.twin is None:
                # make edge from face centre to bndry edge midpoint
                pes.append(HalfEdge(faceCentroids[he.face.id],
                    he.face.id,len(pctrs)))
                first = len(pes)-1
                he.dual = pes[-1]
                # make edge from bndry edge midpoint to vertex
                # need to check if midpoint already added...
                fid = -1
                mp = midpoint(he)
                for ii,pc in enumerate(pctrs):
                    if isClose(pc,mp):
                        fid = ii
                        nxtp = pc
                        break
                if fid < 0:
                    pctrs = np.vstack((pctrs, mp))
                    fid = len(pctrs)-1
                    nxtp = pctrs[-1]
                else:
                    pes[-1].p1 = fid
                # setting halfedge.p1 as expected next insert, but it might not be!
                # will fix up below if the next vertex already exists
                pes.append(HalfEdge(nxtp,
                    fid,len(pctrs)))
                pes[-2].next = pes[-1]
                # need to iterate around vertex until next bndry
                # edge is found
                while he.next.twin is not None:
                    he = he.next.twin
                he = he.next
                # can either go from midpoint-midpoint or midpoint-vertex-midpoint
                fid = -1
                if True:
                    # make edge from midpoint-midpoint
                    # need to check if midpoint already added...
                    mp = midpoint(he)
                    for ii,pc in enumerate(pctrs):
                        if isClose(pc,mp):
                            fid = ii
                            nxtp = pc
                            break
                    if fid < 0:
                        pctrs = np.vstack((pctrs, mp))
                        fid = len(pctrs)-1
                        nxtp = pctrs[-1]
                    else:
                        pes[-1].p1 = fid
                else:
                    # make edge from vertex to next bndry edge midpoint
                    pctrs = np.vstack((pctrs, he.vertex))
                    pctrs = np.vstack((pctrs, midpoint(he)))
                    pes.append(HalfEdge(pctrs[-2],
                        len(pctrs)-2,len(pctrs)-1))
                    pes[-2].next = pes[-1]
                # make edge from bndry edge midpoint to face centre
                pes.append(HalfEdge(nxtp,
                    fid,he.face.id))
                pes[-2].next = pes[-1]
                he.dual = pes[-1]

                he = prev(he)

                # now need to keep iterating until we reach
                #  the edge we started at
                while he != face.hes[j]:
                    # should be valid otherwise geom bad
                    pes.append(HalfEdge(faceCentroids[he.face.id],
                        he.face.id, he.twin.face.id))
                    pes[-2].next = pes[-1]
                    he.dual = pes[-1]
                    he = prev(he.twin)

                pes[-1].next = pes[first]

            else:

                # already checked this twin, exists

                # now iterate around vertex and make new face 
                pes.append(HalfEdge(faceCentroids[i],
                        face.id, he.twin.face.id))
                first = len(pes)-1
                he.dual = pes[-1]

                he = prev(he.twin)

                valid = True
                reachedBndry = False
                while valid:
                    if he.twin is None:
                        # handle bndry later
                        reachedBndry = True
                        valid = False
                        continue

                    # otherwise twin is valid

                    # now need to keep iterating until we reach
                    #  the edge we started at
                    if he == face.hes[j]:
                        valid = False
                        continue

                    # should be valid otherwise geom bad
                    pes.append(HalfEdge(faceCentroids[he.face.id],
                        he.face.id, he.twin.face.id))
                    pes[-2].next = pes[-1]
                    he.dual = pes[-1]
                    he = prev(he.twin)

                if not reachedBndry:
                    pes[-1].next = pes[first]

                if reachedBndry:
                    # make edge from face centre to bndry edge midpoint
                    pes.append(HalfEdge(faceCentroids[he.face.id],
                        he.face.id,len(pctrs)))
                    pes[-2].next = pes[-1]
                    # first already set
                    he.dual = pes[-1]
                    # make edge from bndry edge midpoint to vertex
                    # need to check if midpoint already added...
                    fid = -1
                    mp = midpoint(he)
                    for ii,pc in enumerate(pctrs):
                        if isClose(pc,mp):
                            fid = ii
                            nxtp = pc
                            break
                    if fid < 0:
                        pctrs = np.vstack((pctrs, mp))
                        fid = len(pctrs)-1
                        nxtp = pctrs[-1]
                    else:
                        pes[-1].p1 = fid
                    # setting halfedge.p1 as expected next insert, but it might not be!
                    # will fix up below if the next vertex already exists
                    pes.append(HalfEdge(nxtp,
                        fid,len(pctrs)))
                    pes[-2].next = pes[-1]
                    # need to iterate around vertex until next bndry
                    # edge is found
                    while he.next.twin is not None:
                        he = he.next.twin
                    he = he.next
                    # can either go from midpoint-midpoint or midpoint-vertex-midpoint
                    fid = -1
                    if True:
                        # make edge from midpoint-midpoint
                        # need to check if midpoint already added...
                        mp = midpoint(he)
                        for ii,pc in enumerate(pctrs):
                            if isClose(pc,mp):
                                fid = ii
                                nxtp = pc
                                break
                        if fid < 0:
                            pctrs = np.vstack((pctrs, mp))
                            fid = len(pctrs)-1
                            nxtp = pctrs[-1]
                        else:
                            pes[-1].p1 = fid
                    else:
                        # make edge from vertex to next bndry edge midpoint
                        pctrs = np.vstack((pctrs, he.vertex))
                        pctrs = np.vstack((pctrs, midpoint(he)))
                        pes.append(HalfEdge(pctrs[-2],
                            len(pctrs)-2,len(pctrs)-1))
                        pes[-2].next = pes[-1]
                    # make edge from bndry edge midpoint to face centre
                    pes.append(HalfEdge(nxtp,
                        fid,he.face.id))
                    pes[-2].next = pes[-1]
                    he.dual = pes[-1]

                    he = prev(he)

                    # now need to keep iterating until we reach
                    #  the edge we started at
                    while he != face.hes[j]:
                        # should be valid otherwise geom bad
                        pes.append(HalfEdge(faceCentroids[he.face.id],
                            he.face.id, he.twin.face.id))
                        pes[-2].next = pes[-1]
                        he.dual = pes[-1]
                        he = prev(he.twin)

                    pes[-1].next = pes[first]

print("making twin connections")

# making twin connection
dups = 0
repeated = []
for i,pei in enumerate(pes):
    for j,pej in enumerate(pes):
        if (i == j):
            continue
        if pei.p0 == pej.p0 and pei.p1 == pej.p1:
            dups += 1
        if pej.twin is not None:
            continue
        if pei.p0 == pej.p1 and pei.p1 == pej.p0:
            pes[i].twin = pes[j]
            pes[j].twin = pes[i]
            break
        if isClose(pctrs[pei.p0],pctrs[pej.p1]) and isClose(pctrs[pei.p1],pctrs[pej.p0]):
            repeated.append((pei.p0,pej.p1))
            repeated.append((pei.p1,pej.p0))


if False:
    for i,pei in enumerate(pes):
        for j,pej in enumerate(pes):
            if (i == j):
                continue
            # if pei.twin is not None or pej.twin is not None:
            if pei.twin is None:
                visualise([pei],'r--')
            if pej.twin is None:
                visualise([pej],'g-.')

pbndry = []
for pe in pes:
    if pe.twin is None:
        pbndry.append(pe)

# making faces
print("making p-faces")
pfaces = []
nextId = 0
for i,pe in enumerate(pes):
    if pe.face is None:
        pfaces.append(Face(nextId))
        nextId += 1
        pi = pe
        pfaces[-1].hes.append(pi)
        pi.face = pfaces[-1]
        pi = pi.next
        while pi != pe:
            pfaces[-1].hes.append(pi)
            pi.face = pfaces[-1]
            pi = pi.next

pncells = len(pfaces)

print("calculating p-face centroids and areas")
# calculating face centroids and areas
pfaceAreas = np.zeros((pncells,1))
pfaceCentroids = np.zeros((pncells,3))
for i,pface in enumerate(pfaces):
    pfaceAreas[i] = area(pface)
    pfaceCentroids[i] = centroid(pface)

print("calculating list of unique edges")
# calculating list of unique edges
pedges = []
counter = 0
for i,pe in enumerate(pes):
    if pe.twin is None:
        pedges.append(Edge(counter))
        counter += 1
        pedges[-1].he = pe
        pe.edge = pedges[-1]
    elif pe.edge is None and pe.twin.edge is None:
        pedges.append(Edge(counter))
        counter += 1
        pedges[-1].he = pe
        pe.edge = pedges[-1]
        pe.twin.edge = pedges[-1]

writeVTK = False
writeE4Grid = False

pnpts = len(pctrs)
pnedges = len(pedges)

if writeVTK:
    fp = open("poly2.vtu","w")
    fp.write('<VTKFile type="UnstructuredGrid" byte_order="BigEndian">\n')
    fp.write('<UnstructuredGrid><Piece NumberOfPoints="{}" NumberOfCells="{}">\n'.format(
        pnpts,pncells))
    fp.write('<Points>\n')

    fp.write(' <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
    for p in pctrs:
        fp.write('{:.16e} {:.16e} {:.16e}\n'.format(p[0],p[1],p[2]))

    fp.write(' </DataArray>\n')
    fp.write('</Points>\n')
    fp.write('<Cells>\n')
    fp.write(' <DataArray type="Int32" Name="connectivity" format="ascii">\n')

    poffsets = []
    i = 0
    for pface in pfaces:
        i += len(pface.hes)
        poffsets.append(i)
        line = ""
        for j in pface.hes:
            line += str(j.p0) + " "
        line += "\n"
        fp.write(line)

    fp.write(' </DataArray>\n')
    fp.write(' <DataArray type="Int32" Name="offsets" format="ascii">\n')

    for po in poffsets:
        fp.write(str(po)+"\n")

    fp.write(' </DataArray>\n')
    fp.write(' <DataArray type="UInt8" Name="types" format="ascii">\n')

    for i in range(pncells):
        fp.write('7\n')

    fp.write(' </DataArray>\n')
    fp.write('</Cells>\n')
    fp.write('<CellData>\n')
    fp.write(' <DataArray Name="pos.x" type="Float32" NumberOfComponents="1" format="ascii">\n')
    for i in range(pncells):
        fp.write("{}\n".format(np.random.rand()))

    fp.write(' </DataArray>\n')
    fp.write('</CellData>\n')
    fp.write('</Piece>\n')
    fp.write('</UnstructuredGrid>\n')
    fp.write('</VTKFile>\n')
    fp.close()


print("calculating outsigns")
nbndry = 0
bndryEdges = []
# set outsigns
for pe in pes:
    if pe.os is not None:
        continue
    if pe.twin is None:
        # boundary
        pe.os = 1
        nbndry += 1
        bndryEdges.append(pe)
    else:
        # compare x, if close compare y
        if abs(pctrs[pe.face.id][0] - pctrs[pe.twin.face.id][0]) < 1e-16:
            if pctrs[pe.face.id][1] < pctrs[pe.twin.face.id][1]:
                pe.os = 1
                pe.twin.os = -1
        else:
            if pctrs[pe.face.id][0] < pctrs[pe.twin.face.id][0]:
                pe.os = 1
                pe.twin.os = -1


def vertexData(face,s=True):
    he = face.hes[0]
    nVtxs = 1
    vs = [he.p0]
    hi = he.next
    nVtxs += 1
    vs.append(hi.p0)
    while hi.next != he:
        hi = hi.next
        nVtxs += 1
        vs.append(hi.p0)
    if s:
        line = str(nVtxs) + " "
        for v in vs:
            line += str(v) + " "
        return line
    else:
        return nVtxs, vs

def vertexData2(face,s=True):
    nVtxs = len(face.hes)
    vs = []
    for he in face.hes:
        vs.append(he.p0)
    if s:
        line = str(nVtxs) + " "
        for v in vs:
            line += str(v) + " "
        return line
    else:
        return nVtxs, vs


def edgeData(face,s=True):
    he = face.hes[0]
    nEdges = 1
    es = [he.edge.id]
    hi = he.next
    nEdges += 1
    es.append(hi.edge.id)
    while hi.next != he:
        hi = hi.next
        nEdges += 1
        es.append(hi.edge.id)
    if s:
        line = str(nEdges) + " "
        for e in es:
            line += str(e) + " "
        return line
    else:
        return nEdges, es

def outsignData(face,s=True):
    he = face.hes[0]
    nSigns = 1
    os = [he.os]
    hi = he.next
    nSigns += 1
    os.append(hi.os)
    while hi.next != he:
        hi = hi.next
        nSigns += 1
        os.append(hi.os)
    if s:
        line = str(nSigns) + " "
        for o in os:
            line += str(o) + " "
        return line
    else:
        return nSigns, os

print("visualising")
if True:
    # for p in pfaces:
        # pctr = pfaceCentroids[p.id]
        # for pi in edgeData(p,False)[1]:
            # mp = midpoint(pedges[pi].he)
            # plt.plot([pctr[0],mp[0]],[pctr[1],mp[1]],'k.-',mfc='none')
    #
    visualise(pes)
    plt.show(block=False)

print('writing poly mesh to gmsh format')
if False:
    fp = open('tdisk_poly2.su2','w')
    fp.write('NDIME= 2\n')
    fp.write('NELEM= {}\n'.format(len(pfaces)))
    for i in range(len(pfaces)):
        fp.write('7 ')
        for h in pfaces[i].hes:
            fp.write('{} '.format(h.p0))
        fp.write('{}\n'.format(i))
    fp.write('NPOIN= {}\n'.format(len(pctrs)))
    for i in range(len(pctrs)):
        fp.write('{} {} {}\n'.format(*pctrs[i,:2],i))
    fp.write('NMARK= 1\n')
    fp.write('MARKER_TAG= WALL\n')
    fp.write('MARKER_ELEMS= {}\n'.format(nbndry))
    for i in range(nbndry):
        fp.write('3 {} {}\n'.format(bndryEdges[i].p0, bndryEdges[i].p1))
    fp.close()

# am missing some connections on the edges!
# print("JAMIE! missing connections on edges")

if False:
    # sanity check on outsigns
    sums = 0
    for p in pfaces:
        o = outsignData(p,False)[1]
        for i in o:
            sums += i
    print(sums)
    print(nbndry)


# for p in pedges[34:49]:
    # visualise([p.he])
    # plt.show(block=False)

# colors = ['b','g','r','c','m','y','k']
# plt.figure(1)
# # for i,p in enumerate(pfaces[:1]):
# for i,p in enumerate(pfaces[:1]):
    # nv,vd = vertexData2(p,False)
    # ne,ed = edgeData(p,False)
    # no,od = outsignData(p,False)
    # c = colors[i%len(colors)]
    # # for ei in ed:
        # # # visualise([pedges[ei].he])
        # # vtx0 = pedges[ei].he.vertex
        # # vtx1 = pedges[ei].he.next.vertex
        # # ctr = centroid(p)
        # # vtx0 = (10*vtx0 + ctr) /11.
        # # vtx1 = (10*vtx1 + ctr) /11.
        # # plt.plot([vtx0[0],vtx1[0]], [vtx0[1],vtx1[1]],c)
        # # plt.plot([vtx0[0],ctr[0]],[vtx0[1],ctr[1]],c)
        # # plt.plot([vtx1[0],ctr[0]],[vtx1[1],ctr[1]],c)
    # visualise(p.hes)
    # for h in p.hes:
        # plt.plot([h.vertex[0]],[h.vertex[1]],'go',mfc='none')
    # for h in p.hes:
        # plt.plot([pctrs[h.p0][0]],[pctrs[h.p0][1]],'rx',mfc='none')
    # for j,vi in enumerate(vd):
        # # vtx0 = pes[vi].vertex
        # # vtx1 = pes[vi].next.vertex
        # vtx0 = pctrs[vi]
        # vtx1 = pctrs[vd[(j+1)%nv]]
        # ctr = centroid(p)
        # # vtx0 = (10*vtx0 + ctr) /11.
        # # vtx1 = (10*vtx1 + ctr) /11.
        # plt.plot([ctr[0]],[ctr[1]],'ko')
        # plt.plot([vtx0[0],vtx1[0]], [vtx0[1],vtx1[1]],'r')
        # # plt.plot([vtx0[0],ctr[0]],[vtx0[1],ctr[1]],'g')
        # # plt.plot([vtx1[0],ctr[0]],[vtx1[1],ctr[1]],'b')


# colors = ['b','g','r','c','m','y','k']
# plt.figure(1)
# for i,p in enumerate(pfaces[:1]):
    # nv,vd = vertexData(p,False)
    # ne,ed = edgeData(p,False)
    # no,od = outsignData(p,False)
    # c = colors[i%len(colors)]
    # for ei in ed:
        # # visualise([pedges[ei].he])
        # vtx0 = pedges[ei].he.vertex
        # vtx1 = pedges[ei].he.next.vertex
        # plt.plot([vtx0[0],vtx1[0]], [vtx0[1],vtx1[1]],c)


### '''
### fig = plt.figure()
### if dim == 3:
###     ax = plt.axes(projection='3d')
###     print('error: 3d plotting not supported')
###     exit()
### 
### for i in range(nElems):
###     elem = connectivity[i,:]
###     n = elem.shape[0] - 1
### 
###     if elem[0] == 5:
###         n = 3
###     elif elem[0] == 9:
###         n = 4
###     elif elem[0] == 3:
###         n = 2
### 
###     visPoly(points[elem[1:n+1]])
### '''

faceWeightedCentroids = True
# faceWeightedCentroids = False

centres = np.zeros((nElems,2))
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
    #
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


### '''
### visPoints(centres,lc=mcc[0],m='x')
### 
### for i in range(nMarks):
###     for j in range(markNumFaces[i]):
###         visPoly(points[marks[i][j,1:]],lc='r')
### '''
### 
### 
### nFields = 3
### fields = np.zeros((nElems,nFields))
### 
### # f = lambda x,y: np.sin(0.1 * abs(x * y)) * 4*np.exp(abs(x * y))
### # dfdx = lambda x,y: (x*y**2*np.exp(abs(x*y))*(4*np.sin(0.1*abs(x*y))
###         # + 0.4*np.cos(0.1*abs(x*y))))/abs(x*y)
### # dfdy = lambda x,y: (y*x**2*np.exp(abs(x*y))*(4*np.sin(0.1*abs(x*y))
###         # + 0.4*np.cos(0.1*abs(x*y))))/abs(x*y)
### 
### func = lambda x,y: 4.0*x + 5.0*x*x + 4.0*y + 5.0*y*y + 2*x*y
### dfdx = lambda x,y: 4.0 + 10.0*x + 2*y
### dfdy = lambda x,y: 4.0 + 10.0*y + 2*x
### 
### # func = lambda x,y: 4.0*x + 4.0*y + 8.0*x*y
### # dfdx = lambda x,y: 4.0 + 8.0*y
### # dfdy = lambda x,y: 4.0 + 8.0*x
### 
### for i in range(nElems):
###     fields[i,0] = i
###     fields[i,1] = i/100.
###     fields[i,2] = func(*centres[i,:])
### 
### '''
### # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
### fig = plt.figure(2, clear=True)
### ax = fig.add_subplot(111, projection='3d')
### 
### vmin = np.min(fields[:,0])
### vmax = np.max(fields[:,0])
### 
### for i in range(nElems):
###     elem = connectivity[i,:]
###     n = elem.shape[0] - 1
### 
###     if elem[0] == 5:
###         n = 3
###     elif elem[0] == 9:
###         n = 4
###     elif elem[0] == 3:
###         n = 2
### 
###     ps = points[elem[1:n+1]]
###     fs = [fields[i,1] for j in range(n)]
###     visPatch(ax, ps[:,0], ps[:,1], fs, vmin=vmin, vmax=vmax)
###     # plotField(points[elem[1:n+1]],fields[i,1])
### 
### ax.set_xlim(-1,1)
### ax.set_ylim(-1,1)
### plt.show(block=False)
### '''
### 
### # if still going down this route, need:
### # - hashmap to identify which elements share faces
### # - element - face - element connectivity

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


### '''
### plt.figure(3)
### 
### i = 8
### 
### ch = [cells[i].idx]
### for f in cells[i].faces:
###     for c in f.cells:
###         if c.idx not in ch:
###             visPoly(points[c.verts])
###             ch.append(c.idx)
###             # print('1-ring:',c.idx)
###             for f2 in c.faces:
###                 for c2 in f2.cells:
###                     if c2.idx not in ch:
###                         visPoly(points[c2.verts],lc='g')
###                         # print('2-ring:',c2.idx)
###                         ch.append(c2.idx)
###                         for f3 in c2.faces:
###                             for c3 in f3.cells:
###                                 if c3.idx not in ch:
###                                     visPoly(points[c3.verts],lc='g')
###                                     # print('3-ring:',c3.idx)
###                                     ch.append(c3.idx)
###                                     # for f4 in c3.faces:
###                                         # for c4 in f4.cells:
###                                             # if c4.idx not in ch:
###                                                 # visPoly(points[c4.verts],lc='g')
###                                                 # print('4-ring:',c4.idx)
###                                                 # ch.append(c4.idx)
### 
### '''

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
            _ = visitThisRing.pop(0)
        ringsDone += 1
        visitThisRing = visitNextRing
        visitNextRing = []
        if adaptiveRingSize:
            if (len(cell.ring) < minimumRingSize) and ringsDone > localNumRings:
                localNumRings += 1

### # cells with large rings
### stens = np.array([len(cell.ring) for cell in cells])
### bigRings = np.arange(stens.shape[0])[stens>9]
### 
### if bigRings.shape[0] > 0:
###     ci = 17
###     plt.figure(1)
###     s = 10
###     for ci in bigRings[::s]:
###         for c in cells[ci].ring:
###             visPoly(points[c.verts],lc=mcc[2])
### 
###     for ci in bigRings[::s]:
###         visPoly(points[cells[ci].verts],lc=mcc[8])
### 
###     plt.show()
### 
### 
### 
### '''
### for c in ch:
###     visPoint(cells[c].centre)
### 
### for f in cells[i].faces:
###     visPoly(points[f.verts],lc=mcc[8])
### 
### plt.show(block=False)
### '''
### 
### 
### # figuring out which are some edge triangles
### # visPoly(points[cells[ 8].verts],lc='k')
### # visPoly(points[cells[ 9].verts],lc='r')
### # visPoly(points[cells[11].verts],lc='b')#
### # visPoly(points[cells[12].verts],lc='m')
### # plt.show(block=False)
### 
### 
### if False:
###     # TRIANGLES ONLY!
###     triang = mtri.Triangulation(points[:,0], points[:,1], connectivity[:,1:])
### 
###     plt.tripcolor(points[:,0],points[:,1],connectivity[:,1:],facecolors=fields[:,2],cmap=cm.coolwarm)
###     plt.triplot(triang,'ko-',markersize=2)
###     plt.show(block=False)
### 
###     for c in ch:
###         plt.tripcolor(points[connectivity[c,1:],0],
###                       points[connectivity[c,1:],1],
###                       connectivity[c,1:],facecolors=fields[c,1:2],
###                       cmap=cm.coolwarm
###                       ,vmin=np.min(fields[ch,1:2])
###                       ,vmax=np.max(fields[ch,1:2]))
###         visPoly(points[connectivity[c,1:]])
### 
###     plt.show(block=False)
### 
### 
###     for c in [a.idx for a in cells[8].ring]:
###         plt.tripcolor(points[connectivity[c,1:],0],
###                       points[connectivity[c,1:],1],
###                       connectivity[c,1:],facecolors=fields[c,1:2]*0,
###                       cmap=cm.coolwarm
###                       ,vmin=np.min(fields[ch,1:2])
###                       ,vmax=np.max(fields[ch,1:2]))
###         visPoly(points[connectivity[c,1:]])
### 
###     plt.show(block=False)

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

### dim = 2
### plotS = True
### plt.figure(49)
### ci = 113
### ##### LEAST SQUARES STUFF #####
### for c in cells:
###     c.A = np.zeros((len(c.ring),dim))
###     c.b = np.zeros((len(c.ring),1))
###     c.w = np.eye(len(c.ring))
###     c.x = np.zeros((dim,1))
###     for i,oc in enumerate(c.ring):
###         for j in range(dim):
###             c.A[i,j] = oc.centre[j] - c.centre[j]
###         # p = 0
###         # c.w[i,i] = 1.0
###         # p = 1
###         # c.w[i,i] = 1.0 / np.sum(c.A[i,:]**2.)**0.5
###         # p = 2
###         # c.w[i,i] = 1.0 / np.sum(c.A[i,:]**2.)
###         # p = 2/3
###         # c.w[i,i] = 1.0 / (np.sum(c.A[i,:]**2.)**0.5)**(2/3.)
###         # p = 3/2
###         # c.w[i,i] = 1.0 / (np.sum(c.A[i,:]**2.)**0.5)**(3/2.)
###         # p = -3/2
###         c.w[i,i] = (np.sum(c.A[i,:]**2.)**0.5)**(-3/2.)
###         # c.w[i,i] = 1.0 / (np.sum(c.A[i,:]**2.)**0.5)**(-3/2.)
###         c.b[i,0] = fields[oc.idx,2] - fields[c.idx,2]
###         if plotS and c.idx == ci:
###             visPoly(points[oc.verts],lc=mcc[2])
###             visPoint(oc.centre,lc=mcc[2],ms=10)
### 
###     if plotS and c.idx == ci:
###         visPoly(points[c.verts],lc=mcc[0],ml='-.')
###         visPoint(c.centre,lc=mcc[0],ms=10)
### 
###     # soft constraint
###     if False:
###         for f in c.faces:
###             if f.isBoundary:
###                 c.A = np.vstack((c.A,np.zeros((1,dim))))
###                 for j in range(dim):
###                     c.A[-1,j] = f.centre[j] - c.centre[j]
###                 c.b = np.vstack((c.b,np.zeros((1,1))))
###                 c.b[-1,0] = fFields[f.boundaryIdx,0] - fields[c.idx,2]
### 
###     weighting = True
### 
###     c.AT = c.A.T
###     if weighting:
###         c.LHS = np.dot(np.dot(c.AT,c.w),c.A)
###         c.RHS = np.dot(np.dot(c.AT,c.w),c.b)
###     else:
###         c.LHS = np.dot(c.AT,c.A)
###         c.RHS = np.dot(c.AT,c.b)
### 
###     # hard constraints
###     if True:
###         locBoundaryFaces = 0
###         for f in c.faces:
###             if f.isBoundary:
###                 locBoundaryFaces += 1
### 
###         includeNeighbourBFs = False
### 
###         if includeNeighbourBFs:
###             # may want to include neighbouring cells boundary faces
###             for f in c.faces:
###                 for cn in f.cells:
###                     if cn.idx == c.idx:
###                         continue
###                     for fn in cn.faces:
###                         if fn.isBoundary:
###                             locBoundaryFaces += 1
### 
###         if locBoundaryFaces > 0:
###             c.LHS *= 2.0
###             c.RHS *= 2.0
###             c.C = np.zeros((locBoundaryFaces,dim))
###             c.d = np.zeros((locBoundaryFaces,1))
###             # c.z = np.zeros((locBoundaryFaces,1))
### 
###             # hard constraint; Dirichlet
###             # grad_F0 . (xf - x0) = (ff - f0)
###             if False:
###                 i = 0
###                 for f in c.faces:
###                     if f.isBoundary:
###                         for j in range(dim):
###                             c.C[i,j] = f.centre[j] - c.centre[j]
###                         c.d[i,0] = fFields[f.boundaryIdx,0] - fields[c.idx,2]
###                         i += 1
### 
###             # hard constraint; Neumann
###             # grad_F0 . (xf - x0) = (ff - f0)  <-- at cell
###             # grad_Ff . (x0 - xf) = (f0 - ff)  <-- at face
###             # ff = f0 - grad_Ff . (x0 - xf)
###             # therefore
###             # grad_F0 . (xf - x0) = -grad_Ff . (x0 - xf)
###             if False:
###                 i = 0
###                 for f in c.faces:
###                     if f.isBoundary:
###                         for j in range(dim):
###                             c.C[i,j] = f.centre[j] - c.centre[j]
###                         # only RHS changes
###                         gradFf = np.array([
###                             dfdx(*f.centre),
###                             dfdy(*f.centre)])
###                         dx = c.centre - f.centre
###                         c.d[i,0] = -gradFf[0]*dx[0] - gradFf[1]*dx[1]
###                         i += 1
### 
###             # hard constraint; correct Neumann (grad_f . n is set)
###             if True:
###                 i = 0
###                 for f in c.faces:
###                     if f.isBoundary:
###                         # calculate the normal
###                         p1,p2 = points[f.verts[0]], points[f.verts[1]]
###                         nx,ny = p1[1]-p2[1], p2[0]-p1[0]
###                         cvec = f.centre - c.centre
###                         if np.dot(cvec,np.array([nx,ny])) < 0.0:
###                             nx *= -1.0
###                             ny *= -1.0
###                         #
###                         mag = (nx*nx + ny*ny)**0.5
###                         nx /= mag
###                         ny /= mag
###                         #
###                         gradfnx = dfdx(*f.centre) * nx
###                         gradfny = dfdy(*f.centre) * ny
###                         c.C[i,0] = nx
###                         c.C[i,1] = ny
###                         c.d[i,0] = gradfnx + gradfny
###                         i += 1
###                         if plotS and c.idx == ci:
###                             visPoly(points[f.verts],lc=mcc[7],ml='--')
###                             visPoint(f.centre,lc=mcc[7],m='x',ms=10)
### 
### 
###                 if includeNeighbourBFs:
###                     # include neighbouring cells boundary faces
###                     for f in c.faces:
###                         for cn in f.cells:
###                             if cn.idx == c.idx:
###                                 continue
### 
###                             for fn in cn.faces:
###                                 if fn.isBoundary:
###                                     # calculate the normal
###                                     p1,p2 = points[fn.verts[0]], points[fn.verts[1]]
###                                     nx,ny = p1[1]-p2[1], p2[0]-p1[0]
###                                     cvec = fn.centre - c.centre
###                                     if np.dot(cvec,np.array([nx,ny])) < 0.0:
###                                         nx *= -1.0
###                                         ny *= -1.0
###                                     #
###                                     mag = (nx*nx + ny*ny)**0.5
###                                     nx /= mag
###                                     ny /= mag
###                                     #
###                                     gradfnx = dfdx(*fn.centre) * nx
###                                     gradfny = dfdy(*fn.centre) * ny
###                                     c.C[i,0] = nx
###                                     c.C[i,1] = ny
###                                     c.d[i,0] = gradfnx + gradfny
###                                     i += 1
###                                     if plotS and c.idx == ci:
###                                         visPoly(points[fn.verts],lc=mcc[9],ml='--')
###                                         visPoint(fn.centre,lc=mcc[9],m='*',ms=10)
### 
### 
###             c.LHS = np.vstack((
###                 np.hstack((c.LHS, c.C.T)),
###                 np.hstack((  c.C, np.zeros((locBoundaryFaces,locBoundaryFaces))))
###                 ))
###             c.RHS = np.vstack((
###                 c.RHS,
###                 c.d
###                 ))
### 
### 
### plt.show(block=False)
### 
### # for c in cells:
###     # c.x = np.dot(
###             # np.dot(
###                 # np.linalg.inv(
###                     # np.dot(c.AT,c.A)),
###                 # c.AT),
###             # c.b)
### 
### for c in cells:
###     c.x = np.dot(np.linalg.inv(c.LHS),c.RHS)
### 
### 
### errGradF = np.zeros((len(cells),2))
### for c in cells:
###     errGradF[c.idx,0] = c.x[0] - dfdx(*c.centre)
###     errGradF[c.idx,1] = c.x[1] - dfdy(*c.centre)
### 
### errL2dfdx = (np.sum(errGradF[:,0]**2.) / nElems)**0.5
### errL2dfdy = (np.sum(errGradF[:,1]**2.) / nElems)**0.5
### 
### if False:
###     # fp = open('res.dat', 'a')
###     # fp = open('tmp.dat', 'a')
###     fp = open('tmp2.dat', 'a')
###     fp.write('number of elements: {}\n'.format(nElems))
###     fp.write('type of elements: {}\n'.format(np.unique(connectivity[:,0])))
###     fp.write('number of face rings: {}\n'.format(numRings))
###     fp.write('min, max, avg of dfdx: {:.10f}, {:.10f}, {:.10f}\n'.format(
###             np.min(errGradF[:,0]),
###             np.max(errGradF[:,0]),
###             np.sum(errGradF[:,0]) / nElems))
###     fp.write('min, max, avg of dfdy: {:.10f}, {:.10f}, {:.10f}\n'.format(
###             np.min(errGradF[:,1]),
###             np.max(errGradF[:,1]),
###             np.sum(errGradF[:,1]) / nElems))
###     fp.write('L2 dfdx, dfdy: {:.10f}, {:.10f}\n\n'.format(
###         errL2dfdx,errL2dfdy))
###     fp.close()
### 
### else:
###     print('number of elements: {}'.format(nElems))
###     print('type of elements: {}'.format(np.unique(connectivity[:,0])))
###     print('number of face rings: {}'.format(numRings))
###     print('min, max, avg of dfdx: {:.10f}, {:.10f}, {:.10f}'.format(
###             np.min(errGradF[:,0]),
###             np.max(errGradF[:,0]),
###             np.sum(errGradF[:,0]) / nElems))
###     print('min, max, avg of dfdy: {:.10f}, {:.10f}, {:.10f}'.format(
###             np.min(errGradF[:,1]),
###             np.max(errGradF[:,1]),
###             np.sum(errGradF[:,1]) / nElems))
###     print('L2 dfdx, dfdy: {:.10f}, {:.10f}'.format(
###         errL2dfdx,errL2dfdy))
### 
### # TRIANGLES ONLY!
### triang = mtri.Triangulation(points[:,0], points[:,1], connectivity[:,1:])
### 
### cmap = cm.coolwarm
### cmaplist = [cmap(i) for i in range(cmap.N)]
### cmap = colors.LinearSegmentedColormap.from_list(
###         'ccmap',cmaplist,cmap.N)
### bounds = np.linspace(-0.5,0.5,13)
### norm = colors.BoundaryNorm(bounds,cmap.N)
### 
### vertErrs = np.zeros((nPoints,2))
### numAccum = np.zeros((nPoints))
### for c in cells:
###     for v in c.verts:
###         vertErrs[v,0] += errGradF[c.idx,0]
###         vertErrs[v,1] += errGradF[c.idx,1]
###         numAccum[v] += 1
### 
### vertErrs[:,0] /= numAccum
### vertErrs[:,1] /= numAccum
### 
### vi = -0.5; va = 0.5
### 
### cmap = cm.coolwarm
### 
### fig = plt.figure(111,figsize=(20,20))
### plt.title(r'$log$ of error of $\partial f / \partial x$',y=1.0,pad=-14)
### ax = plt.gca()
### # tpc = ax.tripcolor(points[:,0],points[:,1],connectivity[:,1:],
###         # facecolors=errGradF[:,0],cmap=cm.coolwarm,vmin=vi,vmax=va)
### # tpc = ax.tripcolor(points[:,0],points[:,1],connectivity[:,1:],
###         # facecolors=errGradF[:,0],cmap=cmap)#,norm=norm)
### tpc = ax.tripcolor(points[:,0],points[:,1],connectivity[:,1:],
###         facecolors=np.log(np.abs(errGradF[:,0])),cmap=cmap)#,norm=norm)
### # tpc = ax.tripcolor(triang,vertErrs[:,0],cmap=cmap,#vmin=vi,vmax=va,
###         # norm=norm,shading='flat')
### # tpc = ax.tripcolor(triang,vertErrs[:,0],cmap=cmap,#vmin=vi,vmax=va,
###         # norm=norm,shading='gouraud')
### # cb = plt.colorbar.ColorbarBase(ax,cmap=cmap,norm=norm,ticks=bounds,
###         # boundaries=bounds)
### # cb = plt.colorbar.ColorbarBase(ax,cmap=cmap,norm=norm,ticks=bounds,
###         # boundaries=bounds)
### # plt.triplot(triang,'w-',lw=0.1)
### fig.colorbar(tpc)
### ax.set_aspect('equal')
### ax.axis('off')
### plt.tight_layout()
### # plt.savefig('err_dfdx_4.png',dpi=400)
### plt.show(block=False)
### 
### # fig = plt.figure(111)
### # ax = plt.gca()
### # tpc = ax.tripcolor(points[:,0],points[:,1],connectivity[:,1:],
###         # facecolors=errGradF[:,1],cmap=cm.coolwarm,vmin=vi,vmax=va)
### # # plt.triplot(triang,'w-',lw=0.1)
### # fig.colorbar(tpc)
### # plt.show(block=False)
### 
### 
### 
### fig = plt.figure(112,figsize=(20,20))
### plt.title(r'$log$ of error of $\partial f / \partial y$',y=1.0,pad=-14)
### # plt.title(r'$\frac{\partial f}{\partial y}$',y=1.0,pad=-14)
### ax = plt.gca()
### # tpc = ax.tripcolor(points[:,0],points[:,1],connectivity[:,1:],
###         # facecolors=errGradF[:,1],cmap=cm.coolwarm,vmin=vi,vmax=va)
### # tpc = ax.tripcolor(points[:,0],points[:,1],connectivity[:,1:],
###         # facecolors=errGradF[:,1],cmap=cmap)#,norm=norm)
### tpc = ax.tripcolor(points[:,0],points[:,1],connectivity[:,1:],
###         facecolors=np.log(np.abs(errGradF[:,1])),cmap=cmap)#,norm=norm)
### # cb = plt.colorbar.ColorbarBase(ax,cmap=cmap,norm=norm,ticks=bounds,
###         # boundaries=bounds)
### # plt.triplot(triang,'w-',lw=0.1)
### fig.colorbar(tpc)
### ax.set_aspect('equal')
### ax.axis('off')
### plt.tight_layout()
### # plt.savefig('err_dfdy_4.png',dpi=400)
### plt.show(block=False)
### 
### fig = plt.figure(113)
### for c in cells:
###     for f in c.faces:
###         if f.isBoundary:
###             p1,p2 = points[f.verts[0]], points[f.verts[1]]
###             nx,ny = p1[1]-p2[1], p2[0]-p1[0]
###             cvec = f.centre - c.centre
###             if np.dot(cvec,np.array([nx,ny])) < 0.0:
###                 nx *= -1.0
###                 ny *= -1.0
###             #
###             mag = (nx*nx + ny*ny)**0.5
###             nx /= mag
###             ny /= mag
###             # visualise normals
###             plt.plot([f.centre[0],f.centre[0]+nx],
###                      [f.centre[1],f.centre[1]+ny],'b-')
###             #
###             gradfnx = dfdx(*f.centre) * nx
###             gradfny = dfdy(*f.centre) * ny
### 
### 
### ax = plt.gca()
### tpc = ax.tripcolor(points[:,0],points[:,1],connectivity[:,1:],
###         facecolors=fields[:,-1],cmap=cm.coolwarm)
### # plt.triplot(triang,'ko-',markersize=2)
### fig.colorbar(tpc)
### plt.show(block=False)
### 
### 
### 
### 
### 
### tNs = [117,468,1872,7488]
### qNs = [66,234,935,3743]
### 
### # [2-ring x, 2-ring y, 3-ring x, 3-ring y]
### tL2s = np.array([
###     [0.0881270506, 0.0884061194, 0.1147105438, 0.1146847626],
###     [0.0509759085, 0.0564668092, 0.0639101668, 0.0687018920],
###     [0.0338681162, 0.0364066731, 0.0417259482, 0.0435182344],
###     [0.0234117126, 0.0249487888, 0.0282671254, 0.0294458627],
###         ])
### # [2-ring x, 2-ring y, 3-ring x, 3-ring y]
### qL2s = np.array([
###     [0.1300284042, 0.1374803489, 0.1806276262, 0.1795262481],
###     [0.0742162578, 0.0766044630, 0.0974671968, 0.0967429109],
###     [0.0497103603, 0.0513003805, 0.0608095905, 0.0623536455],
###     [0.0336120199, 0.0342614834, 0.0404498287, 0.0410566144],
###         ])
### 
### '''
### plt.figure(13)
### plt.loglog(tNs,tL2s[:,0],'o-',c=mcc[0],mfc='none',label=' tri,dfdx,2-ring')
### plt.loglog(tNs,tL2s[:,1],'s-',c=mcc[1],mfc='none',label=' tri,dfdy,2-ring')
### plt.loglog(qNs,qL2s[:,0],'o-',c=mcc[8],mfc='none',label='quad,dfdx,2-ring')
### plt.loglog(qNs,qL2s[:,1],'s-',c=mcc[9],mfc='none',label='quad,dfdy,2-ring')
### plt.loglog(tNs,tL2s[:,2],'o-',c=mcc[0],label=' tri,dfdx,3-ring')
### plt.loglog(tNs,tL2s[:,3],'s-',c=mcc[1],label=' tri,dfdy,3-ring')
### plt.loglog(qNs,qL2s[:,2],'o-',c=mcc[8],label='quad,dfdx,3-ring')
### plt.loglog(qNs,qL2s[:,3],'s-',c=mcc[9],label='quad,dfdy,3-ring')
### plt.legend(loc='best')
### plt.show(block=False)
### 
### tooas = np.zeros((len(tNs)-1,4))
### qooas = np.zeros((len(qNs)-1,4))
### for i in range(len(tNs)-1):
###     tr = (tNs[i+1]/tNs[i])**(1.0 / dim)
###     tooas[i,0] = np.log(tL2s[i+1,0]/tL2s[i,0]) / np.log(tr)
###     tooas[i,1] = np.log(tL2s[i+1,1]/tL2s[i,1]) / np.log(tr)
###     tooas[i,2] = np.log(tL2s[i+1,2]/tL2s[i,2]) / np.log(tr)
###     tooas[i,3] = np.log(tL2s[i+1,3]/tL2s[i,3]) / np.log(tr)
### 
### for i in range(len(qNs)-1):
###     qr = (qNs[i+1]/qNs[i])**(1.0 / dim)
###     qooas[i,0] = np.log(qL2s[i+1,0]/qL2s[i,0]) / np.log(qr)
###     qooas[i,1] = np.log(qL2s[i+1,1]/qL2s[i,1]) / np.log(qr)
###     qooas[i,2] = np.log(qL2s[i+1,2]/qL2s[i,2]) / np.log(qr)
###     qooas[i,3] = np.log(qL2s[i+1,3]/qL2s[i,3]) / np.log(qr)
### 
### plt.figure(14)
### plt.plot(tNs[1:],tooas[:,0],'o-',c=mcc[0],mfc='none',label=' tri,dfdx,2-ring')
### plt.plot(tNs[1:],tooas[:,1],'s-',c=mcc[1],mfc='none',label=' tri,dfdy,2-ring')
### plt.plot(qNs[1:],qooas[:,0],'o-',c=mcc[8],mfc='none',label='quad,dfdx,2-ring')
### plt.plot(qNs[1:],qooas[:,1],'s-',c=mcc[9],mfc='none',label='quad,dfdy,2-ring')
### plt.plot(tNs[1:],tooas[:,2],'o-',c=mcc[0],label=' tri,dfdx,3-ring')
### plt.plot(tNs[1:],tooas[:,3],'s-',c=mcc[1],label=' tri,dfdy,3-ring')
### plt.plot(qNs[1:],qooas[:,2],'o-',c=mcc[8],label='quad,dfdx,3-ring')
### plt.plot(qNs[1:],qooas[:,3],'s-',c=mcc[9],label='quad,dfdy,3-ring')
### plt.legend(loc='best')
### plt.show(block=False)
### '''
### 
### # ======================================================
### 
### # with hard constraint
### 
### tNs = [
###      117,
###      468,
###     1872,
###     2836,
###     6028,
###     7488,
###     ]
### 
### tL2s = np.array([
###     [0.3985302546, 0.4035986315],
###     [0.1503884617, 0.1516778791],
###     [0.0587889428, 0.0592583017],
###     [0.0800550351, 0.0776482036],
###     [0.0404355719, 0.0402355751],
###     [0.0217968515, 0.0219391942],
###     ])
### 
### '''
### plt.figure(15)
### plt.loglog(tNs,tL2s[:,0],'o-',c=mcc[0],mfc='none',label=' tri,dfdx,2-ring')
### plt.loglog(tNs,tL2s[:,1],'s-',c=mcc[1],mfc='none',label=' tri,dfdy,2-ring')
### plt.legend(loc='best')
### plt.show(block=False)
### 
### tooas = np.zeros((len(tNs)-1,4))
### for i in range(len(tNs)-1):
###     tr = (tNs[i+1]/tNs[i])**(1.0 / dim)
###     tooas[i,0] = np.log(tL2s[i+1,0]/tL2s[i,0]) / np.log(tr)
###     tooas[i,1] = np.log(tL2s[i+1,1]/tL2s[i,1]) / np.log(tr)
### 
### plt.figure(16)
### plt.plot(tNs[1:],tooas[:,0],'o-',c=mcc[0],mfc='none',label=' tri,dfdx,2-ring')
### plt.plot(tNs[1:],tooas[:,1],'s-',c=mcc[1],mfc='none',label=' tri,dfdy,2-ring')
### plt.legend(loc='best')
### plt.show(block=False)
### '''


def inside(mp,xl,xr,yl,yr):
    if mp[0] > xl and mp[0] < xr:
        if mp[1] > yl and mp[1] < yr:
            return True
    return False

# faces; faceCentroids
# pfaces; pfaceCentroids

xl,xr,yl,yr = -1.7,-1.0,1.0,1.7

xl,xr,yl,yr = -1.7,-0.8,0.8,1.7

if VIS:

    # dual mesh image
    plt.figure(1)
    for face in faces:
        if inside(faceCentroids[face.id],xl-0.1,xr+0.1,yl-0.1,yr+0.1):
            for he in face.hes:
                plt.plot([he.vertex[0],he.next.vertex[0]],
                         [he.vertex[1],he.next.vertex[1]],'k-',lw=1)

    for pface in pfaces:
        if inside(pfaceCentroids[pface.id],xl-0.1,xr+0.1,yl-0.1,yr+0.1):
            for he in pface.hes:
                plt.plot([he.vertex[0],he.next.vertex[0]],
                         [he.vertex[1],he.next.vertex[1]],'k:',lw=1)

    plt.xlim(xl,xr)
    plt.ylim(yl,yr)
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    ax.set_aspect('equal',adjustable='box')
    plt.show(block=False)


    # tri mesh image
    plt.figure(2)
    for face in faces:
        if inside(faceCentroids[face.id],xl-0.1,xr+0.1,yl-0.1,yr+0.1):
            for he in face.hes:
                plt.plot([he.vertex[0],he.next.vertex[0]],
                         [he.vertex[1],he.next.vertex[1]],'k-',lw=0.5)
            plt.plot(faceCentroids[face.id][0],
                     faceCentroids[face.id][1],'ko',ms=4)

    plt.xlim(xl,xr)
    plt.ylim(yl,yr)
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    ax.set_aspect('equal',adjustable='box')
    plt.show(block=False)


    # poly mesh image
    plt.figure(3)
    for face in faces:
        if inside(faceCentroids[face.id],xl-0.1,xr+0.1,yl-0.1,yr+0.1):
            for he in face.hes:
                plt.plot([he.vertex[0],he.next.vertex[0]],
                         [he.vertex[1],he.next.vertex[1]],'ko-',lw=0.5,ms=4)

    plt.xlim(xl,xr)
    plt.ylim(yl,yr)
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    ax.set_aspect('equal',adjustable='box')
    plt.show(block=False)

    # poly mesh image
    plt.figure(4)
    for face in faces:
        if inside(faceCentroids[face.id],xl-0.1,xr+0.1,yl-0.1,yr+0.1):
            for he in face.hes:
                plt.plot([he.vertex[0],he.next.vertex[0]],
                         [he.vertex[1],he.next.vertex[1]],'ko-',lw=0.5,ms=4)

    for pface in pfaces:
        if inside(pfaceCentroids[pface.id],xl-0.1,xr+0.1,yl-0.1,yr+0.1):
            for he in pface.hes:
                plt.plot([he.vertex[0],he.next.vertex[0]],
                         [he.vertex[1],he.next.vertex[1]],
                         'k',lw=0.5,linestyle=(0,(5,10)))

    plt.xlim(xl,xr)
    plt.ylim(yl,yr)
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    ax.set_aspect('equal',adjustable='box')
    plt.show(block=False)

    # poly mesh image
    plt.figure(5)
    for face in faces:
        if inside(faceCentroids[face.id],xl-0.1,xr+0.1,yl-0.1,yr+0.1):
            for he in face.hes:
                plt.plot([he.vertex[0]],
                         [he.vertex[1]],'ko',ms=4)

    for pface in pfaces:
        if inside(pfaceCentroids[pface.id],xl-0.1,xr+0.1,yl-0.1,yr+0.1):
            for he in pface.hes:
                # plt.plot([he.vertex[0],he.next.vertex[0]],
                         # [he.vertex[1],he.next.vertex[1]],
                         # 'k',lw=1,linestyle=(0,(5,10)))
                plt.plot([he.vertex[0],he.next.vertex[0]],
                         [he.vertex[1],he.next.vertex[1]],'k-',lw=0.5)

    plt.xlim(xl,xr)
    plt.ylim(yl,yr)
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    ax.set_aspect('equal',adjustable='box')
    plt.show(block=False)


    # poly mesh image
    plt.figure(6)
    for face in faces:
        if inside(faceCentroids[face.id],xl-0.1,xr+0.1,yl-0.1,yr+0.1):
            for he in face.hes:
                plt.plot([he.vertex[0]],
                         [he.vertex[1]],'k-',ms=4)

    for pface in pfaces:
        if inside(pfaceCentroids[pface.id],xl-0.1,xr+0.1,yl-0.1,yr+0.1):
            for he in pface.hes:
                # plt.plot([he.vertex[0],he.next.vertex[0]],
                         # [he.vertex[1],he.next.vertex[1]],
                         # 'k',lw=1,linestyle=(0,(5,10)))
                plt.plot([he.vertex[0],he.next.vertex[0]],
                         [he.vertex[1],he.next.vertex[1]],'k-',lw=0.5)
            plt.plot(pfaceCentroids[pface.id][0],
                     pfaceCentroids[pface.id][1],'ko',ms=4)

    plt.xlim(xl,xr)
    plt.ylim(yl,yr)
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    ax.set_aspect('equal',adjustable='box')
    plt.show(block=False)


# looking at two ring

plt.figure(11)
ii = 1203
for oc in cells[ii].ring:
    visPoly(points[oc.verts],lc='k',ml='-',lw=2)

visPoly(points[cells[ii].verts],lc='r',ml='-',lw=2)
ax = plt.gca()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()
ax.set_aspect('equal',adjustable='box')
plt.show(block=False)



plt.figure(11)
ii = 1203
for oc in cells[1203].ring[3:]:
    plt.plot([*points[oc.verts][:,0],points[oc.verts][0,0]],
             [*points[oc.verts][:,1],points[oc.verts][0,1]],'r-',lw=2)
    plt.plot(oc.centre[0],oc.centre[1],'ro')

for oc in cells[1203].ring[:3]:
    plt.plot([*points[oc.verts][:,0],points[oc.verts][0,0]],
             [*points[oc.verts][:,1],points[oc.verts][0,1]],'b-',lw=2)
    plt.plot(oc.centre[0],oc.centre[1],'bo')

visPoly(points[cells[ii].verts],lc='k',ml='-',lw=2)
plt.plot(cells[ii].centre[0],cells[ii].centre[1],'ko')
ax = plt.gca()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()
ax.set_aspect('equal',adjustable='box')
plt.show(block=False)


# for node setup
## for c in cells:
##     c.nodes = []
## 
## for f in faces:
##     f.nodes = []
## 
## hm = {}
## iface = 0
## for i in range(nElems):
##     elem = connectivity[i,:]
##     n = elem.shape[0] - 1
##     if elem[0] == 5:
##         n = 3
##     elif elem[0] == 9:
##         n = 4
##     elif elem[0] == 7:
##         for m in range(n,0,-1):
##             if connectivity[i][m] != 0:
##                 n = m
##                 break
##         elem = connectivity[i,:n+1]
##     elem = elem[1:n+1]
##     #
##     for j in elem:
##         nodes[j].cells.append(cells[i])
##     #
##     for k in range(n):
##         v1 = elem[k]
##         v2 = elem[(k+1)%n]
##         v1,v2 = sorted((v1,v2))
##         key = v1 + v2 * ps[1]
##         val = hm.get(key)
##         if val is None:
##             nodes[v1].faces.append(faces[iface])
##             nodes[v2].faces.append(faces[iface])
##             hm[key] = iface
##             iface += 1
## 
## for i in range(nPoints):
##     for c in nodes[i].cells:
##         c.nodes.append(nodes[i])
##     for f in nodes[i].faces:
##         f.nodes.append(nodes[i])



ii = 1203
lps = points[nodes[ii].cells[0].verts]
unique_vs = []
for c in nodes[ii].cells:
    for v in c.verts:
        if v not in unique_vs:
            unique_vs.append(v)

unique_ps = np.array([points[v] for v in unique_vs])

vor = Voronoi(unique_ps)

fig = voronoi_plot_2d(vor)
for c in nodes[ii].cells:
    visPoly(points[c.verts],lc='k')

plt.plot(unique_ps[:,0],unique_ps[:,1],'r*')
plt.show(block=False)


for c in cells:
    visPoly(points[c.verts])

plt.show(block=False)


ii = 1203
ii = 237
lps = points[nodes[ii].cells[0].verts]
unique_vs = []
for c in nodes[ii].cells:
    for v in c.verts:
        if v not in unique_vs:
            unique_vs.append(v)

# cpy = unique_vs[:]
# for uv in cpy:
    # for c in nodes[uv].cells:
        # for v in c.verts:
            # if v not in unique_vs:
                # unique_vs.append(v)

unique_ps = np.array([points[v] for v in unique_vs])

vor = Voronoi(unique_ps)

# fig = voronoi_plot_2d(vor)





p = np.array([-1.31401,-1.4208])
md = 100000
mi = -1
for i,n in enumerate(nodes):
    dist = np.dot(n.vert-p,n.vert-p)**0.5
    if dist < md:
        md = dist
        mi = i



    f.centre = np.sum(points[f.verts],0)/2.0
    p1,p2 = points[f.verts[0]], points[f.verts[1]]
    nf = np.array([p1[1]-p2[1], p2[0]-p1[0]])
    f.norm = nf / np.dot(nf,nf)**0.5



fig = plt.figure()
ax = plt.gca()
for c in nodes[ii].cells:
    visPoly(points[c.verts],lc='k',lw=4)

plt.plot(unique_ps[:,0],unique_ps[:,1],'ko',ms=8)
fig = jamie_voronoi_plot_2d(vor,ax,
        show_vertices=False,show_points=False,
        line_colors='r',line_width=4)

nf = np.array([vor.vertices[1,1]-vor.vertices[4,1],
               vor.vertices[4,0]-vor.vertices[1,0]])
nf = nf / np.dot(nf,nf)**0.5

mp = (unique_ps[0]+unique_ps[2])/2.0
# plt.plot([mp[0],mp[0]-nf[0]],
         # [mp[1],mp[1]-nf[1]],'g-',lw=4)
dist = unique_ps[2]-mp
dist = np.dot(dist,dist)**0.5 * 0.8

mp2 = (vor.vertices[1]+vor.vertices[4])/2.0
plt.arrow(mp2[0],mp2[1],-nf[0]*dist,-nf[1]*dist,
        color='r',ls='-',width=0.002)

ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()
ax.set_aspect('equal',adjustable='box')
ax.set_xlim(sx)
ax.set_ylim(sy)
plt.show(block=False)


fig = plt.figure()
ax = plt.gca()
for c in nodes[ii].cells:
    visPoly(points[c.verts],lc='k',lw=4)

mps = []
newi = [1,0,2,4,5,3,6]
for ni in newi:
    mps.append(nodes[ii].cells[ni].centre)

mps = np.array(mps)
mps = np.vstack((mps,mps[0:1]))
plt.plot(mps[:,0],mps[:,1],'b-',lw=4)

nf = np.array([nodes[ii].cells[1].centre[1]-nodes[ii].cells[0].centre[1],
               nodes[ii].cells[0].centre[0]-nodes[ii].cells[1].centre[0]])
nf = nf / np.dot(nf,nf)**0.5

mp2 = (nodes[ii].cells[1].centre+nodes[ii].cells[0].centre)/2.0
plt.arrow(mp2[0],mp2[1],nf[0]*dist,nf[1]*dist,
        color='b',ls='-',width=0.002)

plt.plot(unique_ps[:,0],unique_ps[:,1],'ko',ms=8)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()
ax.set_aspect('equal',adjustable='box')
# sx = ax.get_xlim()
# sy = ax.get_ylim()
ax.set_xlim(sx)
ax.set_ylim(sy)
plt.show(block=False)



fig = plt.figure()
ax = plt.gca()
for c in nodes[ii].cells:
    visPoly(points[c.verts],lc='k',lw=4)

mps = []
# newi = [1,0,2,4,5,3,6]
mps.append(nodes[ii].cells[1].centre)
mps.append((unique_ps[0]+unique_ps[2])/2.0)
mps.append(nodes[ii].cells[0].centre)
mps.append((unique_ps[0]+unique_ps[1])/2.0)
mps.append(nodes[ii].cells[2].centre)
mps.append((unique_ps[0]+unique_ps[4])/2.0)
mps.append(nodes[ii].cells[4].centre)
mps.append((unique_ps[0]+unique_ps[7])/2.0)
mps.append(nodes[ii].cells[5].centre)
mps.append((unique_ps[0]+unique_ps[5])/2.0)
mps.append(nodes[ii].cells[3].centre)
mps.append((unique_ps[0]+unique_ps[6])/2.0)
mps.append(nodes[ii].cells[6].centre)
mps.append((unique_ps[0]+unique_ps[3])/2.0)

mps = np.array(mps)
mps = np.vstack((mps,mps[0:1]))
plt.plot(mps[:,0],mps[:,1],'g-',lw=4)

nf = np.array([mps[0,1]-mps[1,1],
               mps[1,0]-mps[0,0]])
nf = nf / np.dot(nf,nf)**0.5

mp2 = (mps[0]+mps[1])/2.0
plt.arrow(mp2[0],mp2[1],nf[0]*dist,nf[1]*dist,
        color='g',ls='-',width=0.002)

nf = np.array([mps[1,1]-mps[2,1],
               mps[2,0]-mps[1,0]])
nf = nf / np.dot(nf,nf)**0.5

mp2 = (mps[1]+mps[2])/2.0
plt.arrow(mp2[0],mp2[1],nf[0]*dist,nf[1]*dist,
        color='g',ls='-',width=0.002)

plt.plot(unique_ps[:,0],unique_ps[:,1],'ko',ms=8)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()
ax.set_aspect('equal',adjustable='box')
ax.set_xlim(sx)
ax.set_ylim(sy)
plt.show(block=False)








dist = 0.0324
dist = 0.02

fig = plt.figure()
ax = plt.gca()
for c in nodes[ii].cells:
    visPoly(points[c.verts],lc='k',lw=4)

mps = []
# newi = [1,0,2,4,5,3,6]
mps.append(nodes[ii].cells[1].centre)
mps.append((unique_ps[0]+unique_ps[2])/2.0)
mps.append(nodes[ii].cells[0].centre)
mps.append((unique_ps[0]+unique_ps[1])/2.0)
mps.append(nodes[ii].cells[2].centre)
mps.append((unique_ps[0]+unique_ps[4])/2.0)
mps.append(nodes[ii].cells[4].centre)
mps.append((unique_ps[0]+unique_ps[7])/2.0)
mps.append(nodes[ii].cells[5].centre)
mps.append((unique_ps[0]+unique_ps[5])/2.0)
mps.append(nodes[ii].cells[3].centre)
mps.append((unique_ps[0]+unique_ps[6])/2.0)
mps.append(nodes[ii].cells[6].centre)
mps.append((unique_ps[0]+unique_ps[3])/2.0)

mps = np.array(mps)
mps = np.vstack((mps,mps[0:1]))
plt.plot(mps[:,0],mps[:,1],'g-',lw=4)

for i in range(mps.shape[0]-1):
    nf = np.array([mps[i  ,1]-mps[i+1,1],
                   mps[i+1,0]-mps[i  ,0]])
    nf = nf / np.dot(nf,nf)**0.5
    #
    mp2 = (mps[i]+mps[i+1])/2.0
    plt.arrow(mp2[0],mp2[1],nf[0]*dist,nf[1]*dist,
            color='g',ls='-',width=0.002)

plt.plot(unique_ps[:,0],unique_ps[:,1],'ko',ms=8)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()
ax.set_aspect('equal',adjustable='box')
ax.set_xlim(sx)
ax.set_ylim(sy)
plt.show(block=False)



# building sparse matrix
from scipy.sparse import csr_array
from scipy.sparse.csgraph import reverse_cuthill_mckee

rowInds = []
colInds = []
for c in cells:
    rowInds.append(c.idx)
    colInds.append(c.idx)
    for f in c.faces:
        for oc in f.cells:
            if oc.idx != c.idx:
                rowInds.append(c.idx)
                colInds.append(oc.idx)


mat = csr_array((np.ones(len(rowInds)),(rowInds,colInds)))
rcm = reverse_cuthill_mckee(mat)
rrcm = np.zeros_like(rcm)
for i in range(rcm.shape[0]):
    rrcm[rcm[i]] = i

tam = csr_array((np.ones(len(rowInds)),(rrcm[np.array(rowInds)],
                                        rrcm[np.array(colInds)])))

plt.spy(mat,markersize=1,color='k')
plt.spy(tam,markersize=1,color='k')
plt.show(block=False)

# stats
res = np.unique(rowInds,return_counts=True)
numEntries = np.unique(res[1],return_counts=True)

triEntries = [3,4]
triQuantities = [180, 5848]
polyEntries = [5,6,7,8]
# polyQuantities = [48,127,1237,49]
polyQuantities = [176,71,2801,57]

# triQuantities = [np.log10(a) for a in triQuantities]
# polyQuantities = [np.log10(a) for a in polyQuantities]


# creating the dataset
fig = plt.figure(figsize = (10, 5))

# creating the bar plot
plt.bar(triEntries, triQuantities, color=mcc[0], width = 0.4)
plt.bar(polyEntries, polyQuantities, color=mcc[9], width = 0.4)

# plt.xlabel("Courses offered")
# plt.ylabel("No. of students enrolled")
# plt.title("Students enrolled in different courses")
plt.show()

mey = csr_array((np.ones(len(cells)),
    (np.arange(len(cells),dtype=int),np.arange(len(cells),dtype=int))))

plt.spy(mey,markersize=1,color='k')
plt.show(block=False)
