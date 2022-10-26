import matplotlib.pyplot as plt
import numpy as np
from itertools import islice

from halfedge import *

# types:
# VTK_TRIANGLE = 5
# VTK_POLYGON = 7

filename = "ramp-b0000-t0005.vtu"
# filename = "cust-b0000-t0005.vtu"
fp = open(filename, "r")

# check format
if "UnstructuredGrid" not in fp.readline():
    print("error: bad file {}; not UnstructuredGrid".format(
        filename))
    quit(44)

# get number of points and number of cells
line = fp.readline().split(" ")
npts = int(line[1].split('"')[1])
ncells = int(line[2].split('"')[1])

# check points are float32 or float64
line = fp.readline()
if "Float32" in line:
    ftype = np.float32
else:
    ftype = np.float64
fp.readline()   # could check this for NumberOfComponents="3"

gen = islice(fp, npts)
pts = np.genfromtxt(gen,dtype=ftype)

fp.readline()   # </DataArray>
fp.readline()   # </Points>
fp.readline()   # <Cells>

line = fp.readline()
if "Int32" in line:
    itype = np.int32
else:
    itype = np.int63

if "connectivity" not in line:
    print("error: expected 'connectivity' in line, got {}".format(
        line))
    quit(44)

gen = islice(fp, ncells)
connectivity = np.genfromtxt(gen,dtype=itype).flatten()

fp.readline()   # </DataArray>

line = fp.readline()
if "Int32" in line:
    itype = np.int32
else:
    itype = np.int63

if "offsets" not in line:
    print("error: expected 'offsets' in line, got {}".format(
        line))
    quit(44)

gen = islice(fp, ncells)
offsets = np.genfromtxt(gen,dtype=itype).flatten()

fp.readline()   # </DataArray>

line = fp.readline()
if "UInt8" in line:
    utype = np.uint8
else:
    utype = np.uint

if "types" not in line:
    print("error: expected 'types' in line, got {}".format(
        line))
    quit(44)

gen = islice(fp, ncells)
types = np.genfromtxt(gen,dtype=utype).flatten()
if not (types == 5).all():
    print("error: expected all types = 5 (VTK_TRIANGLE")
    quit(44)

fp.readline()   # </DataArray>
fp.readline()   # </Cells>
line = fp.readline()   # <CellData>

def getType(s):
    ts = {
        "Int8"     : np.int8,
        "UInt8"    : np.uint8,
        "Int16"    : np.int16,
        "UInt16"   : np.uint16,
        "Int32"    : np.int32,
        "UInt32"   : np.uint32,
        "Int64"    : np.int64,
        "UInt64"   : np.uint64,
        "Float32"  : np.float32,
        "Float64"  : np.float64,
            }
    if s not in ts.keys():
        print("error: couldn't match datatype {}.\nAvailable is {}"
                .format(s,ts))
        quit(44)

cellData = {}
valid = True
while valid:
    line = fp.readline()
    if "/CellData" in line:
        break
    sline = line.split('"')
    name = sline[1]
    dtype = getType(sline[3])
    gen = islice(fp, ncells)
    data = np.genfromtxt(gen,dtype=dtype).flatten()
    cellData[name] = data
    line = fp.readline()

fp.close()
# </CellData>
# </Piece>
# </UnstructuredGrid>
# </VTKFile>

# npts
# ncells
# pts
# connectivity
# offsets


# loop over offsets and access connectivity
hes = []

a = 0
for b in offsets:
    # offsets give a range of connectivity values (point ids)
    #  for a face (assumed ordered)
    for i,pid in enumerate(connectivity[a:b]):
        hes.append(HalfEdge(pts[pid], pid,
                connectivity[a+(i+1)%(b-a)]))

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
# set outsigns
for pe in pes:
    if pe.os is not None:
        continue
    if pe.twin is None:
        # boundary
        pe.os = 1
        nbndry += 1
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
    for p in pfaces:
        pctr = pfaceCentroids[p.id]
        for pi in edgeData(p,False)[1]:
            mp = midpoint(pedges[pi].he)
            plt.plot([pctr[0],mp[0]],[pctr[1],mp[1]],'k.-',mfc='none')

    plt.show(block=False)
    visualise(pes)

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

if writeE4Grid:
    # fp = open("polygrid.b0000","w")
    fp = open("polygrid.b0001","w")

    fp.write('unstructured_grid 1.0\n')
    fp.write('label:\n') 
    fp.write('dimensions: 2\n')
    fp.write('vertices: {}\n'.format(pnpts))

    for p in pctrs:
        fp.write('{:.16e} {:.16e} {:.16e}\n'.format(p[0],p[1],p[2]))

    fp.write('faces: {}\n'.format(pnedges))
    for p in pedges:
        fp.write('2 {} {}\n'.format(p.he.p0, p.he.p1))

    fp.write('cells: {}\n'.format(pncells))
    for p in pfaces:
        # triangle vtx 3 0 1 2 faces 3 0 1 2 outsigns 3 1 1 1
        line = 'polygon '
        line += 'vtx ' + vertexData(p)
        line += 'faces ' + edgeData(p)
        line += 'outsigns ' + outsignData(p)
        line += '\n'
        fp.write(line)

    # boundaries: 4
    # PARALLEL faces 49 74 75 78 86 101 118 141 143 161 163 167 177 210 214 220 224 242 249 262 272 274 278 284 286 289 295 309 342 396 398 401 405 406 408 411 416 417 420 422 424 427 429 431 434 436 445 446 495 501 outsigns 49 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

    fp.write('boundaries: {}\n'.format(1))
    line = 'WALL faces '
    line2 = ' '
    line3 = ' '
    nbfs = 0
    for edge in pedges:
        if edge.he.twin is None:
            nbfs += 1
            line2 += str(edge.id) + ' '
            line3 += str(edge.he.os) + ' '
    fp.write(line + str(nbfs) + line2 + 'outsigns ' + str(nbfs) + line3 + '\n')

    fp.close()

# for p in pedges[34:49]:
    # visualise([p.he])
    # plt.show(block=False)

colors = ['b','g','r','c','m','y','k']
plt.figure(1)
# for i,p in enumerate(pfaces[:1]):
for i,p in enumerate(pfaces[:1]):
    nv,vd = vertexData2(p,False)
    ne,ed = edgeData(p,False)
    no,od = outsignData(p,False)
    c = colors[i%len(colors)]
    # for ei in ed:
        # # visualise([pedges[ei].he])
        # vtx0 = pedges[ei].he.vertex
        # vtx1 = pedges[ei].he.next.vertex
        # ctr = centroid(p)
        # vtx0 = (10*vtx0 + ctr) /11.
        # vtx1 = (10*vtx1 + ctr) /11.
        # plt.plot([vtx0[0],vtx1[0]], [vtx0[1],vtx1[1]],c)
        # plt.plot([vtx0[0],ctr[0]],[vtx0[1],ctr[1]],c)
        # plt.plot([vtx1[0],ctr[0]],[vtx1[1],ctr[1]],c)
    visualise(p.hes)
    for h in p.hes:
        plt.plot([h.vertex[0]],[h.vertex[1]],'go',mfc='none')
    for h in p.hes:
        plt.plot([pctrs[h.p0][0]],[pctrs[h.p0][1]],'rx',mfc='none')
    for j,vi in enumerate(vd):
        # vtx0 = pes[vi].vertex
        # vtx1 = pes[vi].next.vertex
        vtx0 = pctrs[vi]
        vtx1 = pctrs[vd[(j+1)%nv]]
        ctr = centroid(p)
        # vtx0 = (10*vtx0 + ctr) /11.
        # vtx1 = (10*vtx1 + ctr) /11.
        plt.plot([ctr[0]],[ctr[1]],'ko')
        plt.plot([vtx0[0],vtx1[0]], [vtx0[1],vtx1[1]],'r')
        # plt.plot([vtx0[0],ctr[0]],[vtx0[1],ctr[1]],'g')
        # plt.plot([vtx1[0],ctr[0]],[vtx1[1],ctr[1]],'b')


colors = ['b','g','r','c','m','y','k']
plt.figure(1)
for i,p in enumerate(pfaces[:1]):
    nv,vd = vertexData(p,False)
    ne,ed = edgeData(p,False)
    no,od = outsignData(p,False)
    c = colors[i%len(colors)]
    for ei in ed:
        # visualise([pedges[ei].he])
        vtx0 = pedges[ei].he.vertex
        vtx1 = pedges[ei].he.next.vertex
        plt.plot([vtx0[0],vtx1[0]], [vtx0[1],vtx1[1]],c)
