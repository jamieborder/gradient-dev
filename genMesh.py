import matplotlib.pyplot as plt
import numpy as np

fname = 'sTetMesh.su2'

N = 20

nVerts = N*N
verts = np.zeros((N,N,2))
xa,xb = 0.0, 1.0
ya,yb = 0.0, 1.0
dx = (xb - xa) / (N-1)
dy = (yb - ya) / (N-1)
ids = np.zeros((N,N)).astype(np.int32)
i = 0
for row in range(N):
    for col in range(N):
        verts[row,col,0] = xa + dx * col
        verts[row,col,1] = ya + dy * row
        ids[row,col] = i
        i += 1

points = verts.reshape((nVerts,2))
pids = ids.reshape((nVerts))

nElems = (N-1)*(N-1)*2
elems = np.zeros((nElems,3),dtype=np.int32)
i = 0
for row in range(N-1):
    for col in range(N-1):
        elems[i,0] = ids[row  ,col  ]
        elems[i,1] = ids[row  ,col+1]
        elems[i,2] = ids[row+1,col  ]
        i += 1
        elems[i,0] = ids[row  ,col+1]
        elems[i,1] = ids[row+1,col+1]
        elems[i,2] = ids[row+1,col  ]
        i += 1

nBFaces = (N-1)*4
bFaces = np.zeros((nBFaces,2),dtype=np.int32)
i = 0
for col in range(N-1):
    bFaces[i,0] = ids[0,col  ]
    bFaces[i,1] = ids[0,col+1]
    i += 1
for row in range(N-1):
    bFaces[i,0] = ids[row  ,-1]
    bFaces[i,1] = ids[row+1,-1]
    i += 1
for col in range(N-1,0,-1):
    bFaces[i,0] = ids[-1,col  ]
    bFaces[i,1] = ids[-1,col-1]
    i += 1
for row in range(N-1,0,-1):
    bFaces[i,0] = ids[row  ,0]
    bFaces[i,1] = ids[row-1,0]
    i += 1

fp = open(fname,'w')
fp.write('NDIME= 2\n')
fp.write('NELEM= {}\n'.format(nElems))
for i in range(nElems):
    fp.write('5 {} {} {} {}\n'.format(*elems[i],i))
fp.write('NPOIN= {}\n'.format(nVerts))
for i in range(nVerts):
    fp.write('{} {} {}\n'.format(*points[i],i))
fp.write('NMARK= 1\n')
fp.write('MARKER_TAG= WALL\n')
fp.write('MARKER_ELEMS= {}\n'.format(nBFaces))
for i in range(nBFaces):
    fp.write('3 {} {}\n'.format(*bFaces[i]))
fp.close()
