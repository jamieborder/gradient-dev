

if writeE4Grid:
    fp = open("polygrid.b0000","w")

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
