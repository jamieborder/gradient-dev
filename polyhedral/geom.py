import matplotlib.pyplot as plt
import numpy as np
import random

def centroid(pts):
    return np.sum(pts, axis=0)/pts.shape[0]

def centroid2(pts):
    ctr = np.zeros((3))
    npts = pts.shape[0]
    wpts = np.vstack((pts,pts[0,:]))
    A = area2(pts)
    for i in range(npts):
        ctr[0] += (wpts[i,0]+wpts[i+1,0])*(wpts[i,0]*wpts[i+1,1]
                - wpts[i+1,0]*wpts[i,1])
        ctr[1] += (wpts[i,1]+wpts[i+1,1])*(wpts[i,0]*wpts[i+1,1]
                - wpts[i+1,0]*wpts[i,1])
    return ctr / (6.*A)

def edgeMidPoints(pts):
    npts = pts.shape[0]
    wpts = np.vstack((pts,pts[0,:]))
    midpts = np.zeros_like(pts)
    for i in range(npts):
        midpts[i,:2] = np.sum(wpts[i:i+2,:2],axis=0) / 2.
    return midpts

def minWidth(pts):
    midpts = edgeMidPoints(pts)


def area(pts):
    val = 0.0
    npts = pts.shape[0]
    wpts = np.vstack((pts,pts[0,:]))
    ymin = np.min(pts[:,1])
    for i in range(npts):
        val += ((wpts[i,1]-ymin + wpts[i,1]-ymin)/2.
                * (wpts[i,0] - wpts[i+1,0]))
    return val

def area2(pts):
    val = 0.0
    npts = pts.shape[0]
    wpts = np.vstack((pts,pts[0,:]))
    for i in range(npts):
        val += wpts[i,0]*wpts[i+1,1] - wpts[i+1,0]*wpts[i,1]
    return 0.5 * val

def show(pts,line=True):
    plt.figure(1)

    if line:
        # wrap first point
        wpts = np.vstack((pts,pts[0,:]))
        plt.plot(wpts[:,0],wpts[:,1],'bo-')
    else:
        for pt in pts:
            plt.plot(pt[0],pt[1],'bo')

    ctr = centroid(pts)
    plt.plot(ctr[0],ctr[1],'r*')
    ctr = centroid2(pts)
    plt.plot(ctr[0],ctr[1],'kx')

    midpts = edgeMidPoints(pts)
    for mpt in midpts:
        plt.plot(mpt[0],mpt[1],'ks')

    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show(block=False)

if __name__ == "__main__":
    npts = 6
    # vs = np.random.rand(npts,3)
    vs = np.zeros((npts,3))

    ang = 2.*np.pi / npts
    # ang = 4.0 / npts
    r = 1.0
    rf = 0.5
    for i in range(npts):
        vs[i,0] = r * np.cos(i*ang+random.random()*rf)
        vs[i,1] = r * np.sin(i*ang+random.random()*rf)

    vs[:,:2] += random.random()*0.5

    if False:
        wvs = np.vstack((vs,vs[0,:]))
        ctr = centroid(vs).reshape((1,3))
        show(vs)
        for i in range(npts):
            tri = np.vstack((wvs[i:i+2,:], ctr))
            show(tri)

