import matplotlib.pyplot as plt
import numpy as np

class Edge:
    def __init__(self, id):
        self.id = id
        self.he = None

class Face:
    def __init__(self, id):
        self.id = id
        self.hes = []

class HalfEdge:
    def __init__(self, vtx, p0, p1):
        # indices of vertices
        self.p0 = p0
        self.p1 = p1

        self.next = None
        self.twin = None
        if (type(vtx).__module__ != np.__name__):
            print("error: invalid vertex {}: {}".format(p0,vtx))
        self.vertex = vtx

        self.face = None
        self.edge = None
        self.dual = None
        self.os = None

def prev(he):
    hi = he.next
    while hi.next != he:
        hi = hi.next
    return hi

def midpoint(he):
    return (he.vertex + he.next.vertex)/2.

def isClose(a,b):
    return np.sum(np.abs(a-b)) < 1e-10

def visualise(hes,oc='b'):
    plt.figure(1)
    for he in hes:
        c = 'b-'
        if he.twin is None:
            c = 'r-'
        if oc != 'b-': c = oc
        plt.plot([he.vertex[0],he.next.vertex[0]],
                 [he.vertex[1],he.next.vertex[1]],c)
    plt.show(block=False)

def area(face):
    val = 0.0
    hi = face.hes[0]
    val += (hi.vertex[0]*hi.next.vertex[1]
            - hi.next.vertex[0]*hi.vertex[1])
    hi = hi.next
    while hi != face.hes[0]:
        val += (hi.vertex[0]*hi.next.vertex[1]
                - hi.next.vertex[0]*hi.vertex[1])
        hi = hi.next

    return 0.5 * val

def centroid(face):
    ctr= np.zeros((3))
    A = area(face)
    hi = face.hes[0]
    ctr[0] += (hi.vertex[0]+hi.next.vertex[0])*(
            hi.vertex[0]*hi.next.vertex[1]
            - hi.next.vertex[0]*hi.vertex[1])
    ctr[1] += (hi.vertex[1]+hi.next.vertex[1])*(
            hi.vertex[0]*hi.next.vertex[1]
            - hi.next.vertex[0]*hi.vertex[1])
    hi = hi.next
    while hi != face.hes[0]:
        ctr[0] += (hi.vertex[0]+hi.next.vertex[0])*(
                hi.vertex[0]*hi.next.vertex[1]
                - hi.next.vertex[0]*hi.vertex[1])
        ctr[1] += (hi.vertex[1]+hi.next.vertex[1])*(
                hi.vertex[0]*hi.next.vertex[1]
                - hi.next.vertex[0]*hi.vertex[1])
        hi = hi.next

    return ctr / (6. * A)
