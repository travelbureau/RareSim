# https://github.com/kroitor/gjk.c
import numpy as np
import numba


@numba.njit
def perpendicular(pt):
    temp = pt[0]
    pt[0] = pt[1]
    pt[1] = -1*temp
    return pt


@numba.njit
def tripleProduct(a, b, c):
    ac = a.dot(c)
    bc = b.dot(c)
    return b*ac - a*bc


@numba.njit
def avgPoint(vertices):
    return np.sum(vertices, axis=0)/vertices.shape[0]


@numba.njit
def indexOfFurthestPoint(vertices, d):
    return np.argmax(vertices.dot(d))


@numba.njit
def support(vertices1, vertices2, d):
    i = indexOfFurthestPoint(vertices1, d)
    j = indexOfFurthestPoint(vertices2, -d)
    return vertices1[i] - vertices2[j]


@numba.njit
def collision(vertices1, vertices2):
    index = 0
    simplex = np.empty((3, 2))

    position1 = avgPoint(vertices1)
    position2 = avgPoint(vertices2)

    d = position1 - position2

    if d[0] == 0 and d[1] == 0:
        d[0] = 1.0

    a = support(vertices1, vertices2, d)
    simplex[index, :] = a

    if d.dot(a) <= 0:
        return 0

    d = -a

    iter_count = 0
    while iter_count < 1e3:
        a = support(vertices1, vertices2, d)
        index += 1
        simplex[index, :] = a
        if d.dot(a) <= 0:
            return 0

        ao = -a

        if index < 2:
            b = simplex[0, :]
            ab = b-a
            d = tripleProduct(ab, ao, ab)
            if np.linalg.norm(d) < 1e-10:
                d = perpendicular(ab)
            continue

        b = simplex[1, :]
        c = simplex[0, :]
        ab = b-a
        ac = c-a

        acperp = tripleProduct(ab, ac, ac)

        if acperp.dot(ao) >= 0:
            d = acperp
        else:
            abperp = tripleProduct(ac, ab, ab)
            if abperp.dot(ao) < 0:
                return 1
            simplex[0, :] = simplex[1, :]
            d = abperp

        simplex[1, :] = simplex[2, :]
        index -= 1

        iter_count += 1
    assert(1 == 0)
    return 0
