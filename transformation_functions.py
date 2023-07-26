import numpy as np

def rotmat(theta, u):
    # get values
    ux = u[0]
    uy = u[1]
    uz = u[2]

    # precompute
    sina = np.sin(theta)
    cosa = np.cos(theta)

    # construct matrix
    R = np.array([[(1 - cosa) * ux * ux + cosa, (1 - cosa) * ux * uy - sina * uz, (1 - cosa) * ux * uz + sina * uy], 
                  [(1 - cosa) * ux * uy + sina * uz, (1 - cosa) * uy * uy + cosa, (1 - cosa) * uy * uz - sina * ux],
                  [(1 - cosa) * ux * uz - sina * uy, (1 - cosa) * uy * uz + sina * ux, (1 - cosa) * uz * uz + cosa]])
    
    return R

def rotate_translate(cp, theta, u, A, t):
    # get rotation matrix
    R = rotmat(theta, u / np.linalg.norm(u))
    # if there are many points
    if len(cp.shape) > 1:
        points = []
        cp = np.transpose(cp)
        for point in cp:
            point = point - A
            res = np.dot(R, point) + t
            res = res + A
            points.append(res)
        points = np.array(points)
        points = np.transpose(points)
        return points
    else:
        # single point
        cp = cp - A
        res = np.dot(R, cp) + t
        res = res + A
        return res

def change_coordinate_system(cp, r, c0):
    # instead of using inverse operation, which is computationally expensive, 
    # we use the transpose operation, since they are equivalent in this context
    r = r.T
    # if there are many points
    if len(cp.shape) > 1:
        points = []
        cp = np.transpose(cp)
        for point in cp:
            res = np.dot(r, point) + np.squeeze(c0)
            points.append(res)
        points = np.array(points)
        points = np.transpose(points)
        return points
    else:
        # single point
        res = np.dot(r, cp) + c0
        return res