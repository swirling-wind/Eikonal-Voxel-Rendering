import math
import taichi as ti
import numpy as np
import taichi.math as tm

eps = 1e-4
inf = 1e10

@ti.func
def trilinear_interp(tex3d, coord: tm.vec3):
    x, y, z = coord
    x0= ti.floor(x, ti.i32)
    y0= ti.floor(y, ti.i32)
    z0= ti.floor(z, ti.i32)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    fx, fy, fz = x - x0, y - y0, z - z0
    c000 = tex3d[x0, y0, z0]
    c100 = tex3d[x1, y0, z0]
    c010 = tex3d[x0, y1, z0]
    c110 = tex3d[x1, y1, z0]
    c001 = tex3d[x0, y0, z1]
    c101 = tex3d[x1, y0, z1]
    c011 = tex3d[x0, y1, z1]
    c111 = tex3d[x1, y1, z1]
    cx00 = tm.mix(c000, c100, fx)
    cx10 = tm.mix(c010, c110, fx)
    cx01 = tm.mix(c001, c101, fx)
    cx11 = tm.mix(c011, c111, fx)
    cxy0 = tm.mix(cx00, cx10, fy)
    cxy1 = tm.mix(cx01, cx11, fy)
    return tm.mix(cxy0, cxy1, fz)

@ti.func
def out_dir(n: tm.vec3):
    u = ti.Vector([1.0, 0.0, 0.0])
    if ti.abs(n[1]) < 1 - 1e-3:
        u = n.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
    v = n.cross(u)
    phi = 2 * math.pi * ti.random(ti.f32)
    r = ti.random(ti.f32)
    ay = ti.sqrt(r)
    ax = ti.sqrt(1 - r)
    return ax * (ti.cos(phi) * u + ti.sin(phi) * v) + ay * n


@ti.func
def ray_aabb_intersection(box_min: tm.vec3, box_max: tm.vec3, 
                          o: tm.vec3, d: tm.vec3):
    intersect = 1

    near_int = -inf
    far_int = inf

    for i in ti.static(range(3)):
        if d[i] == 0: # when ray parallel to plane
            if o[i] < box_min[i] or o[i] > box_max[i]:
                intersect = 0
        else:
            i1 = (box_min[i] - o[i]) / d[i]
            i2 = (box_max[i] - o[i]) / d[i]

            new_far_int = ti.max(i1, i2)
            new_near_int = ti.min(i1, i2)

            far_int = ti.min(new_far_int, far_int)
            near_int = ti.max(new_near_int, near_int)

    if near_int > far_int:
        intersect = 0
    return intersect, near_int, far_int

@ti.func
def ray_aabb_intersection_point(box_min: tm.vec3, box_max: tm.vec3,
                                eye_pos: tm.vec3, dir: tm.vec3):
    intersect = 1

    near_int = -inf
    far_int = inf
    near_point = ti.Vector([0.0, 0.0, 0.0])
    # far_point = ti.Vector([0.0, 0.0, 0.0])

    for i in ti.static(range(3)):
        if dir[i] == 0:
            if eye_pos[i] < box_min[i] or eye_pos[i] > box_max[i]:
                intersect = 0
        else:
            i1 = (box_min[i] - eye_pos[i]) / dir[i]
            i2 = (box_max[i] - eye_pos[i]) / dir[i]

            new_far_int = ti.max(i1, i2)
            new_near_int = ti.min(i1, i2)

            far_int = ti.min(new_far_int, far_int)
            near_int = ti.max(new_near_int, near_int)

    if near_int > far_int:
        intersect = 0

    near_point = eye_pos + dir * near_int
    return intersect, near_point

def np_normalize(v):
    # https://stackoverflow.com/a/51512965/12003165
    return v / np.sqrt(np.sum(v**2))


def np_rotate_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    # https://stackoverflow.com/a/6802723/12003165
    axis = np_normalize(axis)
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), 0],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), 0],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, 0],
                     [0, 0, 0, 1]])
