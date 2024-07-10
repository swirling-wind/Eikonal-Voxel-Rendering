from scene import Scene
import taichi as ti
from taichi.math import *

ti.init(arch=ti.cpu)

@ti.kernel
def vector_matrix():
    u = ti.Vector([1.0, 2.0])
    m = ti.Matrix([[1.0, 2.0], [3.0, 4.0]])
    print("hello")
    print(m @ u)

vector_matrix()
