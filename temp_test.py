import taichi as ti
from taichi.math import *
import numpy as np
ti.init(arch=ti.gpu)

@ti.data_oriented
class TiArray:
    def __init__(self, n):
        self.x = ti.field(dtype=ti.i32, shape=n)

    @ti.kernel
    def inc(self):
        print(self.x)
        for i in self.x:
            self.x[i] += 1

a = TiArray(32)
a.inc()
print(a.x.to_numpy())