import taichi as ti
ti.init(arch=ti.gpu)

@ti.kernel
def foo():
    inf = 1e10
    near_int = -inf
    far_int = inf

    print(near_int)
    print(far_int)

foo()
