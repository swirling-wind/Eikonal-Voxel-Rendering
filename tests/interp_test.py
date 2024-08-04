from common.math_utils import *

ti.init(ti.gpu, debug=True) 

@ti.data_oriented
class Interpolation:
    def __init__(self):
        self.tex2d = ti.field(dtype=ti.f32, shape=(2, 2))
        self.tex3d = ti.field(dtype=ti.f32, shape=(2, 2, 2))

        self.tex2d[0,0] = 1
        self.tex2d[1,0] = 2
        self.tex2d[0,1] = 3
        self.tex2d[1,1] = 4

        self.tex3d[0,0,0] = 1
        self.tex3d[1,0,0] = 2
        self.tex3d[0,1,0] = 3
        self.tex3d[1,1,0] = 4
        self.tex3d[0,0,1] = 5
        self.tex3d[1,0,1] = 6
        self.tex3d[0,1,1] = 7
        self.tex3d[1,1,1] = 8

    @ti.func
    def bilinear_interp(self, tex2d, coord: tm.vec2, x_bound: ti.i32, y_bound: ti.i32):
        x, y = coord
        x0= ti.floor(x, ti.i32)
        y0= ti.floor(y, ti.i32)
        x1, y1 = x0 + 1, y0 + 1
        fx, fy = x - x0, y - y0
        # boundary check
        x0 = tm.max(0, tm.min(x_bound - 1, x0))
        x1 = tm.max(0, tm.min(x_bound - 1, x1))
        y0 = tm.max(0, tm.min(y_bound - 1, y0))
        y1 = tm.max(0, tm.min(y_bound - 1, y1))
        c00 = tex2d[x0, y0]
        c10 = tex2d[x1, y0] 
        c01 = tex2d[x0, y1]
        c11 = tex2d[x1, y1]
        cx0 = tm.mix(c00, c10, fx) 
        cx1 = tm.mix(c01, c11, fx)
        return tm.mix(cx0, cx1, fy)

    @ti.func
    def trilinear_interp(self, tex3d, coord: tm.vec3, x_bound: ti.i32, y_bound: ti.i32, z_bound: ti.i32):
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
    
    @ti.kernel
    def test2d(self):
        print(self.bilinear_interp(self.tex2d, tm.vec2(0.9, 0.9), 2, 2), "\n\n")


    @ti.kernel
    def test3d(self):
        print(self.trilinear_interp(self.tex3d, tm.vec3(0.0, 1.1, 0.9), 2, 2, 2))

    @ti.kernel
    def test_floor(self):
        print(ti.floor(-2.3, ti.i32))

interp = Interpolation()


# def lerp(x0,x1,t):
#     return x0*t+x1*(1-t)

# def lerp2d(vec2[vec2,vec2],t0,t1):



interp.test2d()
interp.test3d()
interp.test_floor()