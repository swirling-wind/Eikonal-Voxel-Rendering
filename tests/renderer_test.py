# Ray marching in voxels
import taichi as ti
import taichi.math as tm
from common.math_utils import *

VOXEL_DX = 1 / 64
SCREEN_RES = (800, 600)
UP_DIR = (0, 1, 0)

class Camera:
    def __init__(self, window, up):
        self._window = window
        self._camera_pos = np.array((4.0, 4.0, 4.0))
        self._lookat_pos = np.array((0.0, 0.0, 0.0))
        self._up = np_normalize(np.array(up))
        self._last_mouse_pos = None

    @property
    def mouse_exclusive_owner(self):
        return True

    def update_camera(self):
        res = self._update_by_wasd()
        res = self._update_by_mouse() or res
        return res

@ti.data_oriented
class Renderer:
    def __init__(self, dx: float, image_res: tuple[int, int], up: tuple, voxel_edges: float, exposure: float=3.0):
        self.image_res = image_res
        self.aspect_ratio = image_res[0] / image_res[1]
        self.vignette_strength = 0.9
        self.vignette_radius = 0.0
        self.vignette_center = [0.5, 0.5]
        self.current_spp = 0

        # Framebuffer and box of the scene
        self.color_buffer = ti.Vector.field(3, dtype=ti.f32)
        self.bbox = ti.Vector.field(3, dtype=ti.f32, shape=2)
        self.fov = ti.field(dtype=ti.f32, shape=())

        # Voxel grid
        self.voxel_color = ti.Vector.field(3, dtype=ti.u8)
        self.voxel_material = ti.field(dtype=ti.i8)
        self.ior = ti.field(dtype=ti.f32)

        self.grad = ti.Vector.field(3, dtype=ti.f32)
        self.irrad = ti.field(dtype=ti.f32)
        self.loc_dir = ti.Vector.field(3, dtype=ti.f32)
        
        self.atten = ti.field(dtype=ti.f32)
        self.scatter_strength = ti.field(dtype=ti.f32)
        self.anisotropy_factor = ti.field(dtype=ti.f32)
        self.opaque = ti.field(dtype=ti.i8)

        # HDR map
        self.hdr_img = ti.Vector.field(3, dtype=ti.f32)
        self.hdr_img_size = (3200, 1600)

        # Scene settings
        self.voxel_edges = voxel_edges
        self.exposure = exposure
        self.camera_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.look_at = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.up = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.floor_height = ti.field(dtype=ti.f32, shape=())
        self.floor_color = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.background_color = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.voxel_dx = dx
        self.voxel_inv_dx = 1 / dx
        # Note that voxel_inv_dx == voxel_grid_res iff the box has width = 1
        self.voxel_grid_res = 128
        voxel_grid_offset = [-self.voxel_grid_res // 2 for _ in range(3)]

        ti.root.dense(ti.ij, image_res).place(self.color_buffer)
        ti.root.dense(ti.ij, self.hdr_img_size).place(self.hdr_img)
        ti.root.dense(ti.ijk,
                      self.voxel_grid_res).place(self.ior,

                                                self.grad,
                                                self.irrad,

                                                self.atten,
                                                self.scatter_strength,
                                                offset=voxel_grid_offset)

        # hdr_image = ti.tools.imread('assets/limpopo_golf_course_3k.hdr').astype('float32')
        hdr_image = ti.tools.imread('assets/Tokyo_BigSight_3k.hdr').astype('float32')

        self.hdr_img.from_numpy(hdr_image / 255)
        self.hdr_process(self.exposure, 2.2)

    @ti.kernel
    def hdr_process(self, exposure: float, gamma: float):
        for i, j in self.hdr_img:
            color = self.hdr_img[i, j] * exposure
            color = pow(color, tm.vec3(gamma))
            self.hdr_img[i, j] = color
    
    @staticmethod
    @ti.func
    def sample_spherical_map(v: tm.vec3) -> tm.vec2:
        uv  = tm.vec2(tm.atan2(v.z, v.x), tm.asin(v.y))
        uv *= tm.vec2(0.5 / tm.pi, 1 / tm.pi)
        uv += 0.5
        return uv
    
    @ti.func
    def hdr_texture(self, uv: tm.vec2) -> tm.vec3:
        x = int(uv.x * (self.hdr_img_size[0] - 1))
        y = int(uv.y * (self.hdr_img_size[1] - 1))
        return self.hdr_img[x, y]

    @ti.func
    def sky_color(self, dir: tm.vec3) -> tm.vec3:
        uv = self.sample_spherical_map(dir)
        return self.hdr_texture(uv)

    @ti.func
    def pos_inside_particle_grid(self, pos: tm.vec3) -> bool:
        # Check if the voxel is inside the bounding box
        return self.bbox[0][0] <= pos[0] and pos[0] < self.bbox[1][
            0] and self.bbox[0][1] <= pos[1] and pos[1] < self.bbox[1][
                1] and self.bbox[0][2] <= pos[2] and pos[2] < self.bbox[1][2]

    @ti.func
    def get_cast_dir(self, u: ti.i32, v: ti.i32) -> tm.vec3:
        fov = self.fov[None]
        d = (self.look_at[None] - self.camera_pos[None]).normalized()
        fu = (2 * fov * (u + ti.random(ti.f32)) / self.image_res[1] -
              fov * self.aspect_ratio - 1e-5)
        fv = 2 * fov * (v + ti.random(ti.f32)) / self.image_res[1] - fov - 1e-5
        du = d.cross(self.up[None]).normalized()
        dv = du.cross(d).normalized()
        d = (d + fu * du + fv * dv).normalized()
        return d

    @ti.kernel
    def ray_marching(self):        
        for u, v in self.color_buffer:
            self.marching(u, v)

    @ti.func
    def marching(self, u: ti.i32, v: ti.i32):
        cam_pos = self.camera_pos[None]
        d = self.get_cast_dir(u, v)
        
        # Leave safe margin to avoid self-intersection
        bbox_min = self.bbox[0]
        bbox_max = self.bbox[1]
        inter, near_pos = ray_aabb_intersection_point(bbox_min, bbox_max, cam_pos, d)

        hit_floor = 0
        floor_inv_pos = cam_pos
        
        contrib = ti.Vector([0.0, 0.0, 0.0]) # each value range: [0,1]

        if inter:            
            pos = near_pos # Ray start from intersection point inside the bounding box to accelerate
            MAX_MARCHING_STEPS = 200

            I = tm.vec3(0.0)
            Is = tm.vec3(0.0)

            A = 0.0 # absorption (e.g: A.rgb)

            for _cur_step in range(MAX_MARCHING_STEPS):
                inv_pos = pos * self.voxel_inv_dx # voxel_inv_dx is 1 / dx, equal to 64
                voxelIrrad = trilinear_interp(self.irrad, inv_pos)
                voxelAtt = trilinear_interp(self.atten, inv_pos)
                scatterStrength = trilinear_interp(self.scatter_strength, inv_pos)

                # --------------------------------------
                # Compute Attenuation factor
                A += voxelAtt

                # --------------------------------------
                # Compute scattering term 
                Is = tm.vec3(voxelIrrad / 255.0 / 3.0)

                #  --------------------------------------
                # Compute combined intensity per voxel and compute final integral
                Ic = scatterStrength * Is
                I += Ic * tm.exp(-A)

                #  --------------------------------------
                # check if we are not outside of the volume
                if (not self.pos_inside_particle_grid(pos)):
                    break

                if pos[1] <= self.floor_height[None]:
                    hit_floor = 1
                    floor_inv_pos = inv_pos
                    break

            if hit_floor: # hit the floor (add floor color and floor position's irradiance)
                floor_irrad = trilinear_interp(self.irrad, floor_inv_pos)
                floor_irrad_vec = tm.vec3(floor_irrad / 255.0)
                contrib = I + (self.floor_color[None] + floor_irrad_vec * 3.0) * tm.exp(-A)
            else: # enter the bounding box and finally hit the background
                contrib = I + self.sky_color(d) * tm.exp(-A)

        else: # directly hit the background without entering the bounding box
            contrib = self.sky_color(d) # self.background_color[None]
        
        self.color_buffer[u, v] += contrib

    @ti.kernel
    def recompute_bbox(self):
        for d in ti.static(range(3)):
            self.bbox[0][d] = 1e9
            self.bbox[1][d] = -1e9
        for I in ti.grouped(self.voxel_material):
            if self.voxel_material[I] != 0:
                for d in ti.static(range(3)):
                    ti.atomic_min(self.bbox[0][d], (I[d] - 1) * self.voxel_dx)
                    ti.atomic_max(self.bbox[1][d], (I[d] + 2) * self.voxel_dx)
        
        # the bounding box is above the floor, lower it
        self.bbox[0][1] = self.bbox[0][1] - 2 * self.voxel_dx

    def reset_framebuffer(self):
        self.current_spp = 0
        self.color_buffer.fill(0)
        
if __name__ == "__main__":
    voxel_edges=0.06,
    exposure=3.0
    renderer = Renderer(dx=VOXEL_DX,
                                 image_res=SCREEN_RES,
                                 up=UP_DIR,
                                 voxel_edges=voxel_edges,
                                 exposure=exposure)
    window = ti.ui.Window("Ray marching",
                                   (1200, 900),
                                   vsync=True)
    camera = Camera(window, up=UP_DIR)
    renderer.set_camera_pos(*camera.position)
    renderer.set_look_at(*camera.look_at)
    renderer.reset_framebuffer()
    renderer.recompute_bbox()
    canvas = window.get_canvas()
    # print(camera.position, camera.look_at)
    # rendered = False

    while window.running:
        should_reset_framebuffer = False            
        if camera.update_camera():
            renderer.set_camera_pos(*camera.position)
            look_at = camera.look_at
            renderer.set_look_at(*look_at)
            should_reset_framebuffer = True
        if should_reset_framebuffer:
            renderer.reset_framebuffer()
        
        # if not rendered:
        renderer.ray_marching()
        renderer.current_spp += 1

        img = renderer.fetch_image()
        canvas.set_image(img)
        window.show()