# Ray marching in voxels
import taichi as ti
import taichi.math as tm
from common.math_utils import *

# MAX_RAY_DEPTH = 3
# DIS_LIMIT = 100

use_directional_light = True

@ti.data_oriented
class Renderer:
    def __init__(self, 
                 hdr_res: tuple[int, int], hdr_name: str,
                 dx: float, image_res: tuple[int, int],                 
                 up: tuple, exposure: float=3.0):
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

        # Viewing ray
        self.light_direction = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.light_direction_noise = ti.field(dtype=ti.f32, shape=())
        self.light_color = ti.Vector.field(3, dtype=ti.f32, shape=())

        # For debugging
        # self.cast_voxel_hit = ti.field(ti.i32, shape=())
        # self.cast_voxel_index = ti.Vector.field(3, ti.i32, shape=())

        # Scene settings
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

        # HDR map
        self.hdr_img = ti.Vector.field(3, dtype=ti.f32)
        self.hdr_img_size = hdr_res

        ti.root.dense(ti.ij, image_res).place(self.color_buffer)
        ti.root.dense(ti.ij, self.hdr_img_size).place(self.hdr_img)
        ti.root.dense(ti.ijk,
                      self.voxel_grid_res).place(self.voxel_color,
                                                self.voxel_material,
                                                self.ior,

                                                self.grad,

                                                self.irrad,
                                                self.loc_dir,

                                                self.atten,
                                                self.scatter_strength,
                                                # self.emission,
                                                offset=voxel_grid_offset)

        self._rendered_image = ti.Vector.field(3, float, image_res)
        self.set_up(*up)
        self.set_fov(0.8) # 0.23

        self.floor_height[None] = 0
        self.floor_color[None] = (1, 1, 1)

        self.ior.fill(1.0)

        self.atten.fill(0.0)
        self.scatter_strength.fill(0.0)

        hdr_image = ti.tools.imread("assets/" + hdr_name).astype('float32')
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

    def set_directional_light(self, direction: tuple, light_direction_noise: float,
                              light_color: tuple):
        direction_norm = (direction[0]**2 + direction[1]**2 +
                          direction[2]**2)**0.5
        self.light_direction[None] = (direction[0] / direction_norm,
                                      direction[1] / direction_norm,
                                      direction[2] / direction_norm)
        self.light_direction_noise[None] = light_direction_noise
        self.light_color[None] = light_color

    @ti.func
    def _to_voxel_index(self, pos: tm.vec3) -> tm.vec3:
        p = pos * self.voxel_inv_dx # voxel_inv_dx is 1 / dx, equal to 64
        voxel_index = ti.floor(p).cast(ti.i32) # type: ignore
        return voxel_index
    
    @ti.func
    def ipos_inside_grid(self, ipos: tm.vec3) -> bool:
        return ipos.min() >= -self.voxel_grid_res // 2 and ipos.max(
        ) < self.voxel_grid_res // 2
    
    @ti.func
    def pos_inside_grid(self, pos: tm.vec3) -> bool:
        return pos.min() >= -1.0 and pos.max() < 1.0 - self.voxel_dx


    @ti.func
    def ipos_inside_particle_grid(self, ipos: tm.vec3) -> bool:
        # Check if the voxel is inside the bounding box
        pos = ipos * self.voxel_dx
        return self.bbox[0][0] <= pos[0] and pos[0] < self.bbox[1][
            0] and self.bbox[0][1] <= pos[1] and pos[1] < self.bbox[1][
                1] and self.bbox[0][2] <= pos[2] and pos[2] < self.bbox[1][2]
    
    @ti.func
    def pos_inside_particle_grid(self, pos: tm.vec3) -> bool:
        # Check if the voxel is inside the bounding box
        return self.bbox[0][0] <= pos[0] and pos[0] < self.bbox[1][
            0] and self.bbox[0][1] <= pos[1] and pos[1] < self.bbox[1][
                1] and self.bbox[0][2] <= pos[2] and pos[2] < self.bbox[1][2]

    @ti.kernel
    def set_camera_pos(self, x: ti.f32, y: ti.f32, z: ti.f32):
        self.camera_pos[None] = ti.Vector([x, y, z])

    @ti.kernel
    def set_up(self, x: ti.f32, y: ti.f32, z: ti.f32):
        self.up[None] = ti.Vector([x, y, z]).normalized()

    @ti.kernel
    def set_look_at(self, x: ti.f32, y: ti.f32, z: ti.f32):
        self.look_at[None] = ti.Vector([x, y, z])

    @ti.kernel
    def set_fov(self, fov: ti.f32):
        self.fov[None] = fov

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

    @ti.func
    def intersect_floor(self, p, d):
        intersection_point = tm.vec3(inf)
        if d[1] < -eps:
            dist = (self.floor_height[None] - p[1]) / d[1]
            intersection_point = p + dist * d
        return intersection_point


    @ti.kernel
    def ray_marching(self):        
        for u, v in self.color_buffer:
            self.marching(u, v)

    @ti.func
    def marching(self, u: ti.i32, v: ti.i32):
        cam_pos = self.camera_pos[None]
        d = self.get_cast_dir(u, v)
        
        bbox_min = self.bbox[0]
        bbox_max = self.bbox[1]
        inter, near_pos = ray_aabb_intersection_point(bbox_min, bbox_max, cam_pos, d)

        hit_floor = 0
        floor_inv_pos = cam_pos

        boundary = False
        
        contrib = ti.Vector([0.0, 0.0, 0.0]) # each value range: [0,1]

        if inter:            
            pos = near_pos # Ray start from intersection point inside the bounding box
  
            MAX_MARCHING_STEPS = 240

            I = tm.vec3(0.0)
            Is = tm.vec3(0.0)
            Ir = tm.vec3(0.0)

            A = 0.0 # absorption (e.g: A.rgb)
            T = 1.0 # transmittance (e.g: T.rgb)
            R = 0.0 # reflection (e.g: R.rgb)

            step_size = self.voxel_dx
            n = 1.0

            for _cur_step in range(MAX_MARCHING_STEPS):
                inv_pos = pos * self.voxel_inv_dx # voxel_inv_dx is 1 / dx, equal to 64

                gradient = trilinear_interp(self.grad, inv_pos)     # 3D vector

                voxelIrrad = trilinear_interp(self.irrad, inv_pos)  # 1D scalar
                loc_dir = trilinear_interp(self.loc_dir, inv_pos)   # 3D vector

                voxelAtt = trilinear_interp(self.atten, inv_pos)    # 1D scalar
                scatterStrength = trilinear_interp(self.scatter_strength, inv_pos) # 1D scalar

                # --------------------------------------
                # Compute Attenuation factor
                A += voxelAtt / 2.0 # * step_size * self.voxel_inv_dx

                # --------------------------------------
                # Compute scattering term 
                ANISOTROPY_FACTOR = 0.1 # Higher value means more anisotropic scattering
                ANISOTROPY_FACTOR_SQUARED = ANISOTROPY_FACTOR**2
                ft = 1 - 2 * ANISOTROPY_FACTOR * tm.dot(loc_dir, tm.normalize(d)) + ANISOTROPY_FACTOR_SQUARED
                Is = tm.vec3(voxelIrrad / 255.0 / 4.0) * 0.5 * (1 - ANISOTROPY_FACTOR_SQUARED) / tm.pow(ft, 1.5)

                # --------------------------------------
                # Compute new direction and refraction index
                oldPos = pos
                d += step_size * gradient / n
                pos += step_size * d / tm.pow(n, 2)
                n += tm.dot(gradient, pos - oldPos) * self.voxel_inv_dx

                # --------------------------------------
                # Compute Reflection Term
                oldT = T

                if tm.length(gradient) > 0.07 and not boundary:
                    FRESNEL_FACTOR = 0.5
                    VOXELAUX_A = 0.6

                    boundary = True

                    R = 1 / tm.pow(1 + ti.abs(tm.dot(tm.normalize(gradient), tm.normalize(d))), 2.0)
                    R = tm.mix(0.1, tm.min((tm.pow(R, 3) * VOXELAUX_A),  1.0), FRESNEL_FACTOR)
                    T = tm.mix(1, T * (1 - R), FRESNEL_FACTOR)
                    
                    # # Phong reflection model [NOT COMPATIBLE WITH THE FORMATION OF REFLeCTION MODEL]
                    # view_dir = -tm.normalize(d)
                    # light_dir = self.light_direction[None]
                    # normal = -tm.normalize(gradient)
                    # reflect_dir = tm.reflect(-light_dir, normal)
                    # Ir += tm.pow(tm.max(tm.dot(view_dir, reflect_dir), 0.0), 3.0)

                    # BRDF reflection model
                    reflect_dir = tm.reflect(tm.normalize(d), tm.normalize(gradient))
                    reflect_pos = pos
                    reflectionColor = tm.vec3(0.0)

                    # Check if the reflection ray intersects the floor plane and add the caustics
                    intersect_pos = self.intersect_floor(reflect_pos, reflect_dir)
                    # if the reflection ray intersects the floor plane, set the reflection color to the floor color and its irradiance
                    if intersect_pos[1] < 1.0 and self.pos_inside_particle_grid(intersect_pos):
                        # print(u, v, "\tintersect_pos: ", intersect_pos)
                        floor_irrad = trilinear_interp(self.irrad, intersect_pos * self.voxel_inv_dx)
                        reflectionColor = self.floor_color[None] + tm.vec3(floor_irrad / 255.0) * 3.0
                    else:  # else if not intersected, set the reflection color to the sky color
                        reflectionColor = self.sky_color(reflect_dir)

                    Ir += reflectionColor
                    # VOXELREFLECTIONDATA_RGB = tm.vec3(1.0)
                    # VOXELREFLECTIONDATA_A = 0.8
                     # tm.mix(reflectionColor, VOXELREFLECTIONDATA_RGB * reflectionColor, VOXELREFLECTIONDATA_A)
                else:
                    R = 0.0
                if tm.length(gradient) < 0.001:
                    boundary = False

                #  --------------------------------------
                # Compute combined intensity per voxel and compute final integral
                SCATTER_FACTOR = 0.2
                REFLECTION_FACTOR = 2.0
                Ic = scatterStrength * Is * SCATTER_FACTOR + Ir * R * REFLECTION_FACTOR
                remaining = tm.exp(-A) * oldT
                I += Ic * remaining
 
                # --------------------------------------
                # Russian Roulette [NO PERFORMANCE IMPROVEMENT]

                # A_THRESHOLD = 2.0
                # T_THRESHOLD = 0.15
                # if A > A_THRESHOLD or T < T_THRESHOLD:
                #     # the probability of continuation is based on the absorption and transmittance
                #     q = tm.clamp(remaining, 0.1, 1.0)
                #     if ti.random() > q:
                #         break
                #     else:
                #         # adjust the intensity and transmittance to ensure the integral is correct
                #         I /= q
                #         T /= q


                #  --------------------------------------
                # check if we are not outside of the volume
                if (not self.pos_inside_particle_grid(pos)): # (not self.pos_inside_grid(pos)): #
                    break

                if pos[1] <= self.floor_height[None]:
                    hit_floor = 1
                    floor_inv_pos = inv_pos
                    break

            if hit_floor: # hit the floor (add floor color and floor position's irradiance)
                floor_irrad = trilinear_interp(self.irrad, floor_inv_pos)
                floor_irrad_vec = tm.vec3(floor_irrad / 255.0)
                contrib = I + (self.floor_color[None] + floor_irrad_vec) * tm.exp(-A)
            else: # enter the bounding box and finally hit the background
                contrib = I + self.sky_color(d) * tm.exp(-A)

        else: # directly hit the background without entering the bounding box
            contrib = self.sky_color(d)
        
        self.color_buffer[u, v] += contrib

    @ti.kernel
    def _render_to_image(self, samples: ti.i32):
        for i, j in self.color_buffer:
            u = 1.0 * i / self.image_res[0]
            v = 1.0 * j / self.image_res[1]

            darken = 1.0 - self.vignette_strength * max((ti.sqrt(
                (u - self.vignette_center[0])**2 +
                (v - self.vignette_center[1])**2) - self.vignette_radius), 0)

            for c in ti.static(range(3)):
                self._rendered_image[i, j][c] = ti.sqrt(
                    self.color_buffer[i, j][c] * darken * self.exposure /
                    samples)

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
        
        self.bbox[0][1] = self.floor_height[None] - self.voxel_dx

    def reset_framebuffer(self):
        self.current_spp = 0
        self.color_buffer.fill(0)

    def fetch_image(self) -> ti.Field:
        self._render_to_image(self.current_spp)
        return self._rendered_image

    @staticmethod
    @ti.func
    def to_vec3u(c: tm.vec3) -> tm.vec3:
        c = ti.math.clamp(c, 0.0, 1.0)
        r = ti.Vector([ti.u8(0), ti.u8(0), ti.u8(0)])
        for i in ti.static(range(3)):
            r[i] = ti.cast(c[i] * 255, ti.u8)
        return r

    @staticmethod
    @ti.func
    def to_vec3(c):
        r = ti.Vector([0.0, 0.0, 0.0])
        for i in ti.static(range(3)):
            r[i] = ti.cast(c[i], ti.f32) / 255.0
        return r
    
    @staticmethod
    @ti.func
    def round_idx(idx_: tm.vec3) -> tm.vec3:
        idx = ti.cast(idx_, ti.f32)
        return ti.Vector(
            [ti.round(idx[0]), # type: ignore
             ti.round(idx[1]), # type: ignore
             ti.round(idx[2])]).cast(ti.i32) # type: ignore
    
    @ti.func
    def set_voxel(self, idx, mat, color: tm.vec3, ior=1.0):
        self.voxel_material[idx] = ti.cast(mat, ti.i8)
        self.voxel_color[idx] = self.to_vec3u(color)
        self.ior[idx] = ior

    @ti.func
    def set_voxel_data(self, idx, atten: ti.f32, scatter_strength: ti.f32):
        self.atten[idx] = atten
        self.scatter_strength[idx] = scatter_strength