import taichi as ti
import taichi.math as tm
from common.math_utils import *
from taichi.types import vector, matrix

MAX_RAY_DEPTH = 3
use_directional_light = True

DIS_LIMIT = 100

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

        # Viewing ray
        self.light_direction = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.light_direction_noise = ti.field(dtype=ti.f32, shape=())
        self.light_color = ti.Vector.field(3, dtype=ti.f32, shape=())

        # For debugging
        # self.cast_voxel_hit = ti.field(ti.i32, shape=())
        # self.cast_voxel_index = ti.Vector.field(3, ti.i32, shape=())

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
        ti.root.dense(ti.ijk,
                      self.voxel_grid_res).place(self.voxel_color,
                                                self.voxel_material,
                                                self.ior,

                                                self.grad,
                                                self.irrad,
                                                self.loc_dir,

                                                self.atten,
                                                self.scatter_strength,
                                                self.anisotropy_factor,
                                                self.opaque,
                                                offset=voxel_grid_offset)

        self._rendered_image = ti.Vector.field(3, float, image_res)
        self.set_up(*up)
        self.set_fov(0.23)

        self.floor_height[None] = 0
        self.floor_color[None] = (1, 1, 1)

        self.ior.fill(1.0)

        self.atten.fill(0.0)
        self.scatter_strength.fill(0.0)


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
    def inside_grid(self, ipos: ti.i32) -> bool:
        return ipos.min() >= -self.voxel_grid_res // 2 and ipos.max(
        ) < self.voxel_grid_res // 2

    @ti.func
    def query_density(self, ipos: ti.i32) -> ti.f32:
        inside = self.inside_grid(ipos)
        ret = 0.0
        if inside:
            ret = self.voxel_material[ipos]
        else:
            ret = 0.0
        return ret

    @ti.func
    def _to_voxel_index(self, pos: tm.vec3) -> tm.vec3:
        p = pos * self.voxel_inv_dx # voxel_inv_dx is 1 / dx, equal to 64
        voxel_index = ti.floor(p).cast(ti.i32) # type: ignore
        return voxel_index

    @ti.func
    def voxel_surface_color(self, pos: tm.vec3):
        # p = pos * self.voxel_inv_dx
        # p -= ti.floor(p)
        voxel_index = self._to_voxel_index(pos)
        voxel_color = ti.Vector([0.0, 0.0, 0.0])
        is_light = 0
        if self.ipos_inside_particle_grid(voxel_index):
            voxel_color = self.voxel_color[voxel_index] * (1.0 / 255)
            if self.voxel_material[voxel_index] == 2:
                is_light = 1

        return voxel_color, is_light # [tm.vec3, ti.i32]

    @ti.func
    def floor_sdf(self, p: tm.vec3, d: tm.vec3) -> ti.f32:  # floor's sdf
        dist = inf
        if d[1] < -eps:
            dist = (self.floor_height[None] - p[1]) / d[1]
        return dist

    @ti.func
    def sdf_normal(self) -> tm.vec3:    # floor's normal
        return ti.Vector([0.0, 1.0, 0.0])  # up

    @ti.func
    def sdf_color(self) -> tm.vec3:   # floor's color
        return self.floor_color[None]

    @ti.func
    def dda_voxel(self, eye_pos: tm.vec3, d: tm.vec3):
        for i in ti.static(range(3)):
            if abs(d[i]) < 1e-6:
                d[i] = 1e-6
        rinv = 1.0 / d
        rsign = ti.Vector([0, 0, 0])
        for i in ti.static(range(3)):
            if d[i] > 0:
                rsign[i] = 1
            else:
                rsign[i] = -1

        bbox_min = self.bbox[0]
        bbox_max = self.bbox[1]
        inter, near, far = ray_aabb_intersection(bbox_min, bbox_max, eye_pos,
                                                 d)
        hit_distance = inf
        hit_light = 0
        normal = ti.Vector([0.0, 0.0, 0.0])
        c = ti.Vector([0.0, 0.0, 0.0])
        voxel_index = ti.Vector([0, 0, 0])
        if inter:
            near = max(0, near)

            pos = eye_pos + d * (near + 5 * eps)

            o = self.voxel_inv_dx * pos
            ipos = int(ti.floor(o))

            dis = (ipos - o + 0.5 + rsign * 0.5) * rinv
            running = 1
            i = 0
            hit_pos = ti.Vector([0.0, 0.0, 0.0])
            while running:
                last_sample = int(self.query_density(ipos))
                if not self.ipos_inside_particle_grid(ipos):
                    running = 0

                if last_sample:
                    mini = (ipos - o + ti.Vector([0.5, 0.5, 0.5]) -
                            rsign * 0.5) * rinv
                    hit_distance = mini.max() * self.voxel_dx + near
                    hit_pos = eye_pos + (hit_distance + 1e-3) * d
                    voxel_index = self._to_voxel_index(hit_pos)
                    c, hit_light = self.voxel_surface_color(hit_pos)
                    running = 0
                else:
                    mm = ti.Vector([0, 0, 0])
                    if dis[0] <= dis[1] and dis[0] < dis[2]:
                        mm[0] = 1
                    elif dis[1] <= dis[0] and dis[1] <= dis[2]:
                        mm[1] = 1
                    else:
                        mm[2] = 1
                    dis += mm * rsign * rinv
                    ipos += mm * rsign
                    normal = -mm * rsign
                i += 1
        return hit_distance, normal, c, hit_light, voxel_index

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

    @ti.func
    def next_hit(self, pos: tm.vec3, d: tm.vec3):
        closest = inf
        normal = ti.Vector([0.0, 0.0, 0.0])
        c = ti.Vector([0.0, 0.0, 0.0])
        hit_light = 0
        closest, normal, c, hit_light, vx_idx = self.dda_voxel(pos, d)

        ray_march_dist = self.floor_sdf(pos, d) # floor's distance function
        if ray_march_dist < DIS_LIMIT and ray_march_dist < closest:
            #  It's a floor hit and closer than voxel hit
            closest = ray_march_dist
            normal = self.sdf_normal()
            c = self.sdf_color()

        # if ti.static(True): # Highlight the selected voxel for debugging
        #     cast_vx_idx = tm.vec3(0,20,0) # The index of the voxel to highlight
        #     if all(cast_vx_idx == vx_idx):
        #         c = ti.Vector([1.0, 0.65, 0.0]) # orange color
        #         # For light sources, we actually invert the material
        #         # hit_light = 1 - hit_light
        return closest, normal, c, hit_light

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

    @ti.kernel
    def ray_marching(self):        
        for u, v in self.color_buffer:
            self.marching(u, v)

    @ti.func
    def trilinear_interp(self, tex3d, coord: tm.vec3):
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
    def marching(self, u: ti.i32, v: ti.i32):
        cam_pos = self.camera_pos[None]
        d = self.get_cast_dir(u, v)
        
        # Leave safe margin to avoid self-intersection
        bbox_min = self.bbox[0]
        bbox_max = self.bbox[1]
        inter, near_pos = ray_aabb_intersection_point(bbox_min, bbox_max, cam_pos, d)

        hit_floor = 0
        floor_inv_pos = cam_pos

        boundary = False
        
        contrib = ti.Vector([0.0, 0.0, 0.0]) # each value range: [0,1]

        if inter:            
            pos = near_pos # Ray start from intersection point inside the bounding box
  
            MAX_MARCHING_STEPS = 150

            I = tm.vec3(0.0)
            Is = tm.vec3(0.0)
            Ir = tm.vec3(0.0)

            A = 0.0 # absorption (e.g: A.rgb)
            T = 1.0 # transmittance (e.g: T.rgb)
            R = 0.0 # reflection (e.g: R.rgb)

            step_size = self.voxel_dx
            n = 1.0

            for _cur_step in range(MAX_MARCHING_STEPS):
                inv_pos = pos * self.voxel_inv_dx
                gradient = self.trilinear_interp(self.grad, inv_pos)
                loc_dir = self.trilinear_interp(self.loc_dir, inv_pos)
                
                voxelIrrad = self.trilinear_interp(self.irrad, inv_pos)
                voxelAtt = self.trilinear_interp(self.atten, inv_pos)
                scatterStrength = self.trilinear_interp(self.scatter_strength, inv_pos)

                # --------------------------------------
                # Compute Attenuation factor
                A += step_size * voxelAtt * self.voxel_inv_dx

                # --------------------------------------
                # Compute scattering term
                ANISOTROPY_FACTOR = 0.25
                ANISOTROPY_FACTOR_SQUARED = ANISOTROPY_FACTOR**2
                ft = 1 - 2 * ANISOTROPY_FACTOR * tm.dot(loc_dir, tm.normalize(d)) + ANISOTROPY_FACTOR_SQUARED
                Is = tm.vec3(voxelIrrad / 255.0) * 0.5 * (1 - ANISOTROPY_FACTOR_SQUARED) / tm.pow(ft, 1.5)

                # --------------------------------------
                # Compute new direction and refraction index
                oldPos = pos
                d += step_size * gradient
                pos += step_size * d / n
                n += tm.dot(gradient, pos - oldPos) * self.voxel_inv_dx

                # --------------------------------------
                # Compute Reflection Term
                oldT = T

                if tm.length(gradient) > 0.1 and not boundary:
                    FRESNEL_FACTOR = 0.5
                    VOXELAUX_A = 0.9

                    boundary = True

                    R = 1 / tm.pow(1 + ti.abs(tm.dot(tm.normalize(gradient), tm.normalize(d))), 2.0)
                    R = tm.mix(0.1, tm.min((tm.pow(R, 3) * VOXELAUX_A),  1.0), FRESNEL_FACTOR)
                    T = tm.mix(1, T * (1 - R), FRESNEL_FACTOR)
                    
                    view_dir = -tm.normalize(d)
                    light_dir = tm.vec3(0.0, 1.0, 0.0)
                    normal = -tm.normalize(gradient)
                    reflect_dir = tm.reflect(-light_dir, normal)
                    Ir = tm.pow(tm.max(tm.dot(view_dir, reflect_dir), 0.0), 4.0) * tm.vec3(1.0, 0.0, 0.0)

                    # dir = tm.reflect(tm.normalize(d), tm.normalize(gradient))
                    # reflectionColor = tm.vec3(1.0, 0.0, 0.0) # self.background_color[None]
                    # VOXEL_REFLECTION_DATA_RGB = tm.vec3(0.8)
                    # VOXEL_REFLECTION_DATA_A = 1
                    # # Ir = tm.vec3(0.0)
                    # Ir = tm.mix(reflectionColor, VOXEL_REFLECTION_DATA_RGB * reflectionColor, VOXEL_REFLECTION_DATA_A)
                else:
                    R = 0.0
                if tm.length(gradient) < 0.001:
                    boundary = False

                #  --------------------------------------
                # Compute combined intensity per voxel and compute final integral
                Ic = scatterStrength * Is + Ir * R * 5
                I += Ic * tm.exp(-A) * oldT

                #  --------------------------------------
                # check if we are not outside of the volume
                if (not self.pos_inside_particle_grid(pos)): # or (not self.inside_grid(voxel_index)):
                    break

                if pos[1] <= self.floor_height[None]:
                    hit_floor = 1
                    floor_inv_pos = inv_pos
                    break

            if hit_floor: # hit the floor (add floor color and floor position's irradiance)
                floor_irrad = self.trilinear_interp(self.irrad, floor_inv_pos)
                floor_irrad_vec = tm.vec3(floor_irrad / 255.0)
                contrib = I + (self.floor_color[None] + floor_irrad_vec * 1.2) * tm.exp(-A)
            else: # enter the bounding box and finally hit the background
                contrib = I + self.background_color[None] * tm.exp(-A)

        else: # directly hit the background without entering the bounding box
            contrib = self.background_color[None]
        
        self.color_buffer[u, v] += contrib


    @ti.kernel
    def path_tracing(self):
        ti.loop_config(block_dim=256)
        for u, v in self.color_buffer:
            d = self.get_cast_dir(u, v)
            pos = self.camera_pos[None]

            contrib = ti.Vector([0.0, 0.0, 0.0])
            throughput = ti.Vector([1.0, 1.0, 1.0])
            c = ti.Vector([1.0, 1.0, 1.0])

            depth = 0
            hit_light = 0
            hit_background = 0

            # Tracing begin
            for _bounce in range(MAX_RAY_DEPTH):
                depth += 1
                closest, normal, c, hit_light = self.next_hit(pos, d)
                hit_pos = pos + closest * d
                if not hit_light and normal.norm() != 0 and closest < 1e8: # type: ignore
                    d = out_dir(normal)
                    pos = hit_pos + 1e-4 * d
                    throughput *= c # multiply color

                    if ti.static(use_directional_light):
                        dir_noise = ti.Vector([
                            ti.random() - 0.5,
                            ti.random() - 0.5,
                            ti.random() - 0.5
                        ]) * self.light_direction_noise[None]
                        light_dir = (self.light_direction[None] +
                                     dir_noise).normalized()
                        dot = light_dir.dot(normal)
                        if dot > 0:
                            _hit_light = 0
                            dist, _, _, _hit_light = self.next_hit(
                                pos, light_dir)
                            if dist > DIS_LIMIT:
                                # far enough to hit directional light
                                contrib += throughput * \
                                    self.light_color[None] * dot
                else:  # hit background or light voxel, terminate tracing
                    hit_background = 1
                    break

                # Russian roulette
                max_c = throughput.max()
                if ti.random() > max_c:
                    throughput = [0, 0, 0]
                    break
                else:
                    throughput /= max_c
            # Tracing end

            if hit_light:
                contrib += throughput * c
            else:
                if depth == 1 and hit_background:
                    # Direct hit to background
                    contrib = self.background_color[None]
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

    @ti.func
    def set_voxel(self, idx, mat, color: tm.vec3, ior=1.0):
        self.voxel_material[idx] = ti.cast(mat, ti.i8)
        self.voxel_color[idx] = self.to_vec3u(color)
        self.ior[idx] = ior

    @ti.func
    def set_voxel_data(self, idx, atten: ti.f32, scatter_strength: ti.f32, 
                       anisotropy_factor: ti.f32, opaque: ti.i8):
        self.atten[idx] = atten
        self.scatter_strength[idx] = scatter_strength
        self.anisotropy_factor[idx] = anisotropy_factor
        self.opaque[idx] = ti.cast(opaque, ti.i8)

    @staticmethod
    @ti.func
    def round_idx(idx_: vector(3, ti.f32)) -> vector(3, ti.i32):
        idx = ti.cast(idx_, ti.f32)
        return ti.Vector(
            [ti.round(idx[0]), # type: ignore
             ti.round(idx[1]), # type: ignore
             ti.round(idx[2])]).cast(ti.i32) # type: ignore