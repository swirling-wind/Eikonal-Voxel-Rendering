import os
from datetime import datetime
# import __main__
import numpy as np
import taichi as ti
import taichi.math as tm
import torch
from scipy.ndimage import gaussian_filter

from render.ray_march import Renderer
from common.math_utils import np_normalize, np_rotate_matrix


VOXEL_DX = 1 / 64
SCREEN_RES = (1280, 720)
UP_DIR = (0, 1, 0)
HELP_MSG_TRANSLATE = '''
====================================================
* Drag with your left mouse button to rotate camera
* Press W/A/S/D/Q/E to move camera
====================================================
'''

MAT_LAMBERTIAN = 1
MAT_LIGHT = 2

class TranslateCamera:
    def __init__(self, window, up):
        self._window = window
        self._camera_pos = np.array((0.0, 0.5, 4.0))
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

    def _update_by_mouse(self):
        win = self._window
        if not self.mouse_exclusive_owner or not win.is_pressed(ti.ui.LMB):
            self._last_mouse_pos = None
            return False
        mouse_pos = np.array(win.get_cursor_pos())
        if self._last_mouse_pos is None:
            self._last_mouse_pos = mouse_pos
            return False
        # Makes camera rotation feels right
        dx, dy = self._last_mouse_pos - mouse_pos
        self._last_mouse_pos = mouse_pos

        out_dir = self._lookat_pos - self._camera_pos
        leftdir = self._compute_left_dir(np_normalize(out_dir))

        scale = 3
        rotx = np_rotate_matrix(self._up, dx * scale)
        roty = np_rotate_matrix(leftdir, dy * scale)

        out_dir_homo = np.array(list(out_dir) + [0.0])
        new_out_dir = np.matmul(np.matmul(roty, rotx), out_dir_homo)[:3]
        self._lookat_pos = self._camera_pos + new_out_dir

        return True

    def _update_by_wasd(self):
        win = self._window
        tgtdir = self.target_dir
        leftdir = self._compute_left_dir(tgtdir)
        lut = [
            ('w', tgtdir),
            ('a', leftdir),
            ('s', -tgtdir),
            ('d', -leftdir),
            ('e', [0, -1, 0]),
            ('q', [0, 1, 0]),
        ]
        dir = np.array([0.0, 0.0, 0.0])
        pressed = False
        for key, d in lut:
            if win.is_pressed(key):
                pressed = True
                dir += np.array(d)
        if not pressed:
            return False
        dir *= 0.05
        self._lookat_pos += dir
        self._camera_pos += dir
        return True

    @property
    def position(self):
        return self._camera_pos

    @property
    def look_at(self):
        return self._lookat_pos

    @property
    def target_dir(self):
        return np_normalize(self.look_at - self.position)

    def _compute_left_dir(self, tgtdir):
        cos = np.dot(self._up, tgtdir)
        if abs(cos) > 0.999:
            return np.array([-1.0, 0.0, 0.0])
        return np.cross(self._up, tgtdir)

class RotateCamera:
    def __init__(self, window, up):
        self._window = window
        self._camera_pos = np.array((0.0, 0.0, 4.0))
        self._lookat_pos = np.array((0.0, 0.0, 0.0))
        self._up = np_normalize(np.array(up))
        # self._last_mouse_pos = None
        self._world_up = np.array((0.0, 1.0, 0.0))

    def update_camera(self):
        res = self._update_by_wasd()
        return res

    def _update_by_wasd(self):
        win = self._window
        tgtdir = self.target_dir
        leftdir = self._compute_left_dir(tgtdir)
        updir = self._up
        lut = [
            ('w', tgtdir, 0.0),
            ('s', -tgtdir, 0.0),
            ('a', self._world_up, -0.05),
            ('d', self._world_up, 0.05),
            ('e', leftdir, -0.05),
            ('q', leftdir, 0.05),
        ]
        pressed = False
        for key, d, angle in lut:
            if win.is_pressed(key):
                pressed = True
                if key in ['w', 's']:
                    self._camera_pos += d * 0.05 # Translate
                else:
                    self._rotate_camera(d, angle) # Rotate
        return pressed

    def _rotate_camera(self, axis, angle):
        rot_mat = self._rotation_matrix(axis, angle)
        cam_pos = self._camera_pos - self._lookat_pos
        cam_pos = np.dot(rot_mat, cam_pos)

        self._camera_pos = cam_pos + self._lookat_pos
        self._up = np.dot(rot_mat, self._up)

    def _rotation_matrix(self, axis, angle):
        cos_val = np.cos(angle)
        sin_val = np.sin(angle)
        x, y, z = axis
        rot_mat = np.array([
            [cos_val + x**2 * (1 - cos_val), x*y * (1 - cos_val) - z*sin_val, x*z * (1 - cos_val) + y*sin_val],
            [y*x * (1 - cos_val) + z*sin_val, cos_val + y**2 * (1 - cos_val), y*z * (1 - cos_val) - x*sin_val],
            [z*x * (1 - cos_val) - y*sin_val, z*y * (1 - cos_val) + x*sin_val, cos_val + z**2 * (1 - cos_val)]
        ])
        return rot_mat

    @property
    def position(self):
        return self._camera_pos

    @property
    def look_at(self):
        return self._lookat_pos

    @property
    def target_dir(self):
        return np_normalize(self.look_at - self.position)

    def _compute_left_dir(self, tgtdir):
        cos = np.dot(self._up, tgtdir)
        if abs(cos) > 0.999:
            return np.array([-1.0, 0.0, 0.0])
        return np.cross(self._up, tgtdir)


class Scene:
    def __init__(self, voxel_edges=0.06, exposure=3.0):        
        self.renderer = Renderer(dx=VOXEL_DX, image_res=SCREEN_RES,
                                 up=UP_DIR, voxel_edges=voxel_edges,
                                 exposure=exposure)
        
    @staticmethod
    @ti.func
    def round_idx(idx_: tm.vec3) -> tm.vec3:
        idx = ti.cast(idx_, ti.f32)
        return ti.Vector(
            [ti.round(idx[0]), # type: ignore
             ti.round(idx[1]), # type: ignore
             ti.round(idx[2])]).cast(ti.i32) # type: ignore

    @ti.func
    def set_voxel(self, idx: tm.vec3, origin: tm.vec3,
                  mat: ti.i8, color: tm.vec3, ior=1.0):
        self.renderer.set_voxel(self.round_idx(idx + origin), mat, color, ior)

    @ti.func
    def set_voxel_data(self, idx: tm.vec3, origin: tm.vec3,
                       atten: ti.f32, scatter_strength: ti.f32, 
                       anisotropy_factor: ti.f32, opaque: ti.u8):
        self.renderer.set_voxel_data(self.round_idx(idx + origin), atten=atten, scatter_strength=scatter_strength,
                                     anisotropy_factor=anisotropy_factor, opaque=opaque)

    def set_floor(self, height, color):
        self.renderer.floor_height[None] = height
        self.renderer.floor_color[None] = color

    def set_directional_light(self, direction: tuple, direction_noise: float, color: tuple):
        self.renderer.set_directional_light(direction, direction_noise, color)

    def set_background_color(self, color):
        self.renderer.background_color[None] = color

    def offline_render(self, camera_pos_list: list[np.ndarray]) -> list[np.ndarray]:
        torch.cuda.empty_cache()
        # self.camera = Camera(None, up=UP_DIR)
        render_res_list = []
        for cur_camera_pos in camera_pos_list:
            self.renderer.set_camera_pos(cur_camera_pos[0], cur_camera_pos[1], cur_camera_pos[2])
            self.renderer.set_look_at(0, 0, 0)
            self.renderer.reset_framebuffer()
            self.renderer.recompute_bbox()

            for _cur_spp in range(5):
                self.renderer.ray_marching()
                self.renderer.current_spp += 1
            
            raw_img = self.renderer.fetch_image().to_numpy()
            transposed_img = np.transpose(raw_img, (1, 0, 2))[::-1, :, :]
            render_res_list.append(transposed_img)
        return render_res_list

    def rt_render(self, translate_mode: bool = True):
        torch.cuda.empty_cache()
        
        if translate_mode:
            print(HELP_MSG_TRANSLATE)
            self.window = ti.ui.Window("Ray marching (Translate mode)",
                                   SCREEN_RES,
                                   vsync=True)
        
            self.camera = TranslateCamera(self.window, up=UP_DIR)
        
        else:
            self.window = ti.ui.Window("Ray marching (Rotate mode)",
                                   SCREEN_RES,
                                   vsync=True)
        
            self.camera = RotateCamera(self.window, up=UP_DIR)

        self.renderer.set_camera_pos(*self.camera.position)
        self.renderer.set_look_at(*self.camera.look_at)
        self.renderer.reset_framebuffer()
        self.renderer.recompute_bbox()
        canvas = self.window.get_canvas()
        # print(self.camera.position, self.camera.look_at)
        # rendered = False

        while self.window.running:
            should_reset_framebuffer = False            
            if self.camera.update_camera():
                self.renderer.set_camera_pos(*self.camera.position)
                look_at = self.camera.look_at
                self.renderer.set_look_at(*look_at)
                should_reset_framebuffer = True
            if should_reset_framebuffer:
                self.renderer.reset_framebuffer()
           
            # if not rendered:
            self.renderer.ray_marching()
            self.renderer.current_spp += 1

            img = self.renderer.fetch_image()
            # if self.window.is_pressed('p'):   # Save screenshot
            #     self.save_screenshot(img)
            canvas.set_image(img)
            self.window.show()

        self.window.destroy()

    def save_screenshot(self, image: ti.Field):
        timestamp = datetime.today().strftime('%Y-%m-%d-%H%M%S')
        dirpath = os.getcwd()
        fname = os.path.join(dirpath, 'snapshots', f"{timestamp}.jpg")
        ti.tools.image.imwrite(image, fname) # type: ignore
        print(f"Screenshot has been saved to {fname}")
    
    #### IOR ####
    @property
    def ior(self) -> np.ndarray:
        return self.renderer.ior.to_numpy()

    @ior.setter
    def ior(self, ior_grid: ti.types.ndarray()):
        self.renderer.ior.from_numpy(ior_grid)

    #### Gradient ####
    @property
    def gradient(self) -> np.ndarray:
        return self.renderer.grad.to_numpy()
        
    @gradient.setter
    def gradient(self, grad_field: np.ndarray):
        self.renderer.grad.from_numpy(grad_field)    
    
    #### Irradiance ####
    @property
    def irradiance(self) -> np.ndarray:
        return self.renderer.irrad.to_numpy()
    
    @irradiance.setter
    def irradiance(self, irrad_field: ti.types.ndarray()):
        self.renderer.irrad.from_numpy(irrad_field)

    #### Loc Dir ####
    @property
    def local_diretion(self) -> np.ndarray:
        return self.renderer.loc_dir.to_numpy()
    
    @local_diretion.setter
    def local_diretion(self, loc_dir_field: ti.types.ndarray()):
        self.renderer.loc_dir.from_numpy(loc_dir_field)

    #### Attenuation ####
    @property
    def attenuation(self) -> np.ndarray:
        return self.renderer.atten.to_numpy()
    
    @attenuation.setter
    def attenuation(self, atten_field: ti.types.ndarray()):
        self.renderer.atten.from_numpy(atten_field)

    #### Scatter strength ####
    @property
    def scatter_strength(self) -> np.ndarray:
        return self.renderer.scatter_strength.to_numpy()
    
    @scatter_strength.setter
    def scatter_strength(self, scatter_field: ti.types.ndarray()):
        self.renderer.scatter_strength.from_numpy(scatter_field)

    #### Anisotropy factor ####
    @property
    def anisotropy_factor(self) -> np.ndarray:
        return self.renderer.anisotropy_factor.to_numpy()
    
    @anisotropy_factor.setter
    def anisotropy_factor(self, aniso_field: ti.types.ndarray()):
        self.renderer.anisotropy_factor.from_numpy(aniso_field)
    
    #### Opaque ####
    @property
    def opaque(self) -> np.ndarray:
        return self.renderer.opaque.to_numpy()
    
    @opaque.setter
    def opaque(self, opaque_field: ti.types.ndarray()):
        self.renderer.opaque.from_numpy(opaque_field)


    #### Truncate outside the surface of objects ####
    def truncate_outside_surface(self, gradient_threshold: float = 0.05):
        outside_mask = np.linalg.norm(self.gradient, axis=-1) < gradient_threshold
        temp_ior = self.ior.copy()
        temp_ior[outside_mask] = 1.0
        self.ior = temp_ior

        temp_atten = self.attenuation.copy()
        temp_atten[outside_mask] = 0.0
        self.attenuation = temp_atten

        temp_scatter = self.scatter_strength.copy()
        temp_scatter[outside_mask] = 0.0
        self.scatter_strength = temp_scatter

        temp_grad = self.gradient.copy()
        temp_grad[outside_mask, :] = 0
        self.gradient = temp_grad

    def apply_filter(self, config: dict):
        sigma = config["Gaus Sigma"]
        radius = config["Gaus Radius"]
        
        self.ior = gaussian_filter(self.ior, sigma=sigma, radius=radius)
        self.attenuation = gaussian_filter(self.attenuation, sigma=sigma, radius=radius)
        self.scatter_strength = gaussian_filter(self.scatter_strength, sigma=sigma, radius=radius)



