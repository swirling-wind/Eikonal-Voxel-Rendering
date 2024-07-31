import os
from datetime import datetime
# import __main__
import numpy as np
import taichi as ti
import taichi.math as tm
from taichi.types import vector, matrix
from render.renderer import Renderer
from common.math_utils import np_normalize, np_rotate_matrix

VOXEL_DX = 1 / 64
SCREEN_RES = (1280, 720)
UP_DIR = (0, 1, 0)
HELP_MSG = '''
====================================================
* Drag with your left mouse button to rotate camera
* Press W/A/S/D/Q/E to move camera
====================================================
'''

MAT_LAMBERTIAN = 1
MAT_LIGHT = 2

class Camera:
    def __init__(self, window, up):
        self._window = window
        self._camera_pos = np.array((0.0, 0.0, 2.0))
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


class Scene:
    def __init__(self, voxel_edges=0.06, exposure=3):        
        self.renderer = Renderer(dx=VOXEL_DX,
                                 image_res=SCREEN_RES,
                                 up=UP_DIR,
                                 voxel_edges=voxel_edges,
                                 exposure=exposure)
        
    @staticmethod
    @ti.func
    def round_idx(idx_: vector(3, ti.f32)) -> vector(3, ti.i32):
        idx = ti.cast(idx_, ti.f32)
        return ti.Vector(
            [ti.round(idx[0]), # type: ignore
             ti.round(idx[1]), # type: ignore
             ti.round(idx[2])]).cast(ti.i32) # type: ignore

    @ti.func
    def set_voxel(self, idx: tm.vec3, mat: ti.i8, color: tm.vec3, ior=1.0):
        self.renderer.set_voxel(self.round_idx(idx), mat, color, ior)

    @ti.func
    def get_voxel(self, idx):
        mat, color = self.renderer.get_voxel(self.round_idx(idx))
        return mat, color

    def set_floor(self, height, color):
        self.renderer.floor_height[None] = height
        self.renderer.floor_color[None] = color

    def set_directional_light(self, direction: tuple, direction_noise: float, color: tuple):
        self.renderer.set_directional_light(direction, direction_noise, color)

    def set_background_color(self, color):
        self.renderer.background_color[None] = color

    def display(self, ray_marching=False):
        print(HELP_MSG)
        self.window = ti.ui.Window("Path Tracing",
                                   SCREEN_RES,
                                   vsync=True)
        self.camera = Camera(self.window, up=UP_DIR)
        self.renderer.set_camera_pos(*self.camera.position)
        self.renderer.set_look_at(*self.camera.look_at)
        self.renderer.reset_framebuffer()
        self.renderer.recompute_bbox()
        canvas = self.window.get_canvas()
        # print(self.camera.position, self.camera.look_at)

        while self.window.running:
            should_reset_framebuffer = False
            if self.camera.update_camera():
                self.renderer.set_camera_pos(*self.camera.position)
                look_at = self.camera.look_at
                self.renderer.set_look_at(*look_at)
                should_reset_framebuffer = True

            if should_reset_framebuffer:
                self.renderer.reset_framebuffer()

            #  for _ in range(num_samples) to adjust fps with samples per pixel (spp) 
            if ray_marching:
                self.renderer.ray_marching()
            else:
                self.renderer.path_tracing()
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
        ior_grid = self.renderer.ior
        if isinstance(ior_grid, np.ndarray):
            return ior_grid
        else:
            return ior_grid.to_numpy()

    @ior.setter
    def ior(self, ior_grid: ti.types.ndarray()):
        self.renderer.ior = ior_grid

    #### Gradient ####
    @property
    def gradient(self) -> np.ndarray:
        grad_grid = self.renderer.grad
        return grad_grid if isinstance(grad_grid, np.ndarray) else grad_grid.to_numpy()

    @gradient.setter
    def gradient(self, grad_field: ti.types.ndarray()):
        self.renderer.grad = grad_field

    #### Attenuation ####
    @property
    def attenuation(self) -> np.ndarray:
        atten_grid = self.renderer.atten
        return atten_grid if isinstance(atten_grid, np.ndarray) else atten_grid.to_numpy()

    @attenuation.setter
    def attenuation(self, atten_field: ti.types.ndarray()):
        self.renderer.atten = atten_field

    #### Loc Dir ####
    @property
    def local_diretion(self) -> np.ndarray:
        local_direction = self.renderer.loc_dir
        return local_direction if isinstance(local_direction, np.ndarray) else local_direction.to_numpy()

    @local_diretion.setter
    def local_diretion(self, loc_dir_field: ti.types.ndarray()):
        self.renderer.loc_dir = loc_dir_field

    #### Irradiance ####
    @property
    def irradiance(self) -> np.ndarray:
        irradiance_field = self.renderer.irrad
        return irradiance_field if isinstance(irradiance_field, np.ndarray) else irradiance_field.to_numpy()
    
    @irradiance.setter
    def irradiance(self, irrad_field: ti.types.ndarray()):
        self.renderer.irrad = irrad_field