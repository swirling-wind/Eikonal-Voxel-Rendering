import os
from datetime import datetime
# import __main__
import numpy as np
import taichi as ti
import taichi.math as tm
import torch
from scipy.ndimage import gaussian_filter

from render.ray_march import Renderer
from .camera import RotateCamera, TranslateCamera

VOXEL_DX = 1 / 64
# SCREEN_RES = (1600, 900)
UP_DIR = (0, 1, 0)
HELP_MSG_TRANSLATE = '''
====================================================
* Drag with your left mouse button to rotate camera
* Press W/A/S/D/Q/E to move camera
====================================================
'''

MAT_LAMBERTIAN = 1
MAT_LIGHT = 2

class Scene:
    def __init__(self, hdr_res: tuple[int, int], hdr_name: str, screen_res: tuple[int, int], exposure=3.0):  
        self.screen_res = screen_res      
        self.renderer = Renderer(hdr_res, hdr_name, VOXEL_DX, self.screen_res, UP_DIR, exposure)

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
                       atten: ti.f32, scatter_strength: ti.f32):
        self.renderer.set_voxel_data(self.round_idx(idx + origin), atten=atten, scatter_strength=scatter_strength)

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

            for _cur_spp in range(10):
                self.renderer.ray_marching()
                self.renderer.current_spp += 1
            
            raw_img = self.renderer.fetch_image().to_numpy()
            transposed_img = np.transpose(raw_img, (1, 0, 2))[::-1, :, :]
            render_res_list.append(transposed_img)
        return render_res_list

    def rt_render(self, free_mode: bool = True):
        torch.cuda.empty_cache()
        
        if free_mode:
            self.window = ti.ui.Window("Ray marching (Translate mode)",
                                   self.screen_res,
                                   vsync=True)
        
            self.camera = TranslateCamera(self.window, up=UP_DIR)
        
        else:
            self.window = ti.ui.Window("Ray marching (Rotate mode)",
                                   self.screen_res,
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

    def apply_filter(self, proc_config: dict):
        sigma = proc_config["Gauss Sigma"]
        radius = proc_config["Gauss Radius"]
        
        self.ior = gaussian_filter(self.ior, sigma=sigma, radius=radius)
        self.attenuation = gaussian_filter(self.attenuation, sigma=sigma, radius=radius)
        self.scatter_strength = gaussian_filter(self.scatter_strength, sigma=sigma, radius=radius)



