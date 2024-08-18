from setup.scene import Scene
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

def normalize_camera_pos(cam_pos: tuple | np.ndarray, distance=2.5) -> np.ndarray:
    cam_pos = np.array(cam_pos)
    cam_pos = cam_pos / np.linalg.norm(cam_pos) * distance
    return cam_pos

def save_offline_render(scene: Scene, scene_config: dict, filename: str, to_plot=True):
    camera_pos_list = [normalize_camera_pos(pos) for pos in scene_config["Cam Pos"]]
    img_list = scene.offline_render(camera_pos_list)
    for idx, img in enumerate(img_list):
        img = np.clip(img, 0, 1)

        if scene_config["Save Fig"]:
            
            pil_image = Image.fromarray((img * 255).astype(np.uint8))
            image_path = os.path.join(os.getcwd(), "images", scene_config["HDR Name"], scene_config["Name"], filename)
            pil_image.save(image_path + f"_{idx}.png")

        if to_plot:
            plt.imshow(img)
            plt.title(f"Camera Pos: [{camera_pos_list[idx][0]:.2f}, {camera_pos_list[idx][1]:.2f}, {camera_pos_list[idx][2]:.2f}]")
            plt.axis('off')
            plt.show()