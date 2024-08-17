from setup.scene import Scene
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def normalize_camera_pos(cam_pos: tuple | np.ndarray, distance=2.5) -> np.ndarray:
    cam_pos = np.array(cam_pos)
    cam_pos = cam_pos / np.linalg.norm(cam_pos) * distance
    return cam_pos

def save_offline_render(scene: Scene, pos_list: list[tuple], scene_config: dict, filename: str, to_plot=True):
    camera_pos_list = [normalize_camera_pos(pos) for pos in pos_list]
    img_list = scene.offline_render(camera_pos_list)
    for idx, img in enumerate(img_list):
        img = np.clip(img, 0, 1)
        pil_image = Image.fromarray((img * 255).astype(np.uint8))
        pil_image.save("./images/" + scene_config["Name"] + "/" + filename + f"_{idx}.png")

        if to_plot:
            plt.imshow(img)
            plt.title(f"Camera Pos: [{camera_pos_list[idx][0]:.2f}, {camera_pos_list[idx][1]:.2f}, {camera_pos_list[idx][2]:.2f}]")
            plt.axis('off')
            plt.show()