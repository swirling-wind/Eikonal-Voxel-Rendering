from setup.scene import Scene
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import CircleCollection

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


def save_colorbar(save_path, vmin=0, vmax=105):
    fig, ax = plt.subplots(figsize=(1, 13))
    fig.subplots_adjust(bottom=0.5)

    cmap = plt.get_cmap('hot') # type: ignore
    norm = plt.Normalize(vmin=vmin, vmax=vmax) # type: ignore
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), # type: ignore
                      cax=ax, orientation='vertical')
    cb.ax.tick_params(labelsize=28)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)


def save_color_wheel(save_path):
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))

    n_points = 100000
    theta = np.random.uniform(0, 2*np.pi, n_points)
    radii = np.sqrt(np.random.uniform(0, 1, n_points)) / 1.4  # 使用平方根分布来确保均匀填充
    adjusted_theta = (-theta + np.pi/2 * 3) % (2*np.pi)

    colors = plt.cm.hsv(adjusted_theta / (2*np.pi))

    circles = CircleCollection(sizes=[40]*n_points, offsets=list(zip(theta, radii)),
                               transOffset=ax.transData, facecolors=colors, edgecolors='none')
    ax.add_collection(circles)

    ax.set_ylim(0, 1)

    ax.set_yticks([])
    ax.set_xticks([])

    ax.plot([0, 0], [0, 0.99], color='black', linewidth=6)  # x轴
    ax.plot([np.pi/2, np.pi/2], [0, 0.99], color='black', linewidth=6)  # y轴

    font_size = 64
    ax.text(0, 1, '+x', ha='center', va='bottom', fontsize=font_size)
    ax.text(np.pi/2, 1, '+y', ha='left', va='center', fontsize=font_size)

    ax.set_frame_on(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    # 设置背景透明
    ax.set_facecolor('none')
    fig.patch.set_alpha(0.0)

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
