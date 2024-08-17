import numpy as np
import taichi as ti

from common.math_utils import np_normalize, np_rotate_matrix

INITIAL_POS = (0.0, 0.0, 3.0)

class TranslateCamera:
    def __init__(self, window, up):
        self._window = window
        self._camera_pos = np.array(INITIAL_POS)
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
        self._camera_pos = np.array(INITIAL_POS)
        self._lookat_pos = np.array((0.0, 0.0, 0.0))
        self._up = np_normalize(np.array(up))
        self._world_up = np.array((0.0, 1.0, 0.0))
        self._pitch_limit = np.radians(2)

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
        if axis is self._world_up:
            # Rotate around the lookat point on left-right axis
            rot_mat = self._rotation_matrix(axis, angle)
        else:
            tgtdir = self.target_dir
            cos_angle = np.dot(tgtdir, self._world_up)
            cos_limit = np.cos(self._pitch_limit)
            if (angle > 0 and cos_angle <= -cos_limit) or (angle < 0 and cos_angle >= cos_limit):
            # if (angle > 0 and self._camera_pos[1] >= 3) or \
            #    (angle < 0 and self._camera_pos[1] <= -3):
                return
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

def normalize_camera_pos(cam_pos: tuple | np.ndarray, distance=2.5) -> np.ndarray:
    cam_pos = np.array(cam_pos)
    cam_pos = cam_pos / np.linalg.norm(cam_pos) * distance
    return cam_pos
