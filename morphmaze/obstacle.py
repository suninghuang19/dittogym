import os
import cv2
import gym
import json
import numpy as np
import taichi as ti
from morphmaze.morphmaze import morphmaze

OBS_ACT_CENTER_Y = 0.28

@ti.data_oriented
class OBSTACLE(morphmaze):
    def __init__(self, cfg_path=None, action_dim=2*8**2):
        super(OBSTACLE, self).__init__(cfg_path=cfg_path, action_dim=action_dim)
        print("*******************Morphological Maze OBSTACLE-v0*******************")
        # initial robot task-OBSTACLE
        self.obs_auto_reset = False
        self.add_rectangular(0.0, 0.0, 0.16, 0.16)
        for i in range(len(self.x_list)):
            self.x_save[i] = self.x_list[i]
            self.material_save[i] = self.material_list[i]
            self.mass_save[i] = self.mass_list[i]
        self.reset_()
        self.center_point = [
            np.mean(self.x_save.to_numpy()[:self.robot_particles_num, 0]),
            np.mean(self.x_save.to_numpy()[:self.robot_particles_num, 1]),
        ]
        self.anchor = ti.Vector.field(2, dtype=float, shape=())
        self.anchor[None] = [np.mean(self.x.to_numpy()[:self.robot_particles_num, 0]) - 0.4, 0.0]
        self.set_bg_env()
        self.set_obs_field()
        self.update_obs(fix_y=OBS_ACT_CENTER_Y)
        self.init_location = np.mean(self.x.to_numpy()[:self.robot_particles_num], axis=0)
        self.prev_location = self.init_location
        self.gui = None

    def reset(self):
        self.reset_()
        self.anchor[None] = [self.init_location[0] - 0.4, 0.0]
        return self.state

    def step(self, action):
        self.update_grid_actuation(action, fix_y=OBS_ACT_CENTER_Y)
        for i in range(self.repeat_times):
            self.update_particle_actuation()
            self.p2g()
            self.grid_operation()
            self.g2p()
            if self.visualize and i == 0:
                self.render(self.gui, log=True)
        # state (relative x, y)
        x_numpy = self.x.to_numpy()
        if not np.isnan(x_numpy[:self.robot_particles_num, 0]).any():
            self.anchor[None] = [np.mean(x_numpy[:self.robot_particles_num, 0]) - 0.4, 0.0]
        else:
            self.anchor[None] = [self.prev_location[0] - 0.4, 0.0]
        self.center_point = [np.mean(x_numpy[:self.robot_particles_num, 0]), np.mean(x_numpy[:self.robot_particles_num, 1])]
        self.set_bg_env()
        self.set_obs_field()
        self.update_obs(fix_y=OBS_ACT_CENTER_Y)
        # if not os.path.exists("./observation"):
        #     os.makedirs("./observation")
        # cv2.imwrite("./observation/state.png", self.state[0])
        # cv2.imwrite("./observation/vx.png", self.state[1])
        # cv2.imwrite("./observation/vy.png", self.state[2])
        if not np.isnan(self.center_point).any():
            self.prev_location = self.center_point
        else:
            self.center_point = self.prev_location
        terminated = False
        # # location
        location_reward = 0
        x_mean = self.center_point[0]
        location_reward = np.clip(np.sign(x_mean - self.init_location[0]) * (2 * (x_mean - self.init_location[0]))**2 + 5 * (x_mean - self.init_location[0]), a_min=-20, a_max=20)
        # velocity
        velocity_reward = 0
        vx_mean = np.mean(self.v.to_numpy()[:, 0])
        velocity_reward = np.clip(np.sign(vx_mean) * (2 * vx_mean)**2 + 5 * (vx_mean), a_min=-20, a_max=20)
        # action
        action_reward = -np.sum(np.linalg.norm(self.action, axis=(1, 2))) 
        # split
        split = np.clip(np.linalg.norm([np.std(self.x.to_numpy()[:, 0]), np.std(self.x.to_numpy()[:, 1])]), a_min=0, a_max=0.2)
        if split > 0.1:
            split_reward = -((20 * split) ** 2)
            terminated = True
        else:
            split_reward = 0
        reward = (
            + self.reward_params[0] * location_reward
            + self.reward_params[1] * velocity_reward
            + self.reward_params[2] * split_reward
            + self.reward_params[3] * action_reward
            + self.reward_params[4]
        )
        if x_mean < -0.2:
            terminated = True
        # info
        info = {}
        if np.isnan(self.state).any():
            raise ValueError("state has nan") 

        return (self.state, reward, terminated, False, info)

    def render(self, gui, log=False, record_id=None):
        self.gui = gui
        if not log:
            self.visualize = False
            self.frames_num = 0
            return None
        else:
            if record_id is not None:
                self.record_id = record_id
            self.visualize = True
            start_point = int(self.anchor[None][0] * 512)
            if start_point < 0:
                while start_point < 0:
                    start_point += 512
            elif start_point > 512:
                while start_point > 512:
                    start_point -= 512
            image = cv2.imread(os.path.join(self.current_directory, "./bg/bg.png"), cv2.IMREAD_COLOR).astype(np.uint8).transpose(1, 0, 2)[:, ::-1, :]
            image = np.concatenate([image[start_point:512, :, :], image[:start_point, :, :]], axis=0)
            gui.set_image(image)
            self.gui.line(begin=(0, 20 / 128 - 0.015), end=(1, 20 / 128 - 0.015), radius=7, color=0x647D8E)
            for j in range(6):
                i = 0
                while (26 - j) / 128 - 0.001 * i >= 20 / 128:
                    self.gui.line(begin=(max(0, (30 + 3 * j) / 128 - self.anchor[None][0]), (26 - j) / 128 - 0.001 * i), end=(min(1, (33 + 3 * j) / 128 - self.anchor[None][0]), (26 - j) / 128 - 0.001 * i), radius=3, color=0x394C31)
                    i += 1
            self.gui.circles(
                self.x.to_numpy() - np.array([self.anchor[None][0], 0]),
                radius=1.5,
                color=0xFF5722,
            )
            if not os.path.exists(os.path.join(self.current_directory, "../results")):
                os.makedirs(os.path.join(self.current_directory, "../results"))
            if not os.path.exists(os.path.join(self.current_directory, "../results/" + self.save_file_name + "/record_" + str(self.record_id))):
                os.makedirs(os.path.join(self.current_directory, "../results/" + self.save_file_name + "/record_" + str(self.record_id)))
            self.gui.show(
                os.path.join(self.current_directory, "../results/"
                + self.save_file_name
                + "/record_"
                + str(self.record_id)
                + "/frame_%04d.png" % self.frames_num)
            )
            self.frames_num += 1

    def add_rectangular(self, x, y, w, h, is_object=False):
        '''
        generate square robot with specifc density
        changable parameter, should align with number of particles
        (x, y) rectangular left bottom point
        (w, h) rectangular width and height
        '''
        w_count = int(w / self.dx) * 6
        h_count = int(h / self.dx) * 6
        real_dx = w / w_count
        real_dy = h / h_count
        for i in range(h_count):
            for j in range(w_count):
                self.x_list.append(
                    [
                        x + (j + 0.5) * real_dx + self.offset_x,
                        y + (i + 0.5) * real_dy + self.offset_y,
                    ]
                )
                self.material_list.append(1 if is_object else 0)
                self.mass_list.append(self.mass[1] if is_object else self.mass[0])

    @ti.kernel
    def set_bg_env(self):
        for i, j in self.shape_field:
            self.shape_field[i, j] = 0.0
            self.vx_field[i, j] = 0.0
            self.vy_field[i, j] = 0.0
            if ((self.n_grid * self.obs_res_resize - i) <= 26 * self.obs_res_resize and (30 / 128 <= (j / (self.obs_res_resize * self.n_grid) + self.anchor[None][0]) <= 33 / 128) or
                (self.n_grid * self.obs_res_resize - i) <= 25 * self.obs_res_resize and (33 / 128 <= (j / (self.obs_res_resize * self.n_grid) + self.anchor[None][0]) <= 36 / 128) or
                (self.n_grid * self.obs_res_resize - i) <= 24 * self.obs_res_resize and (36 / 128 <= (j / (self.obs_res_resize * self.n_grid) + self.anchor[None][0]) <= 39 / 128) or
                (self.n_grid * self.obs_res_resize - i) <= 23 * self.obs_res_resize and (39 / 128 <= (j / (self.obs_res_resize * self.n_grid) + self.anchor[None][0]) <= 42 / 128) or
                (self.n_grid * self.obs_res_resize - i) <= 22 * self.obs_res_resize and (42 / 128 <= (j / (self.obs_res_resize * self.n_grid) + self.anchor[None][0]) <= 45 / 128) or
                (self.n_grid * self.obs_res_resize - i) <= 21 * self.obs_res_resize and (45 / 128 <= (j / (self.obs_res_resize * self.n_grid) + self.anchor[None][0]) <= 48 / 128)):                
                self.shape_field[i, j] = 1.0        

    @ti.kernel
    def grid_operation(self):
        # specific grid operation for OBSTACLE
        for i, j in self.grid_m:
            inv_m = 1 / (self.grid_m[i, j] + 1e-10)
            self.grid_v[i, j] = inv_m * self.grid_v[i, j]
            self.grid_v[i, j][0] += self.dt * self.gravity[None][0]
            self.grid_v[i, j][1] += self.dt * self.gravity[None][1]
            # self.grid_v[i, j] = 0.999 * self.grid_v[i, j]
            # # infinite horizon
            if i > 30 - int(self.anchor[None][0] * self.n_grid) - self.bound and i < 33 - int(self.anchor[None][0] * self.n_grid) and j <= 26  and self.grid_v[i, j][0] > 0:
                self.grid_v[i, j][0] = 0
            if i > 33 - int(self.anchor[None][0] * self.n_grid) - self.bound and i < 36 - int(self.anchor[None][0] * self.n_grid) and j <= 25  and self.grid_v[i, j][0] > 0:
                self.grid_v[i, j][0] = 0
            if i > 36 - int(self.anchor[None][0] * self.n_grid) - self.bound and i < 39 - int(self.anchor[None][0] * self.n_grid) and j <= 24  and self.grid_v[i, j][0] > 0:
                self.grid_v[i, j][0] = 0
            if i > 39 - int(self.anchor[None][0] * self.n_grid) - self.bound and i < 42 - int(self.anchor[None][0] * self.n_grid) and j <= 23  and self.grid_v[i, j][0] > 0:
                self.grid_v[i, j][0] = 0
            if i > 42 - int(self.anchor[None][0] * self.n_grid) - self.bound and i < 45 - int(self.anchor[None][0] * self.n_grid) and j <= 22  and self.grid_v[i, j][0] > 0:
                self.grid_v[i, j][0] = 0
            if i > 45 - int(self.anchor[None][0] * self.n_grid) - self.bound and i < 48 - int(self.anchor[None][0] * self.n_grid) and j <= 21  and self.grid_v[i, j][0] > 0:
                self.grid_v[i, j][0] = 0
            
            if i < 33 - int(self.anchor[None][0] * self.n_grid) + self.bound and i > 30 - int(self.anchor[None][0] * self.n_grid) and j <= 26  and self.grid_v[i, j][0] < 0:
                self.grid_v[i, j][0] = 0
            if i < 36 - int(self.anchor[None][0] * self.n_grid) + self.bound and i > 33 - int(self.anchor[None][0] * self.n_grid) and j <= 25  and self.grid_v[i, j][0] < 0:
                self.grid_v[i, j][0] = 0
            if i < 39 - int(self.anchor[None][0] * self.n_grid) + self.bound and i > 36 - int(self.anchor[None][0] * self.n_grid) and j <= 24  and self.grid_v[i, j][0] < 0:
                self.grid_v[i, j][0] = 0
            if i < 42 - int(self.anchor[None][0] * self.n_grid) + self.bound and i > 39 - int(self.anchor[None][0] * self.n_grid) and j <= 23  and self.grid_v[i, j][0] < 0:
                self.grid_v[i, j][0] = 0
            if i < 45 - int(self.anchor[None][0] * self.n_grid) + self.bound and i > 42 - int(self.anchor[None][0] * self.n_grid) and j <= 22  and self.grid_v[i, j][0] < 0:
                self.grid_v[i, j][0] = 0
            if i < 48 - int(self.anchor[None][0] * self.n_grid) + self.bound and i > 45 - int(self.anchor[None][0] * self.n_grid) and j <= 21  and self.grid_v[i, j][0] < 0:
                self.grid_v[i, j][0] = 0
            
            if 30 - int(self.anchor[None][0] * self.n_grid) <= i <= 33 - int(self.anchor[None][0] * self.n_grid) and j <= 26 and self.grid_v[i, j][1] < 0:
                self.grid_v[i, j][1] = 0
            if 33 - int(self.anchor[None][0] * self.n_grid) <= i <= 36 - int(self.anchor[None][0] * self.n_grid) and j <= 25 and self.grid_v[i, j][1] < 0:
                self.grid_v[i, j][1] = 0
            if 36 - int(self.anchor[None][0] * self.n_grid) <= i <= 39 - int(self.anchor[None][0] * self.n_grid) and j <= 24 and self.grid_v[i, j][1] < 0:
                self.grid_v[i, j][1] = 0
            if 39 - int(self.anchor[None][0] * self.n_grid) <= i <= 42 - int(self.anchor[None][0] * self.n_grid) and j <= 23 and self.grid_v[i, j][1] < 0:
                self.grid_v[i, j][1] = 0
            if 42 - int(self.anchor[None][0] * self.n_grid) <= i <= 45 - int(self.anchor[None][0] * self.n_grid) and j <= 22 and self.grid_v[i, j][1] < 0:
                self.grid_v[i, j][1] = 0
            if 45 - int(self.anchor[None][0] * self.n_grid) <= i <= 48 - int(self.anchor[None][0] * self.n_grid) and j <= 21 and self.grid_v[i, j][1] < 0:
                self.grid_v[i, j][1] = 0

            # up
            if j < self.bound * 20 and self.grid_v[i, j][1] < 0:
                self.grid_v[i, j] = [0, 0]
                normal = ti.Vector([0.0, 1.0])
                lsq = (normal**2).sum()
                if lsq > 0.5:
                    if ti.static(self.coeff < 0):
                        self.grid_v[i, j] = [0, 0]
                    else:
                        lin = self.grid_v[i, j].dot(normal)
                        if lin < 0:
                            vit = self.grid_v[i, j] - lin * normal
                            lit = vit.norm() + 1e-10
                            if lit + self.coeff * lin <= 0:
                                self.grid_v[i, j] = [0, 0]
                            else:
                                self.grid_v[i, j] = vit * (1 + self.coeff * lin / lit)
            # down
            if j > self.n_grid - self.bound * 10 and self.grid_v[i, j][1] > 0:
                self.grid_v[i, j][1] = 0

if __name__ == "__main__":
    ti.init(arch=ti.gpu)
    gui = ti.GUI("Taichi MPM Morphological Maze", res=512, background_color=0x112F41, show_gui=False)
    env = OBSTACLE("./cfg/obstacle.json")
    env.reset()
    env.render(gui, log=True, record_id=0)
    while True:
        env.step(2 * np.random.rand(env.action_space.shape[0]) - 1)
