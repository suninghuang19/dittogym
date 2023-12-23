import os
import cv2
import gym
import json
import numpy as np
import taichi as ti
from morphmaze.morphmaze import morphmaze

OBS_ACT_CENTER_Y = 0.28
GAP = 36
GAP_1 = 40

@ti.data_oriented
class SLOT(morphmaze):
    def __init__(self, action_dim, cfg_path=None):
        super(SLOT, self).__init__(cfg_path=cfg_path, action_dim=action_dim)
        print("*******************Morphological Maze SLOT-v0*******************")
        # initial robot task-SLOT
        self.obs_auto_reset = False
        self.add_circle(0.0, 0.0, 0.16, is_object=False)
        self.add_rectangular(0.52, 0.162, 0.13, 0.03, is_object=True)
        self.target = ti.Vector.field(2, dtype=float, shape=())
        for i in range(len(self.x_list)):
            self.x_save[i] = self.x_list[i]
            self.material_save[i] = self.material_list[i]
            self.mass_save[i] = self.mass_list[i]
        self.reset_()
        self.center_point = [
            np.mean(self.x_save.to_numpy()[:self.robot_particles_num, 0]),
            np.mean(self.x_save.to_numpy()[:self.robot_particles_num, 1]),
        ]
        self.object_center_point = [
            np.mean(self.x_save.to_numpy()[self.robot_particles_num:, 0]),
            np.mean(self.x_save.to_numpy()[self.robot_particles_num:, 1]),
        ]
        self.anchor = ti.Vector.field(2, dtype=float, shape=())
        self.anchor[None] = [np.mean(self.x.to_numpy()[:self.robot_particles_num, 0]) - 0.25, 0.0]
        self.set_bg_env()
        self.set_obs_field()
        self.update_obs(fix_y=OBS_ACT_CENTER_Y)
        self.init_location = np.mean(self.x.to_numpy()[:self.robot_particles_num], axis=0)
        self.prev_location = self.init_location
        self.init_object_location = np.mean(self.x.to_numpy()[self.robot_particles_num:], axis=0)
        self.prev_object_location = self.init_object_location
        self.x_init = None
        self.y_init = None
        self.x_target = None
        self.gui = None     

    def reset(self):
        self.reset_()
        self.anchor[None] = [self.init_location[0] - 0.25, 0.0]
        x_numpy = self.x.to_numpy()[:self.robot_particles_num]
        object_numpy = self.x.to_numpy()[self.robot_particles_num:]
        self.target[None] = np.mean(object_numpy, axis=0)
        self.y_init = np.argpartition(np.linalg.norm(x_numpy - self.target.to_numpy(), axis=1), 2000)[:2000]
        self.x_init = np.argpartition(np.abs(x_numpy[self.y_init, 0]), 1999)[:1999]
        self.x_target = x_numpy[self.y_init[self.x_init]]
        self.mean_init = np.mean(np.linalg.norm(self.x_target - self.target.to_numpy(), axis=1))
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
        x_numpy = self.x.to_numpy()[:self.robot_particles_num]
        object_numpy = self.x.to_numpy()[self.robot_particles_num:]
        self.target[None] = np.mean(object_numpy, axis=0)   
        self.y_init = np.argpartition(np.linalg.norm(x_numpy - self.target.to_numpy(), axis=1), 2000)[:2000]
        self.x_init = np.argpartition(np.abs(x_numpy[self.y_init, 0]), 1999)[:1999]
        self.x_target = x_numpy[self.y_init[self.x_init]]
        if not np.isnan(x_numpy[:self.robot_particles_num, 0]).any():
            self.anchor[None] = [np.mean(x_numpy[:self.robot_particles_num, 0]) - 0.25, 0.0]
        else:
            self.anchor[None] = [self.prev_location[0] - 0.25, 0.0]
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
        x_mean = 2 * (-np.mean(np.linalg.norm(self.x_target - self.target.to_numpy(), axis=1)) + self.mean_init)
        location_reward = np.clip(np.sign(x_mean) * (4 * (x_mean))**2 + 10 * (x_mean), a_min=-20, a_max=50)
        # velocity
        vx_mean = np.mean(self.v.to_numpy()[:self.robot_particles_num, 0])
        vy_mean = np.mean(self.v.to_numpy()[:self.robot_particles_num, 1])
        velocity_reward = np.clip(np.sign(vx_mean) * (2 * vx_mean)**2 + 10 * (vx_mean), a_min=-20, a_max=20)\
            + np.clip(np.sign(vy_mean) * (2 * vy_mean)**2 + 10 * (vy_mean), a_min=-20, a_max=20)
        if self.center_point[0] < -0.05:
            terminated = True
        object_mean = np.mean(self.x.to_numpy()[self.robot_particles_num:], axis=0)
        location_reward += 200 * (object_mean[0] - 0.585)
        if object_mean[1] < 0.24:
            terminated = True
            if object_mean[0] < 0.67:
                location_reward -= 100
            else:
                location_reward += 100
        # action
        action_reward = -np.sum(np.linalg.norm(self.action, axis=(1, 2)))
        # split
        split = np.clip(np.linalg.norm([np.std(self.x.to_numpy()[:self.robot_particles_num, 0]),\
            np.std(self.x.to_numpy()[:self.robot_particles_num, 1])]), a_min=0, a_max=0.2)
        if split > 0.12:
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
            self.gui.circles(self.x.to_numpy() - self.anchor[None].to_numpy(),
                        radius=3,
                        palette=[0xFF5722, 0xe17c2d],
                        palette_indices=self.material)
            self.gui.line(begin=(0, 20 / 128 - 0.015), end=(1, 20 / 128 - 0.015), radius=7, color=0x647D8E)
            self.gui.line(begin=(40 / 128 - self.anchor[None][0], GAP_1 / 128), end=(69 / 128 - self.anchor[None][0], GAP_1 / 128), radius=3.5, color=0x394C31)
            # horizon
            i = 0
            while (GAP + 1) / 128 + 0.001 * i <= 1:
                self.gui.line(begin=(max(0, 33 / 128 - self.anchor[None][0]), GAP / 128 + 0.001 * i), end=(min(1, 44 / 128 - self.anchor[None][0]), GAP / 128 + 0.001 * i), radius=1.2, color=0x394C31)
                i += 1
            i = 0
            while GAP_1 / 128 + 0.003 - 0.001 * i >= 20 / 128:
                self.gui.line(begin=(max(0, 81 / 128 - self.anchor[None][0]), GAP_1 / 128 + 0.003 - 0.001 * i), end=(min(1, 84.5 / 128 - self.anchor[None][0]), GAP_1 / 128 + 0.003 - 0.001 * i), radius=1.2, color=0x394C31)
                i += 1  
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

    @ti.kernel
    def set_bg_env(self):
        for i, j in self.shape_field:
            self.shape_field[i, j] = 0.0
            self.vx_field[i, j] = 0.0
            self.vy_field[i, j] = 0.0
            if (self.n_grid * 8 - i) / (self.n_grid * 8) > GAP / 128 and (j / (self.n_grid * 8) + self.anchor[None][0] > 32 / 128 and j / (self.n_grid * 8) + self.anchor[None][0] < 45 / 128):
                self.shape_field[i, j] = 1
            if (self.n_grid * 8 - i) / (self.n_grid * 8) <= GAP_1 / 128 and (j / (self.n_grid * 8) + self.anchor[None][0] >= 81 / 128 and j / (self.n_grid * 8) + self.anchor[None][0] <= 85 / 128):
                self.shape_field[i, j] = 1
            if (GAP_1 - 2) / 128 <= (self.n_grid * 8 - i) / (self.n_grid * 8) <= GAP_1 / 128 and (j / (self.n_grid * 8) + self.anchor[None][0] > 45 / 128 and j / (self.n_grid * 8) + self.anchor[None][0] < 70 / 128):
                self.shape_field[i, j] = 1

    @ti.kernel
    def grid_operation(self):
        # specific grid operation for SLOT
        for i, j in self.grid_m:
            inv_m = 1 / (self.grid_m[i, j] + 1e-10)
            self.grid_v[i, j] = inv_m * self.grid_v[i, j]
            self.grid_v[i, j][0] += self.dt * self.gravity[None][0]
            self.grid_v[i, j][1] += self.dt * self.gravity[None][1]
            # self.grid_v[i, j] = 0.999 * self.grid_v[i, j]
            # # infinite horizon
            if i > 32 - int(self.anchor[None][0] * self.n_grid) - self.bound and i < 35 - int(self.anchor[None][0] * self.n_grid) and j > GAP - 1 and self.grid_v[i, j][0] > 0:
                self.grid_v[i, j][0] = 0
            if i < 45 - int(self.anchor[None][0] * self.n_grid) + self.bound and i > 43 - int(self.anchor[None][0] * self.n_grid) and j > GAP - 1 and self.grid_v[i, j][0] < 0:
                self.grid_v[i, j][0] = 0

            if i > 45 - int(self.anchor[None][0] * self.n_grid) - self.bound and i < 47 - int(self.anchor[None][0] * self.n_grid) and j == GAP_1 and self.grid_v[i, j][0] > 0:
                self.grid_v[i, j][0] = 0
            if i < 70 - int(self.anchor[None][0] * self.n_grid) + self.bound and i > 68 - int(self.anchor[None][0] * self.n_grid) and j == GAP_1 and self.grid_v[i, j][0] < 0:
                self.grid_v[i, j][0] = 0

            if i > 81 - int(self.anchor[None][0] * self.n_grid) - self.bound and i < 83 - int(self.anchor[None][0] * self.n_grid) and j < GAP_1 and self.grid_v[i, j][0] > 0:
                self.grid_v[i, j][0] = 0
            if i < 85 - int(self.anchor[None][0] * self.n_grid) + self.bound and i > 83 - int(self.anchor[None][0] * self.n_grid) and j < GAP_1 and self.grid_v[i, j][0] < 0:
                self.grid_v[i, j][0] = 0

            # up
            if (j < self.bound * 20 and self.grid_v[i, j][1] < 0) or\
                 (i > 45 - int(self.anchor[None][0] * self.n_grid) - self.bound and i <= 70 - int(self.anchor[None][0] * self.n_grid) and GAP_1 == j and self.grid_v[i, j][1] < 0) or\
                 (i > 82 - int(self.anchor[None][0] * self.n_grid) - self.bound and i <= 85 - int(self.anchor[None][0] * self.n_grid) and GAP_1 == j and self.grid_v[i, j][1] < 0):
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
            if (j > self.n_grid - self.bound * 10 and self.grid_v[i, j][1] > 0) or\
                (i > 31 - int(self.anchor[None][0] * self.n_grid) and i < 46 - int(self.anchor[None][0] * self.n_grid) and j >= GAP - 1 and j <= GAP and self.grid_v[i, j][1] > 0) or\
                (i >= 45 - int(self.anchor[None][0] * self.n_grid) and i <= 70 - int(self.anchor[None][0] * self.n_grid) and j == GAP_1 and self.grid_v[i, j][1] > 0):
                    self.grid_v[i, j][1] = 0

            if i == 0 or i == self.n_grid - 1 or j == 0 or j == self.n_grid - 1:
                self.grid_v[i, j] = [0, 0]

if __name__ == "__main__":
    ti.init(arch=ti.gpu)
    gui = ti.GUI("Taichi MPM Morphological Maze", res=512, background_color=0x112F41, show_gui=False)
    env = SLOT("./cfg/slot.json")
    env.reset()
    env.render(gui, log=True, record_id=0)
    while True:
        env.step(2 * np.random.rand(env.action_space.shape[0]) - 1)
    