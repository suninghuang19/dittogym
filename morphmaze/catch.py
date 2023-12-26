import os
import cv2
import gym
import json
import numpy as np
import taichi as ti
from morphmaze.morphmaze import morphmaze

OBS_ACT_CENTER_X = 0.5
OBS_ACT_CENTER_Y = 0.4
OFFSET = 6

@ti.data_oriented
class CATCH(morphmaze):
    def __init__(self, cfg_path, action_dim, action_res_resize):
        super(CATCH, self).__init__(cfg_path=cfg_path, action_res_resize=action_res_resize, action_dim=action_dim)
        print("*******************Morphological Maze CATCH-v0*******************")
        # initial robot task-CATCH
        self.obs_auto_reset = False
        self.add_circle(0.05, 0.2, 0.18, is_object=False)
        self.add_rectangular(-0.06, 0.0, 0.05, 0.05, is_object=True)
        self.target = ti.Vector.field(2, dtype=float, shape=())
        self.target[None] = [0.15, 0.25]
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
        self.anchor[None] = [-0.5, 0.0]
        self.set_bg_env()
        self.set_obs_field()
        self.update_obs(fix_x=OBS_ACT_CENTER_X, fix_y=OBS_ACT_CENTER_Y)
        self.init_location = np.mean(self.x.to_numpy()[:self.robot_particles_num], axis=0)
        self.prev_location = self.init_location
        self.init_object_location = np.mean(self.x.to_numpy()[self.robot_particles_num:], axis=0)
        self.prev_object_location = self.init_object_location
        self.gui = None     

    def reset(self):
        self.reset_()
        self.anchor[None] = [-0.5, 0.0]
        self.init_location = np.mean(self.x.to_numpy()[:self.robot_particles_num], axis=0)
        self.y_init = np.argpartition(np.linalg.norm(self.x.to_numpy()[:self.robot_particles_num]\
            - np.array([-0.035, 0.105]), axis=1), 1202)[:1201]
        self.x_init = np.argpartition(np.abs(self.x.to_numpy()[:self.robot_particles_num][self.y_init, 0]), 1200)[:1200]
        self.x_target = self.x.to_numpy()[self.y_init[self.x_init]]
        return self.state

    def step(self, action):
        self.update_grid_actuation(action, fix_x=OBS_ACT_CENTER_X, fix_y=OBS_ACT_CENTER_Y)
        for i in range(self.repeat_times):
            self.update_particle_actuation()
            self.p2g()
            self.grid_operation()
            self.g2p()
            if self.visualize and i == 0:
                self.render(self.gui, log=True)
        # state (relative x, y)
        x_numpy = self.x.to_numpy()
        self.y_init = np.argpartition(np.linalg.norm(x_numpy[:self.robot_particles_num]\
            - np.array(self.object_center_point), axis=1), 1202)[:1201]
        self.x_init = np.argpartition(np.abs(x_numpy[:self.robot_particles_num][self.y_init, 0]), 1200)[:1200]
        self.x_target = self.x.to_numpy()[self.y_init[self.x_init]]
        self.center_point = [np.mean(x_numpy[:self.robot_particles_num, 0]), np.mean(x_numpy[:self.robot_particles_num, 1])]
        self.object_center_point = [np.mean(x_numpy[self.robot_particles_num:, 0]), np.mean(x_numpy[self.robot_particles_num:, 1])]
        self.set_bg_env()
        self.set_obs_field()
        self.update_obs(fix_x=OBS_ACT_CENTER_X, fix_y=OBS_ACT_CENTER_Y)
        # if not os.path.exists("./observation"):
        #     os.makedirs("./observation")
        # cv2.imwrite("./observation/state.png", self.state[0])
        # cv2.imwrite("./observation/vx.png", self.state[1])
        # cv2.imwrite("./observation/vy.png", self.state[2])
        if not np.isnan(self.center_point).any():
            self.prev_location = self.center_point
        else:
            self.center_point = self.prev_location

        if not np.isnan(self.object_center_point).any():
            self.prev_ball_location = self.object_center_point
        else:
            self.object_center_point = self.prev_ball_location
            
        terminated = False
        # # location
        location_reward = 0
        robot_mean = self.x.to_numpy()[self.y_init[self.x_init]].mean(axis=0)
        ball_mean = self.object_center_point
        ball_location_distance = -np.clip(abs(ball_mean[0] - self.target[None][0])\
            + abs(ball_mean[1] - self.target[None][1]), a_min=0, a_max=0.5)
        robot_ball_distance = -np.clip(abs(robot_mean[0] - ball_mean[0])\
            + abs(robot_mean[1] - ball_mean[1]), a_min=0, a_max=0.5)
        location_reward = 5 * ball_location_distance + 3 * robot_ball_distance
        # velocity
        velocity_reward = 0
        # action
        action_reward = -np.sum(np.linalg.norm(self.action, axis=(1, 2)))
        # split
        split = np.clip(np.linalg.norm([np.std(self.x.to_numpy()[:self.robot_particles_num, 0]),\
            np.std(self.x.to_numpy()[:self.robot_particles_num, 1])]), a_min=0, a_max=0.2)
        if split > 0.13:
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
            self.gui.line(begin=(0, 20 / 128 - 0.015), end=(1, 20 / 128 - 0.015), radius=7, color=0x647D8E)

            i = 0
            while 32 / 128 + 0.001 * i <= 35 / 128:
                self.gui.line(begin=(max(0, 32 / 128 + 0.001 * i - self.anchor[None][0]), 20 / 128), end=(min(1, 32 / 128 + 0.001 * i - self.anchor[None][0]), 60 / 128), radius=3.5, color=0x394C31)
                i += 1  
            i = 0
            while -35 / 128 + 0.001 * i <= -32 / 128:
                self.gui.line(begin=(max(0, -35 / 128 + 0.001 * i - self.anchor[None][0]), 20 / 128), end=(min(1, -35 / 128 + 0.001 * i - self.anchor[None][0]), 60 / 128), radius=3.5, color=0x394C31)
                i += 1  
            i = 0
            while 20 / 128 + 0.001 * i <= 27 / 128:
                self.gui.line(begin=(max(0, (OFFSET + 10) / 128 - self.anchor[None][0]), 20 / 128 + 0.001 * i), end=(min(1, 32 / 128 - self.anchor[None][0]), 20 / 128 + 0.001 * i), radius=3.5, color=0x394C31)
                i += 1  
            i = 0
            while 27 / 128 + 0.001 * i <= 37 / 128:
                self.gui.line(begin=(max(0, (OFFSET + 20) / 128 - self.anchor[None][0]), 27 / 128 + 0.001 * i), end=(min(1, 32 / 128 - self.anchor[None][0]), 27 / 128 + 0.001 * i), radius=3.5, color=0x394C31)
                i += 1  
            i = 0
            while 37 / 128 + 0.001 * i <= 45 / 128:
                self.gui.line(begin=(max(0, (OFFSET -5) / 128 - self.anchor[None][0]), 37 / 128 + 0.001 * i), end=(min(1, 32 / 128 - self.anchor[None][0]), 37 / 128 + 0.001 * i), radius=3.5, color=0x394C31)
                i += 1  
            i = 0
            while 20 / 128 + 0.001 * i <= 27 / 128:
                self.gui.line(begin=(max(0, -32 / 128 - self.anchor[None][0]), 20 / 128 + 0.001 * i), end=(min(1, (-OFFSET - 20) / 128 - self.anchor[None][0]), 20 / 128 + 0.001 * i), radius=3.5, color=0x394C31)
                i += 1  
            i = 0
            while 27 / 128 + 0.001 * i <= 37 / 128:
                self.gui.line(begin=(max(0, -32 / 128 - self.anchor[None][0]), 27 / 128 + 0.001 * i), end=(min(1, (-OFFSET - 10) / 128 - self.anchor[None][0]), 27 / 128 + 0.001 * i), radius=3.5, color=0x394C31)
                i += 1  
            i = 0
            while 37 / 128 + 0.001 * i <= 45 / 128:
                self.gui.line(begin=(max(0, -32 / 128 - self.anchor[None][0]), 37 / 128 + 0.001 * i), end=(min(1, (-OFFSET - 5) / 128 - self.anchor[None][0]), 37 / 128 + 0.001 * i), radius=3.5, color=0x394C31)
                i += 1  

            self.gui.circles(self.x.to_numpy() - self.anchor[None].to_numpy(),
                        radius=1.5,
                        palette=[0xFF5722, 0xe17c2d],
                        palette_indices=self.material)

            if 1e-5 < self.target[None][0] - self.anchor[None][0] < 1 - 1e-5 and 1e-5 < self.target[None][1] - self.anchor[None][1] < 1 - 1e-5:
                self.gui.circle([self.target[None][0] - self.anchor[None][0], self.target[None][1] - self.anchor[None][1]], radius=5, color=0x7F3CFF)
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
            if (self.target[None][0] - self.anchor[None][0]) * 8 * self.n_grid - 10 < j < (self.target[None][0] - self.anchor[None][0]) * 8 * self.n_grid + 10 and\
                (1 - (self.target[None][1] - self.anchor[None][1])) * 8 * self.n_grid - 10 < i < (1 - (self.target[None][1] - self.anchor[None][1])) * 8 * self.n_grid + 10:
                self.shape_field[i, j] = 1.0
            # right wall
            if 37 / 128 < (self.n_grid * 8 - i) / (self.n_grid * 8) < 45 / 128 and (OFFSET - 5) / 128 < (j / (self.n_grid * 8) + self.anchor[None][0]) < 32 / 128:
                self.shape_field[i, j] = 1
            if 27 / 128 < (self.n_grid * 8 - i) / (self.n_grid * 8) < 37 / 128 and (OFFSET + 20) / 128 < (j / (self.n_grid * 8) + self.anchor[None][0]) < 32 / 128:
                self.shape_field[i, j] = 1
            if 20 / 128 < (self.n_grid * 8 - i) / (self.n_grid * 8) < 27 / 128 and (OFFSET + 10) / 128 < (j / (self.n_grid * 8) + self.anchor[None][0]) < 32 / 128:
                self.shape_field[i, j] = 1
            # left wall
            if 37 / 128 < (self.n_grid * 8 - i) / (self.n_grid * 8) < 45 / 128 and (-OFFSET - 5) / 128 > (j / (self.n_grid * 8) + self.anchor[None][0]) > -32 / 128:
                self.shape_field[i, j] = 1
            if 27 / 128 < (self.n_grid * 8 - i) / (self.n_grid * 8) < 37 / 128 and -(OFFSET + 10) / 128 > (j / (self.n_grid * 8) + self.anchor[None][0]) > -32 / 128:
                self.shape_field[i, j] = 1
            if 27 / 128 < (self.n_grid * 8 - i) / (self.n_grid * 8) < 37 / 128 and -(OFFSET + 20) / 128 > (j / (self.n_grid * 8) + self.anchor[None][0]) > -32 / 128:
                self.shape_field[i, j] = 1
            if 20 / 128 < (self.n_grid * 8 - i) / (self.n_grid * 8) < 27 / 128 and -(OFFSET + 20) / 128 > (j / (self.n_grid * 8) + self.anchor[None][0]) > -32 / 128:
                self.shape_field[i, j] = 1

    @ti.kernel
    def grid_operation(self):
        # specific grid operation for CATCH
        for i, j in self.grid_m:
            inv_m = 1 / (self.grid_m[i, j] + 1e-10)
            self.grid_v[i, j] = inv_m * self.grid_v[i, j]
            self.grid_v[i, j][0] += self.dt * self.gravity[None][0]
            self.grid_v[i, j][1] += self.dt * self.gravity[None][1]
            self.grid_v[i, j] = 0.999 * self.grid_v[i, j]

            if i > 32 - int(self.anchor[None][0] * self.n_grid) - self.bound and i < 35 - int(self.anchor[None][0] * self.n_grid) and j < 80 - int(self.anchor[None][1] * self.n_grid) and self.grid_v[i, j][0] > 0:
                self.grid_v[i, j][0] = 0
            if i < -32 - int(self.anchor[None][0] * self.n_grid) + self.bound and i > -35 - int(self.anchor[None][0] * self.n_grid) and j < 80 - int(self.anchor[None][1] * self.n_grid) and self.grid_v[i, j][0] < 0:
                self.grid_v[i, j][0] = 0

            # right wall
            # left
            if i > OFFSET - 5 - int(self.anchor[None][0] * self.n_grid) - self.bound and i < OFFSET - 5 + 2 - int(self.anchor[None][0] * self.n_grid) and 36 - int(self.anchor[None][1] * self.n_grid) < j < 46 - int(self.anchor[None][1] * self.n_grid) and self.grid_v[i, j][0] > 0:
                self.grid_v[i, j][0] = 0
            if i > OFFSET + 20 - int(self.anchor[None][0] * self.n_grid) - self.bound and i < OFFSET + 20 + 2 - int(self.anchor[None][0] * self.n_grid) and 26 - int(self.anchor[None][1] * self.n_grid) < j < 38 - int(self.anchor[None][1] * self.n_grid) and self.grid_v[i, j][0] > 0:
                self.grid_v[i, j][0] = 0        
            if i > OFFSET + 10 - int(self.anchor[None][0] * self.n_grid) - self.bound and i < OFFSET + 10 + 2 - int(self.anchor[None][0] * self.n_grid) and 19 - int(self.anchor[None][1] * self.n_grid) < j < 28 - int(self.anchor[None][1] * self.n_grid) and self.grid_v[i, j][0] > 0:
                self.grid_v[i, j][0] = 0  
            # up
            if i > OFFSET - 5 - int(self.anchor[None][0] * self.n_grid) and i < 32 - int(self.anchor[None][0] * self.n_grid) and 44 - int(self.anchor[None][1] * self.n_grid) < j < 46 - int(self.anchor[None][1] * self.n_grid) and self.grid_v[i, j][1] < 0:
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
            if i > OFFSET + 10 - int(self.anchor[None][0] * self.n_grid) and i < OFFSET + 20 - int(self.anchor[None][0] * self.n_grid) and 26 - int(self.anchor[None][1] * self.n_grid) < j < 28 - int(self.anchor[None][1] * self.n_grid) and self.grid_v[i, j][1] < 0:
                self.grid_v[i, j][1] = 0
            # down
            if i > OFFSET - 5 - int(self.anchor[None][0] * self.n_grid) and i < OFFSET + 20 - int(self.anchor[None][0] * self.n_grid) and 36 - int(self.anchor[None][1] * self.n_grid) < j < 38 - int(self.anchor[None][1] * self.n_grid) and self.grid_v[i, j][1] > 0:
                self.grid_v[i, j][1] = 0
                
            # left wall
            # right
            if i < -OFFSET - 5 - int(self.anchor[None][0] * self.n_grid) + self.bound and i > -OFFSET - 5 - 2 - int(self.anchor[None][0] * self.n_grid) and 36 - int(self.anchor[None][1] * self.n_grid) < j < 46 - int(self.anchor[None][1] * self.n_grid) and self.grid_v[i, j][0] < 0:
                self.grid_v[i, j][0] = 0
            if i < -OFFSET - 10 - int(self.anchor[None][0] * self.n_grid) + self.bound and i > -OFFSET - 10 - 2 - int(self.anchor[None][0] * self.n_grid) and 26 - int(self.anchor[None][1] * self.n_grid) < j < 38 - int(self.anchor[None][1] * self.n_grid) and self.grid_v[i, j][0] < 0:
                self.grid_v[i, j][0] = 0        
            if i < -OFFSET - 20 - int(self.anchor[None][0] * self.n_grid) + self.bound and i > -OFFSET - 20 - 2 - int(self.anchor[None][0] * self.n_grid) and 19 - int(self.anchor[None][1] * self.n_grid) < j < 28 - int(self.anchor[None][1] * self.n_grid) and self.grid_v[i, j][0] < 0:
                self.grid_v[i, j][0] = 0  
            # up
            if i < -OFFSET - 5 - int(self.anchor[None][0] * self.n_grid) and i > -32 - int(self.anchor[None][0] * self.n_grid) and 44 - int(self.anchor[None][1] * self.n_grid) < j < 46 - int(self.anchor[None][1] * self.n_grid) and self.grid_v[i, j][1] < 0:
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
            if i < -OFFSET - 5 - int(self.anchor[None][0] * self.n_grid) and i > -OFFSET - 10 - int(self.anchor[None][0] * self.n_grid) and 36 - int(self.anchor[None][1] * self.n_grid) < j < 38 - int(self.anchor[None][1] * self.n_grid) and self.grid_v[i, j][1] > 0:
                self.grid_v[i, j][1] = 0
            # down
            if i < -OFFSET - 10 - int(self.anchor[None][0] * self.n_grid) and i > -OFFSET - 20 - int(self.anchor[None][0] * self.n_grid) and 26 - int(self.anchor[None][1] * self.n_grid) < j < 28 - int(self.anchor[None][1] * self.n_grid) and self.grid_v[i, j][1] > 0:
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
            if j > self.n_grid - 3 and self.grid_v[i, j][1] > 0:
                self.grid_v[i, j][1] = 0

if __name__ == "__main__":
    ti.init(arch=ti.gpu)
    gui = ti.GUI("Taichi MPM Morphological Maze", res=512, background_color=0x112F41, show_gui=False)
    env = CATCH("./cfg/catch.json")
    env.reset()
    env.render(gui, log=True, record_id=0)
    while True:
        env.step(2 * np.random.rand(env.action_space.shape[0]) - 1)