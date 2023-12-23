import os
import cv2
import gym
import json
import numpy as np
import taichi as ti
from morphmaze.morphmaze import morphmaze

OBS_ACT_CENTER_Y = 0.28

@ti.data_oriented
class KICK(morphmaze):
    def __init__(self, action_dim, cfg_path=None):
        super(KICK, self).__init__(cfg_path=cfg_path, action_dim=action_dim)
        print("*******************Morphological Maze KICK-v0*******************")
        # initial robot task-KICK
        self.add_circle(0.0, 0.0, 0.18, is_object=False)
        self.add_rectangular(0.3, 0.0, 0.05, 0.05, is_object=True)
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
        self.anchor[None] = [np.mean(self.x.to_numpy()[:self.robot_particles_num, 0]) - 0.3, 0.0]
        self.set_obs_field()
        self.update_obs(fix_y=OBS_ACT_CENTER_Y)
        self.init_location = np.mean(self.x.to_numpy()[:self.robot_particles_num], axis=0)
        self.prev_location = self.init_location
        self.init_object_location = np.mean(self.x.to_numpy()[self.robot_particles_num:], axis=0)
        self.prev_object_location = self.init_object_location
        self.gui = None     

    def reset(self):
        self.reset_()
        self.anchor[None] = [self.init_location[0] - 0.3, 0.0]
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
        if not np.isnan(x_numpy[:, 0]).any():
            self.anchor[None] = [np.mean(x_numpy[:self.robot_particles_num, 0]) - 0.3, 0.0]
        else:
            self.anchor[None] = [self.prev_location[0] - 0.3, 0.0]
        self.robot_center_point = [np.mean(x_numpy[:self.robot_particles_num, 0]), np.mean(x_numpy[:self.robot_particles_num, 1])]
        self.object_center_point = [np.mean(x_numpy[self.robot_particles_num:, 0]), np.mean(x_numpy[self.robot_particles_num:, 1])]
        self.set_obs_field()
        self.update_obs(fix_y=OBS_ACT_CENTER_Y)
        # if not os.path.exists("./observation"):
        #     os.makedirs("./observation")
        # cv2.imwrite("./observation/state.png", self.state[0])
        # cv2.imwrite("./observation/vx.png", self.state[1])
        # cv2.imwrite("./observation/vy.png", self.state[2])
        if not np.isnan(self.robot_center_point).any():
            self.prev_location = self.robot_center_point
        else:
            self.robot_center_point = self.prev_location

        if not np.isnan(self.object_center_point).any():
            self.prev_ball_location = self.object_center_point
        else:
            self.object_center_point = self.prev_ball_location
        terminated = False
        # # location
        location_reward = 0
        robot_x_mean = self.robot_center_point[0]
        ball_x_mean = self.object_center_point[0]
        ball_location_reward = np.clip(np.sign(ball_x_mean - self.init_object_location[0]) * (2 * (ball_x_mean - self.init_object_location[0]))**2 + 4 * (ball_x_mean - self.init_object_location[0]), a_min=-20, a_max=30)
        robot_location_reward = np.clip(np.sign(robot_x_mean - self.init_location[0]) * (2 * (robot_x_mean - self.init_location[0]))**2 + 4 * (robot_x_mean - self.init_location[0]), a_min=-20, a_max=30)
        robot_ball_distance = -np.clip(abs(robot_x_mean - ball_x_mean), a_min=0, a_max=0.3)
        location_reward = ball_location_reward + robot_location_reward + 10 * robot_ball_distance
        # velocity
        velocity_reward = 0
        robot_vx_mean = np.mean(self.v.to_numpy()[:self.robot_particles_num, 0])
        ball_vx_mean = np.mean(self.v.to_numpy()[self.robot_particles_num:, 0])
        velocity_reward = np.clip(np.sign(robot_vx_mean) * (2 * robot_vx_mean)**2 + 5 * robot_vx_mean, a_min=-20, a_max=20) + np.clip(np.sign(ball_vx_mean) * (2 * ball_vx_mean)**2 + 5 * ball_vx_mean, a_min=-20, a_max=20)
        # action
        action_reward = -np.sum(np.linalg.norm(self.action, axis=(1, 2)))
        # split
        split = np.clip(np.linalg.norm([np.std(self.x.to_numpy()[:self.robot_particles_num, 0]), np.std(self.x.to_numpy()[:self.robot_particles_num, 1])]), a_min=0, a_max=0.2)
        if split > 0.11:
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
        if robot_x_mean < -0.2:
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
            self.gui.circles(self.x.to_numpy() - self.anchor[None].to_numpy(),
                        radius=1.5,
                        palette=[0xFF5722, 0x7F3CFF],
                        palette_indices=self.material)
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
    def grid_operation(self):
        # specific grid operation for KICK
        for i, j in self.grid_m:
            inv_m = 1 / (self.grid_m[i, j] + 1e-10)
            self.grid_v[i, j] = inv_m * self.grid_v[i, j]
            self.grid_v[i, j][0] += self.dt * self.gravity[None][0]
            self.grid_v[i, j][1] += self.dt * self.gravity[None][1]
            self.grid_v[i, j] = 0.999 * self.grid_v[i, j]
            # # infinite horizon
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
    env = KICK("./cfg/kick.json")
    env.reset()
    env.render(gui, log=True, record_id=0)
    while True:
        env.step(2 * np.random.rand(env.action_space.shape[0]) - 1)