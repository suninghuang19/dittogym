import os
import cv2
import gym
import json
import numpy as np
import taichi as ti
from morphmaze.morphmaze import morphmaze

@ti.data_oriented
class shapematch(morphmaze):
    def __init__(self, cfg_path, action_res, action_res_resize, wandb_logger=None, robot_img_path=None, particles_num=15000):
        super(shapematch, self).__init__(cfg_path=cfg_path, action_res_resize=action_res_resize,\
            action_res=action_res, wandb_logger=wandb_logger)
        print("*******************Morphological_Maze SHAPE_MATCH*******************")
        # initial robot task-SHAPE_MATCH
        if robot_img_path is not None:
            self.cfg["particle_num_list"][0] = particles_num
            self.set_params(self.cfg)
            self.add_self_designed_robot(robot_img_path, particles_num)
        else:
            self.add_circle(0.0, 0.0, 0.17, is_object=False)
        print("n_particles: ", self.n_particles, "action_space: ", self.action_space.shape)
        self.target_robot = cv2.imread(os.path.join(self.current_directory,\
            "./target_for_shape_match/{}.jpg".format(self.cfg["target"])), cv2.IMREAD_GRAYSCALE).astype(np.int32)
        self.target_robot[self.target_robot < 125] = 0
        self.target_robot[self.target_robot >= 125] = 255
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
        self.anchor[None] = [0.0, 0.0]
        self.set_obs_field()
        self.update_obs()
        self.init_location = np.mean(self.x.to_numpy()[:self.robot_particles_num], axis=0)
        self.prev_location = self.init_location
        self.gui = None

    def reset(self):
        self.reset_()
        return self.state

    def step(self, action):
        self.update_grid_actuation(action)
        for i in range(self.repeat_times):
            self.update_particle_actuation()
            self.p2g()
            self.grid_operation()
            self.g2p()
        # state (relative x, y)
        x_numpy = self.x.to_numpy()
        self.center_point = [np.mean(x_numpy[:self.robot_particles_num, 0]),\
            np.mean(x_numpy[:self.robot_particles_num, 1])]
        self.set_obs_field()
        self.update_obs()
        # if not os.path.exists("./observation"):
        #     os.makedirs("./observation")
        # cv2.imwrite("./observation/state.png", self.state[0])
        # cv2.imwrite("./observation/vx.png", self.state[1])
        # cv2.imwrite("./observation/vy.png", self.state[2])
        terminated = False
        # shape
        shape_reward = 0
        new_state = self.state[0].astype(np.int32)
        new_state[new_state < 80] = 0
        new_state[new_state >= 80] = 255
        shape_reward = -0.1 * np.sum(np.abs(new_state - self.target_robot)) / 255
        reward = (shape_reward + self.reward_params[0])
        # info
        info = {}

        if np.isnan(self.state).any():
            raise ValueError("state has nan")   
        
        return (self.state, reward, terminated, False, info)

    def render(self, gui, record=False, record_id=None, mode=None):
        self.gui = gui
        if not record:
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
            image = cv2.imread(os.path.join(self.current_directory, "./bg/bg.png"), cv2.IMREAD_COLOR)\
                .astype(np.uint8).transpose(1, 0, 2)[:, ::-1, :]
            image = np.concatenate([image[start_point:512, :, :], image[:start_point, :, :]], axis=0)
            gui.set_image(image)
            self.gui.circles(
                        self.x.to_numpy() - np.array([self.anchor[None][0], 0]),
                        radius=1.5,
                        palette=[0xFF5722, 0x7F3CFF],
                        palette_indices=self.material)
            if not os.path.exists(self.save_file_name + "/videos/record_" + str(self.record_id)):
                os.makedirs(self.save_file_name + "/videos/record_" + str(self.record_id))
            img_path = os.path.join(self.save_file_name 
                                    + "/videos/record_" + str(self.record_id)
                                    + "/frame_%04d.png" % self.frames_num)
            self.gui.show(img_path)
            self.frames_num += 1
            if mode == "rgb_array":
                return cv2.imread(img_path)
            else:
                return None

    @ti.kernel
    def grid_operation(self):
        # specific grid operation for SHAPE_MATCH
        for i, j in self.grid_m:
            inv_m = 1 / (self.grid_m[i, j] + 1e-10)
            self.grid_v[i, j] = inv_m * self.grid_v[i, j]
            self.grid_v[i, j][0] += self.dt * self.gravity[None][0]
            self.grid_v[i, j][1] += self.dt * self.gravity[None][1]
            # self.grid_v[i, j] = 0.999 * self.grid_v[i, j]
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
            if j > self.n_grid - self.bound * 10 and self.grid_v[i, j][1] > 0:
                self.grid_v[i, j][1] = 0
