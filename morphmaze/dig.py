import os
import cv2
import gym
import json
import numpy as np
import taichi as ti
from morphmaze.morphmaze import morphmaze

OBS_ACT_CENTER_X = 0.5
OBS_ACT_CENTER_Y = 0.4

@ti.data_oriented
class DIG(morphmaze):
    def __init__(self, cfg_path=None, action_dim=2*8**2):
        super(DIG, self).__init__(cfg_path=cfg_path, action_dim=action_dim)
        print("*******************Morphological Maze DIG-v0*******************")
        # initial robot task-DIG
        self.add_circle(-0.05, 0.5, 0.14, is_object=False) 
        self.add_dust(-0.25, 0.0, 0.5, 0.33)
        self.target = ti.Vector.field(2, dtype=float, shape=())
        self.target[None] = [0.2, 0.17]
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
        return self.state

    def step(self, action):
        self.update_grid_actuation(action)
        for i in range(self.repeat_times):
            self.update_particle_actuation()
            self.p2g()
            self.grid_operation()
            self.g2p()
            if self.visualize and i == 0:
                self.render(self.gui, log=True)
        # state (relative x, y)
        x_numpy = self.x.to_numpy()
        self.anchor[None] = [-0.5, 0.0]
        self.center_point = np.mean(x_numpy[:self.robot_particles_num], axis=0)
        if not np.isnan(self.center_point).any():
            self.prev_location = self.center_point
        else:
            self.center_point = self.prev_location
        self.set_obs_field()
        self.update_obs(fix_x=OBS_ACT_CENTER_X, fix_y=OBS_ACT_CENTER_Y)
        if not os.path.exists("./observation"):
            os.makedirs("./observation")
        cv2.imwrite("./observation/state.png", self.state[0])
        cv2.imwrite("./observation/vx.png", self.state[1])
        cv2.imwrite("./observation/vy.png", self.state[2])
        terminated = False
        # # location
        location = np.mean(np.linalg.norm(self.x.to_numpy()[:self.robot_particles_num] - self.target[None].to_numpy(), ord=1, axis=1))
        location_reward = -np.sqrt(location)
        # velocity
        vx_mean = np.mean(self.v.to_numpy()[:self.robot_particles_num, 0]) 
        vy_mean = - np.mean(self.v.to_numpy()[:self.robot_particles_num, 1])
        velocity_reward = np.sign(vx_mean) * np.clip((2 * vx_mean)**2 + 5 * abs(vx_mean), a_min=-20, a_max=20) + np.sign(vy_mean) * np.clip((2 * vy_mean)**2 + 5 * abs(vy_mean), a_min=-20, a_max=20)
        # action
        action_reward = -np.sum(np.linalg.norm(self.action, axis=(1, 2)))
        # split
        split = np.clip(
            np.linalg.norm(
                [np.std(self.x.to_numpy()[:self.robot_particles_num, 0]), np.std(self.x.to_numpy()[:self.robot_particles_num, 1])]
            ),
            a_min=0,
            a_max=0.2,
        )
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
            self.gui.line(begin=(0, 20 / 128 - 0.015), end=(1, 20 / 128 - 0.015), radius=7, color=0x647D8E)
            self.gui.circles(self.x.to_numpy() - self.anchor[None].to_numpy(),
                        radius=3,
                        palette=[0xFF5722, 0xA55D35],
                        palette_indices=self.material)
            i = 0
            while 33 / 128 + 0.001 * i <= 35.8 / 128:
                self.gui.line(begin=(max(0, 33 / 128 + 0.001 * i - self.anchor[None][0]), 20 / 128), end=(min(1, 33 / 128 + 0.001 * i - self.anchor[None][0]), 128 / 128), radius=3.5, color=0x394C31)
                i += 1  
            i = 0
            while -35 / 128 + 0.001 * i <= -32.2 / 128:
                self.gui.line(begin=(max(0, -35 / 128 + 0.001 * i - self.anchor[None][0]), 20 / 128), end=(min(1, -35 / 128 + 0.001 * i - self.anchor[None][0]), 128 / 128), radius=3.5, color=0x394C31)
                i += 1  
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
    def set_obs_field(self):
        for i, j in self.shape_field:
            self.shape_field[i, j] = 0.0
            self.vx_field[i, j] = 0.0
            self.vy_field[i, j] = 0.0
            if (self.target[None][0] - self.anchor[None][0]) * 8 * self.n_grid - 10 < j < (self.target[None][0] - self.anchor[None][0]) * 8 * self.n_grid + 10 and\
                (1 - (self.target[None][1] - self.anchor[None][1])) * 8 * self.n_grid - 10 < i < (1 - (self.target[None][1] - self.anchor[None][1])) * 8 * self.n_grid + 10:
                self.shape_field[i, j] = 1.0
        for p in range(self.n_particles):
            if (
                self.x[p][0] - self.anchor[None][0] > 1e-5
                and self.x[p][0] - self.anchor[None][0] < 1.0 - 1e-5
                and self.material[p] == 0
            ):
                self.shape_field[
                    int((1 - (self.x[p][1] - self.anchor[None][1])) * 8 * self.n_grid),
                    int((self.x[p][0] - self.anchor[None][0]) * 8 * self.n_grid),
                ] = 1
                self.vx_field[
                    int((1 - (self.x[p][1] - self.anchor[None][1])) * 8 * self.n_grid),
                    int((self.x[p][0] - self.anchor[None][0]) * 8 * self.n_grid),
                ] = ti.max(-10, ti.min(10, 2 * self.v[p][0]))
                self.vy_field[
                    int((1 - (self.x[p][1] - self.anchor[None][1])) * 8 * self.n_grid),
                    int((self.x[p][0] - self.anchor[None][0]) * 8 * self.n_grid),
                ] = ti.max(-10, ti.min(10, 2 * self.v[p][1]))
                
            if (
                self.x[p][0] - self.anchor[None][0] > 1e-5
                and self.x[p][0] - self.anchor[None][0] < 1.0 - 1e-5
                and self.material[p] == 1
            ):
                self.shape_field[
                    int((1 - (self.x[p][1] - self.anchor[None][1])) * 8 * self.n_grid),
                    int((self.x[p][0] - self.anchor[None][0]) * 8 * self.n_grid),
                ] = 1

    @ti.kernel
    def p2g(self):
        for i, j in self.grid_m:
            self.grid_v[i, j] = [0, 0]
            self.grid_m[i, j] = 0

        for p in self.x:  # Particle to grid (P2G)
            if self.material[p] == 0:
                h = 0.05
                mu, la = self.mu_0 * h, self.lambda_0 * h
                if (
                    self.x[p][0] - self.anchor[None][0] > 1e-5
                    and self.x[p][0] - self.anchor[None][0] < 1.0 - 1e-5
                ):
                    base = ((self.x[p] - self.anchor[None]) * self.inv_dx - 0.5).cast(int)
                    fx = (self.x[p] - self.anchor[None]) * self.inv_dx - base.cast(float)
                    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
                    self.F[p] = (
                        ti.Matrix.diag(dim=2, val=1) + self.dt * self.C[p]
                    ) @ self.F[p]
                    self.U[p], self.sig[p], self.V[p] = ti.svd(self.F[p])
                    self.F[p] = self.compute_von_mises(
                        self.F[p], self.U[p], self.sig[p], self.V[p], self.yield_stress, mu
                    )
                    J = self.F[p].determinant()
                    r, s = ti.polar_decompose(self.F[p])
                    act = (
                        ti.Matrix([[1.0, 0.0], [0.0, 0.0]]) * self.particle_actuation[p][0]
                        + ti.Matrix([[0.0, 0.0], [0.0, 1.0]]) * self.particle_actuation[p][1]
                    )
                    cauchy = (
                        2 * mu * (self.F[p] - r) @ self.F[p].transpose()
                        + ti.Matrix.diag(2, la * (J - 1) * J)
                        + self.F[p] @ act @ self.F[p].transpose()
                    )
                    stress = (
                        -(self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * cauchy
                    )
                    affine = stress + self.p_mass[p] * self.C[p]
                    for i, j in ti.static(ti.ndrange(3, 3)):
                        offset = ti.Vector([i, j])
                        dpos = (offset.cast(float) - fx) * self.dx
                        weight = w[i][0] * w[j][1]
                        self.grid_v[base + offset] += weight * (
                            self.p_mass[p] * self.v[p] + affine @ dpos
                        )
                        self.grid_m[base + offset] += weight * self.p_mass[p]
            elif self.material[p] == 1:
                h = ti.max(0.1, ti.min(5, ti.exp(10 * (1.0 - self.Jp[p]))))
                mu, la = 5e3 / (2 * (1 + 0.2)) * h, 5e3 * 0.2 / ((1 + 0.2) * (1 - 2 * 0.2)) * h
                if (
                    self.x[p][0] - self.anchor[None][0] > 1e-5
                    and self.x[p][0] - self.anchor[None][0] < 1.0 - 1e-5
                ):
                    base = ((self.x[p] - self.anchor[None]) * self.inv_dx - 0.5).cast(int)
                    fx = (self.x[p] - self.anchor[None]) * self.inv_dx - base.cast(float)
                    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
                    self.F[p] = (ti.Matrix.identity(float, 2) + self.dt * self.C[p]) @ self.F[p]
                    self.U[p], self.sig[p], self.V[p] = ti.svd(self.F[p])
                    J = 1.0
                    # clamp sig to forget about large deformation (strech and compress)
                    for d in ti.static(range(2)):
                        new_sig = ti.min(ti.max(self.sig[p][d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
                        # plastic deformation
                        self.Jp[p] *= self.sig[p][d, d] / new_sig
                        self.sig[p][d, d] = new_sig
                        # only consider elastic deformation
                        J *= self.sig[p][d, d]
                    self.F[p] = self.U[p] @ self.sig[p] @ self.V[p].transpose()
                    stress = 2 * mu * (self.F[p] - self.U[p] @ self.V[p].transpose()) @ self.F[p].transpose(
                    ) + ti.Matrix.identity(float, 2) * la * J * (J - 1)
                    stress = (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * stress # formula #4 PK1 stress
                    affine = stress + self.p_mass[p] * self.C[p]
                    for i, j in ti.static(ti.ndrange(3, 3)):
                        offset = ti.Vector([i, j])
                        dpos = (offset.cast(float) - fx) * self.dx
                        weight = w[i][0] * w[j][1]
                        self.grid_v[base + offset] += weight * (
                            self.p_mass[p] * self.v[p] + affine @ dpos
                        )
                        self.grid_m[base + offset] += weight * self.p_mass[p]

    @ti.kernel
    def grid_operation(self):
        # specific grid operation for DIG
        for i, j in self.grid_m:
            inv_m = 1 / (self.grid_m[i, j] + 1e-10)
            self.grid_v[i, j] = inv_m * self.grid_v[i, j]
            self.grid_v[i, j][0] += self.dt * self.gravity[None][0]
            self.grid_v[i, j][1] += self.dt * self.gravity[None][1]
            # self.grid_v[i, j] = 0.999 * self.grid_v[i, j]
            # # infinite horizon
            if i > 32 - int(self.anchor[None][0] * self.n_grid) - self.bound and i < 35 - int(self.anchor[None][0] * self.n_grid) and self.grid_v[i, j][0] > 0:
                self.grid_v[i, j][0] = 0
            if i < -32 - int(self.anchor[None][0] * self.n_grid) + self.bound and i > -35 - int(self.anchor[None][0] * self.n_grid) and self.grid_v[i, j][0] < 0:
                self.grid_v[i, j][0] = 0
            # up
            if j < self.bound * 20 and self.grid_v[i, j][1] < 0:
                self.grid_v[i, j][1] = 0
            # down
            if j > self.n_grid - 3 and self.grid_v[i, j][1] > 0:
                self.grid_v[i, j][1] = 0

if __name__ == "__main__":
    ti.init(arch=ti.gpu)
    gui = ti.GUI("Taichi MPM Morphological Maze", res=512, background_color=0x112F41, show_gui=False)
    env = DIG("./cfg/dig.json")
    env.reset()
    env.render(gui, log=True, record_id=0)
    while True:
        env.step(2 * np.random.rand(env.action_space.shape[0]) - 1)
