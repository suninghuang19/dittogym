import os
import cv2
import gym
import json
import time
import numpy as np
import taichi as ti


@ti.data_oriented
class morphmaze(gym.Env):
    def __init__(self, action_dim, cfg_path=None):
        print("*******************Welcome to Morphological Maze*******************")
        current_file_path = os.path.abspath(__file__)
        self.current_directory = os.path.dirname(current_file_path)
        if cfg_path is None:
            print("No config file!")
            return
        else:
            cfg = json.load(open(os.path.join(self.current_directory, cfg_path), "r"))
            cfg["action_dim"] = action_dim
            self.set_params(cfg)
        self.obs_auto_reset = True
        self.gui = None
        if not os.path.exists(os.path.join(self.current_directory, "../results")):
            os.makedirs(os.path.join(self.current_directory, "../results"))
        if not os.path.exists(os.path.join(self.current_directory, "../results", self.save_file_name)):
            os.makedirs(os.path.join(self.current_directory, "../results", self.save_file_name))
        json.dump(self.cfg, open(self.current_directory + "/../results/" + self.save_file_name + "/config.json", "w"), indent=4, sort_keys=True)

    def set_params(self, cfg):
        self.reward_params = cfg["reward_params"]
        self.frames_num = 0
        self.visualize = cfg["visualize"]
        self.save_file_name = cfg["save_file_name"] + "_" + time.strftime("%Y%m%d-%H%M%S")

        # mpm params
        self.offset_x = cfg["offset"][0]
        self.offset_y = cfg["offset"][1]
        self.n_particles = np.sum(cfg["particle_num_list"])
        self.robot_particles_num = cfg["particle_num_list"][0]
        self.object_particles_num = self.n_particles - self.robot_particles_num
        self.n_grid = cfg["n_grid"]
        self.dx = 1 / self.n_grid
        self.inv_dx = float(self.n_grid)
        self.x_list = []
        self.mass_list = []
        self.material_list = []
        self.bound = 1
        self.coeff = 0.5
        self.dt = 1e-4
        self.p_vol = 1
        self.mu_0 = cfg["mu_0"]
        self.lambda_0 = cfg["lambda_0"]
        self.yield_stress = cfg["yield_stress"]
        self.repeat_times = cfg["repeat_times"]
        self.mass = cfg["mass"]
        self.mass_save = ti.field(dtype=float, shape=self.n_particles)
        self.p_mass = ti.field(dtype=float, shape=self.n_particles)
        self.x = ti.Vector.field(2, dtype=float, shape=self.n_particles)
        self.x_save = ti.Vector.field(2, dtype=float, shape=self.n_particles)
        self.v = ti.Vector.field(2, dtype=float, shape=self.n_particles)
        self.C = ti.Matrix.field(2, 2, dtype=float, shape=self.n_particles)
        self.F = ti.Matrix.field(2, 2, dtype=float, shape=self.n_particles)
        self.U = ti.Matrix.field(2, 2, dtype=float, shape=self.n_particles)
        self.V = ti.Matrix.field(2, 2, dtype=float, shape=self.n_particles,)
        self.sig = ti.Matrix.field(2, 2, dtype=float, shape=self.n_particles)
        self.material = ti.field(dtype=int, shape=self.n_particles)
        self.material_save = ti.field(dtype=int, shape=self.n_particles)
        self.Jp = ti.field(dtype=float, shape=self.n_particles)
        self.grid_v = ti.Vector.field(2, dtype=float, shape=(self.n_grid, self.n_grid))
        self.grid_m = ti.field(dtype=float, shape=(self.n_grid, self.n_grid))
        self.gravity = ti.Vector.field(2, dtype=float, shape=())
        self.gravity[None] = cfg["gravity"]
        
        # action space (ax, ay)
        self.max_actuation = cfg["max_actuation"]
        self.action_res_resize = cfg["action_res_resize"]
        self.action = np.zeros(cfg["action_dim"])
        self.grid_actuation = ti.Vector.field(2, dtype=float, shape=(self.n_grid, self.n_grid))
        self.particle_actuation = ti.Vector.field(2, dtype=float, shape=self.n_particles)
        self.action_space = gym.spaces.box.Box(low=-1, high=1, shape=(cfg["action_dim"],), dtype=np.float32)
        print("n_particles: ", self.n_particles, "action_space: ", self.action_space.shape)

        # observation space (shape, vx, vy)
        self.obs_res = cfg["obs_res"]
        self.obs_res_resize = cfg["obs_res_resize"]
        self.shape_field = ti.field(dtype=float, shape=(self.obs_res_resize * self.n_grid, self.obs_res_resize * self.n_grid))
        self.vx_field = ti.field(dtype=float, shape=(self.obs_res_resize * self.n_grid, self.obs_res_resize * self.n_grid))
        self.vy_field = ti.field(dtype=float, shape=(self.obs_res_resize * self.n_grid, self.obs_res_resize * self.n_grid))
        low = np.zeros((3, self.obs_res, self.obs_res), dtype=np.uint8)
        high = np.full((3, self.obs_res, self.obs_res), 255, dtype=np.uint8)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.uint8)
        self.cfg = cfg

    def add_circle(self, x, y, r, is_object=False):
        '''
        (x, y) circle center
        r circle radius
        '''
        if is_object:
            particle_num = self.object_particles_num
        else:
            particle_num = self.robot_particles_num
        for _ in range(particle_num):
            randm_x = np.random.rand()
            randm_y = np.random.rand()
            sign_x = 1 if np.random.rand() > 0.5 else -1
            sign_y = 1 if np.random.rand() > 0.5 else -1
            y_length = np.sqrt((0.5 * r)**2 - (0.5 * r * randm_x)**2)
            self.x_list.append(
                [
                    x + 0.5 * (1 + sign_x * randm_x) * r + self.offset_x,
                    y + 0.5  * r + sign_y * randm_y * y_length + self.offset_y
                ]
            )
            self.material_list.append(1 if is_object else 0)
            self.mass_list.append(self.mass[1] if is_object else self.mass[0])

    def add_rectangular(self, x, y, w, h, is_object=False):
        '''
        (x, y): left bottom point of the rectangular
        (w, h): width and height of the rectangular
        '''
        if is_object:
            particle_num = self.object_particles_num
        else:
            particle_num = self.robot_particles_num
        for _ in range(particle_num):
            randm_x = np.random.rand()
            randm_y = np.random.rand()
            sign_x = 1 if np.random.rand() > 0.5 else -1
            sign_y = 1 if np.random.rand() > 0.5 else -1
            self.x_list.append(
                [
                    x + 0.5 * (1 + sign_x * randm_x) * w + self.offset_x,
                    y + 0.5 * (1 + sign_y * randm_y) * h + self.offset_y
                ]
            )
            self.material_list.append(1 if is_object else 0)
            self.mass_list.append(self.mass[1] if is_object else self.mass[0])
    
    def add_dust(self, x, y, w, h):
        '''
        (x, y): left bottom point of the dust
        (w, h): width and height range of the dust
        '''
        for _ in range(self.object_particles_num):
            self.x_list.append(
                [
                    x + np.random.rand() * w + 0.0,
                    y + np.random.rand() * h + 0.16,
                ]
            )
            self.material_list.append(1)
            self.mass_list.append(self.mass[1])

    def central_obs(self, state, fix_x=None, fix_y=None):
        if not np.isnan(self.center_point).any():
            x = self.center_point[0] - self.anchor[None][0] if fix_x is None else fix_x
            y = self.center_point[1] - self.anchor[None][1] if fix_y is None else fix_y
        else:
            x = self.prev_location[0] - self.anchor[None][0] if fix_x is None else fix_x
            y = self.prev_location[1] - self.anchor[None][1] if fix_y is None else fix_y
        mid_index = [int(x * self.inv_dx * self.obs_res_resize),
                    int(y * self.inv_dx * self.obs_res_resize)]
        up_index = mid_index[1] + int(0.5 * self.obs_res * self.obs_res_resize)
        down_index = mid_index[1] - int(0.5 * self.obs_res * self.obs_res_resize)
        left_index = mid_index[0] - int(0.5 * self.obs_res * self.obs_res_resize)
        right_index = mid_index[0] + int(0.5 * self.obs_res * self.obs_res_resize)
        if up_index > self.obs_res_resize * self.n_grid:
            up_index = self.obs_res_resize * self.n_grid
            down_index = up_index - int(1.0 * self.obs_res * self.obs_res_resize)
        if down_index < 0:
            down_index = 0
            up_index = down_index + int(1.0 * self.obs_res * self.obs_res_resize)
        if left_index < 0:
            left_index = 0
            right_index = left_index + int(1.0 * self.obs_res * self.obs_res_resize)
        if right_index > self.obs_res_resize * self.n_grid:
            right_index = self.obs_res_resize * self.n_grid
            left_index = right_index - int(1.0 * self.obs_res * self.obs_res_resize)
        state = state[
            :,
            self.n_grid * self.obs_res_resize - up_index : self.n_grid * self.obs_res_resize - down_index,
            left_index:right_index,
        ]
        return state

    def update_obs(self, fix_x=None, fix_y=None):
        state = self.shape_field.to_numpy().reshape(1, self.obs_res_resize * self.n_grid, self.obs_res_resize * self.n_grid)
        vx = self.vx_field.to_numpy().reshape(1, self.obs_res_resize * self.n_grid, self.obs_res_resize * self.n_grid)
        vy = self.vy_field.to_numpy().reshape(1, self.obs_res_resize * self.n_grid, self.obs_res_resize * self.n_grid)
        state = 255 * self.central_obs(state, fix_x, fix_y).squeeze(0)
        state = (np.clip(cv2.resize(state, (self.obs_res, self.obs_res), interpolation=cv2.INTER_AREA), a_min=0, a_max=255)
                .astype(np.uint8)
                .reshape(1, self.obs_res, self.obs_res))
        vx = (255 * (self.central_obs(vx, fix_x, fix_y) + 10) / 20).squeeze(0)
        vx = cv2.resize(
            vx,
            (self.obs_res, self.obs_res),
            interpolation=cv2.INTER_AREA,
        ).reshape(1, self.obs_res, self.obs_res)
        vx = np.clip(10 * (vx - 127) + 127, a_min=0, a_max=255).astype(np.uint8)
        vy = (255 * (self.central_obs(vy, fix_x, fix_y) + 10) / 20).squeeze(0)
        vy = cv2.resize(
            vy,
            (self.obs_res, self.obs_res),
            interpolation=cv2.INTER_AREA,
        ).reshape(1, self.obs_res, self.obs_res)
        vy = np.clip(10 * (vy - 127) + 127, a_min=0, a_max=255).astype(np.uint8)
        state = np.concatenate([state, vx, vy], axis=0)
        self.state = state

    def update_grid_actuation(self, action, fix_x=None, fix_y=None):
        side_length = int(np.sqrt(self.action_space.shape[0] / 2))
        final_action_res = int(side_length * self.action_res_resize)
        action = action.reshape(2, side_length, side_length)
        action_x = cv2.resize(
            action[0],
            (final_action_res, final_action_res),
            interpolation=cv2.INTER_CUBIC,
        )
        action_y = cv2.resize(
            action[1],
            (final_action_res, final_action_res),
            interpolation=cv2.INTER_CUBIC,
        )
        action = np.stack([action_x, action_y], axis=0)
        self.action = np.clip(self.max_actuation * action, a_min=-self.max_actuation, a_max=self.max_actuation)
        if not np.isnan(self.center_point).any():
            x = self.center_point[0] - self.anchor[None][0] if fix_x is None else fix_x
            y = self.center_point[1] - self.anchor[None][1] if fix_y is None else fix_y
        else:
            x = self.prev_location[0] - self.anchor[None][0] if fix_x is None else fix_x
            y = self.prev_location[1] - self.anchor[None][1] if fix_y is None else fix_y
        base_index = np.array([int(x * self.inv_dx - 0.5 * final_action_res),
                                int(y * self.inv_dx - 0.5 * final_action_res)])
        # grid term use the exact same arrangment as in mpm
        grid_actuation = np.zeros((2, self.n_grid, self.n_grid))
        if base_index[0] + final_action_res > self.n_grid:
            base_index[0] = self.n_grid - final_action_res
        if base_index[0] < 0:
            base_index[0] = 0
        if base_index[1] + final_action_res > self.n_grid:
            base_index[1] = self.n_grid - final_action_res
        if base_index[1] < 0:
            base_index[1] = 0
        grid_actuation[
            :,
            base_index[0] : base_index[0] + final_action_res,
            base_index[1] : base_index[1] + final_action_res,
        ] = self.action
        self.grid_actuation.from_numpy(grid_actuation.transpose(1, 2, 0))

    @ti.kernel
    def update_particle_actuation(self):
        # G2P for action signals
        for p in self.x:
            if (
                self.x[p][0] - self.anchor[None][0] > 1e-5
                and self.x[p][0] - self.anchor[None][0] < 1.0 - 1e-5
            ):
                base = ((self.x[p] - self.anchor[None]) * self.inv_dx - 0.5).cast(int)
                fx = (self.x[p] - self.anchor[None]) * self.inv_dx - base.cast(float)
                w = [
                    0.5 * (1.5 - fx) ** 2,
                    0.75 - (fx - 1.0) ** 2,
                    0.5 * (fx - 0.5) ** 2,
                ]
                new_actuation = ti.Vector.zero(float, 2)
                for i, j in ti.static(ti.ndrange(3, 3)):
                    g_actuation = self.grid_actuation[base + ti.Vector([i, j])]
                    weight = w[i][0] * w[j][1]
                    new_actuation += weight * g_actuation
                self.particle_actuation[p] = new_actuation

    @ti.kernel
    def set_obs_field(self):
        for i, j in self.shape_field:
            if self.obs_auto_reset:
                self.shape_field[i, j] = 0.0
                self.vx_field[i, j] = 0.0
                self.vy_field[i, j] = 0.0
        for p in range(self.n_particles):
            if (
                self.x[p][0] - self.anchor[None][0] > 1e-5
                and self.x[p][0] - self.anchor[None][0] < 1.0 - 1e-5
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

    @ti.kernel
    def reset_(self):
        for i in range(self.n_particles):
            self.x[i] = self.x_save[i]
            self.v[i] = [0, 0]
            self.material[i] = self.material_save[i]
            self.p_mass[i] = self.mass_save[i]
            self.F[i] = [[1, 0], [0, 1]]
            self.Jp[i] = 1
            self.C[i] = [[0, 0], [0, 0]]
            self.particle_actuation[i] = [0, 0]
        for i, j in self.grid_actuation:
            self.grid_actuation[i, j] = [0, 0]

    @ti.func
    def norm(self, x, eps=1e-8):
        return ti.sqrt(x.dot(x) + eps)

    @ti.func
    def make_matrix_from_diag(self, d):
        return ti.Matrix([[d[0], 0.0], [0.0, d[1]]], dt=float)

    @ti.func
    def compute_von_mises(self, F, U, sig, V, yield_stress, mu):
        epsilon = ti.Vector.zero(float, 2)
        sig = ti.max(sig, 0.05)  # add this to prevent NaN in extrem cases
        epsilon = ti.Vector([ti.log(sig[0, 0]), ti.log(sig[1, 1])])
        epsilon_hat = epsilon - (epsilon.sum() / 2)
        epsilon_hat_norm = self.norm(epsilon_hat)
        delta_gamma = epsilon_hat_norm - yield_stress / (2 * mu)
        if delta_gamma > 0:  # Yields
            epsilon -= 1 * (delta_gamma / epsilon_hat_norm) * epsilon_hat
            sig = self.make_matrix_from_diag(ti.exp(epsilon))
            F = U @ sig @ V.transpose()
        return F

    @ti.kernel
    def p2g(self):
        for i, j in self.grid_m:
            self.grid_v[i, j] = [0, 0]
            self.grid_m[i, j] = 0

        for p in self.x:  # Particle to grid (P2G)
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
                if self.material[p] == 0:
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
                )
                if self.material[p] == 0:
                    cauchy += self.F[p] @ act @ self.F[p].transpose()
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
    
    @ti.kernel
    def g2p(self):
        for p in self.x:  # grid to particle (G2P)
            if (
                self.x[p][0] - self.anchor[None][0] > 1e-5
                and self.x[p][0] - self.anchor[None][0] < 1.0 - 1e-5
            ):
                base = ((self.x[p] - self.anchor[None]) * self.inv_dx - 0.5).cast(int)
                fx = (self.x[p] - self.anchor[None]) * self.inv_dx - base.cast(float)
                w = [
                    0.5 * (1.5 - fx) ** 2,
                    0.75 - (fx - 1.0) ** 2,
                    0.5 * (fx - 0.5) ** 2,
                ]
                new_v = ti.Vector.zero(float, 2)
                new_C = ti.Matrix.zero(float, 2, 2)
                for i, j in ti.static(ti.ndrange(3, 3)):
                    dpos = ti.Vector([i, j]).cast(float) - fx
                    g_v = self.grid_v[base + ti.Vector([i, j])]
                    weight = w[i][0] * w[j][1]
                    new_v += weight * g_v
                    new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
                self.v[p], self.C[p] = new_v, new_C
                self.x[p] += self.dt * self.v[p]
                
    def reset(self):
        NotImplementedError

    def step(self, action):
        NotImplementedError

    def render(self, gui, log=False, record_id=None):
        NotImplementedError

    @ti.kernel
    def grid_operation(self):
        NotImplementedError
