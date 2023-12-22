import os
import cv2
import gym
import numpy as np
import taichi as ti


@ti.data_oriented
class DIG(gym.Env):
    def __init__(
        self,
        quality=1,
        n_grid=128,
        mu_0=8000,
        lambda_0=1000,
        yield_stress=65,
        max_actuation=50,
        action_dim=2*8**2,
        action_res_resize=4,
        obs_res=64,
        obs_res_resize=8,
        repeat_times=100,
        reward_params=[20, 5, 1, 0.0002, 15],
        visualize=False,
        save_file_name=None,
    ):
        print("*******************Morphological Maze DIG-v0*******************")
        self.reward_params = reward_params
        self.frames_num = 0
        self.visualize = visualize
        self.save_file_name = save_file_name   
        current_file_path = os.path.abspath(__file__)
        self.current_directory = os.path.dirname(current_file_path)

        # mpm params
        self.n_grid = n_grid * quality
        self.dx = 1 / self.n_grid
        self.inv_dx = float(self.n_grid)
        self.quality = quality
        self.x_list = []
        self.material_list = []
        self.bound = 1
        self.coeff = 0.5
        self.dt = 1e-4 / self.quality
        self.p_vol = 1
        self.mu_0 = mu_0
        self.lambda_0 = lambda_0
        self.yield_stress = yield_stress
        self.repeat_times = repeat_times

        # initial robot
        self.offset_x = -0.15
        self.offset_y = 0.0
        self.dust_particles_num = 8000
        self.add_voxel(-0.05, 0.5, 0.14, 0.14) 
        self.add_voxel(-0.25, 0.0, 0.5, 0.33, is_dust=True)
        self.target = ti.Vector.field(2, dtype=float, shape=())
        self.target[None] = [0.2, 0.17]

        # mpm variables
        self.n_particles = len(self.x_list) * self.quality**2
        self.robot_particles_num = self.n_particles - self.dust_particles_num
        self.p_mass = ti.field(dtype=float, shape=self.n_particles)
        self.x = ti.Vector.field(2, dtype=float, shape=self.n_particles)
        self.x_save = ti.Vector.field(2, dtype=float, shape=self.n_particles)
        self.material_save = ti.field(dtype=int, shape=self.n_particles)
        for i in range(len(self.x_list)):
            self.x_save[i] = self.x_list[i]
            self.material_save[i] = self.material_list[i]
        self.v = ti.Vector.field(2, dtype=float, shape=self.n_particles)
        self.C = ti.Matrix.field(2, 2, dtype=float, shape=self.n_particles)
        self.F = ti.Matrix.field(2, 2, dtype=float, shape=self.n_particles)
        self.U = ti.Matrix.field(2, 2, dtype=float, shape=self.n_particles)
        self.V = ti.Matrix.field(2, 2, dtype=float, shape=self.n_particles,)
        self.sig = ti.Matrix.field(2, 2, dtype=float, shape=self.n_particles)
        self.material = ti.field(dtype=int, shape=self.n_particles)
        self.Jp = ti.field(dtype=float, shape=self.n_particles)
        self.grid_v = ti.Vector.field(2, dtype=float, shape=(self.n_grid, self.n_grid))
        self.grid_m = ti.field(dtype=float, shape=(self.n_grid, self.n_grid))
        self.gravity = ti.Vector.field(2, dtype=float, shape=())
        self.gravity[None] = [0, -19]

        # action space
        self.max_actuation = max_actuation
        self.action_res_resize = action_res_resize
        self.action = np.zeros(action_dim)
        self.grid_actuation = ti.Vector.field(2, dtype=float, shape=(self.n_grid, self.n_grid))
        self.particle_actuation = ti.Vector.field(2, dtype=float, shape=self.n_particles)
        self.action_space = gym.spaces.box.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
        print("n_particles: ", self.n_particles, "action_space: ", self.action_space.shape)

        # observation space
        self.obs_res = obs_res
        self.obs_res_resize = obs_res_resize
        self.obs_real_size = int(np.sqrt(action_dim / 2)) * action_res_resize
        self.shape_field = ti.field(dtype=float, shape=(self.obs_res_resize * self.n_grid, self.obs_res_resize * self.n_grid))
        self.vx_field = ti.field(dtype=float, shape=(self.obs_res_resize * self.n_grid, self.obs_res_resize * self.n_grid))
        self.vy_field = ti.field(dtype=float, shape=(self.obs_res_resize * self.n_grid, self.obs_res_resize * self.n_grid))
        self.center_point = [
            np.mean(self.x_save.to_numpy()[:self.robot_particles_num, 0]),
            np.mean(self.x_save.to_numpy()[:self.robot_particles_num, 1]),
        ]
        self.anchor = ti.Vector.field(2, dtype=float, shape=())
        self.anchor[None] = [-0.5, 0.0]
        self.reset_()
        self.set_observation_field()
        state = self.shape_field.to_numpy().reshape(1, self.obs_res_resize * self.n_grid, self.obs_res_resize * self.n_grid)
        vx = self.vx_field.to_numpy().reshape(1, self.obs_res_resize * self.n_grid, self.obs_res_resize * self.n_grid)
        vy = self.vy_field.to_numpy().reshape(1, self.obs_res_resize * self.n_grid, self.obs_res_resize * self.n_grid)
        state = 255 * self.robot_state(state).squeeze(0)
        state = (np.clip(cv2.resize(state, (self.obs_res, self.obs_res), interpolation=cv2.INTER_AREA), a_min=0, a_max=255)
                .astype(np.uint8)
                .reshape(1, self.obs_res, self.obs_res))
        vx = (255 * (self.robot_state(vx) + 10) / 20).squeeze(0)
        vx = cv2.resize(
            vx,
            (self.obs_res, self.obs_res),
            interpolation=cv2.INTER_AREA,
        ).reshape(1, self.obs_res, self.obs_res)
        vx = np.clip(10 * (vx - 127) + 127, a_min=0, a_max=255).astype(np.uint8)
        vy = (255 * (self.robot_state(vy) + 10) / 20).squeeze(0)
        vy = cv2.resize(
            vy,
            (self.obs_res, self.obs_res),
            interpolation=cv2.INTER_AREA,
        ).reshape(1, self.obs_res, self.obs_res)
        vy = np.clip(10 * (vy - 127) + 127, a_min=0, a_max=255).astype(np.uint8)
        state = np.concatenate([state, vx, vy], axis=0)
        self.state = state
        low = np.zeros(
            (self.state.shape[0], self.obs_res, self.obs_res),
            dtype=np.uint8,
        )
        high = np.full(
            (self.state.shape[0], self.obs_res, self.obs_res),
            255,
            dtype=np.uint8,
        )

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.uint8)
        self.prev_location = None
        self.gui = None

    def reset(self):
        self.reset_()
        return self.state

    def step(self, action):
        self.update_grid_actuation(action)
        for i in range(self.repeat_times):
            self.update_particle_actuation()
            self.substep()
            if self.visualize and i == 0:
                self.render(self.gui, log=True)
        # state (relative x, y)
        x_numpy = self.x.to_numpy()
        self.anchor[None] = [-0.5, 0.0]
        self.center_point = np.mean(x_numpy[:self.robot_particles_num], axis=0)
        self.set_observation_field()
        state = self.shape_field.to_numpy().reshape(1, self.obs_res_resize * self.n_grid, self.obs_res_resize * self.n_grid)
        new_state = 255 * self.robot_state(state).squeeze(0)
        new_state = (np.clip(cv2.resize(new_state, (self.obs_res, self.obs_res), interpolation=cv2.INTER_AREA), a_min=0, a_max=255)
                .astype(np.uint8)
                .reshape(1, self.obs_res, self.obs_res))
        new_vx = self.vx_field.to_numpy().reshape(1, self.obs_res_resize * self.n_grid, self.obs_res_resize * self.n_grid)
        new_vx = (255 * (self.robot_state(new_vx) + 10) / 20).squeeze(0)
        new_vx = cv2.resize(
            new_vx,
            (self.obs_res, self.obs_res),
            interpolation=cv2.INTER_AREA,
        ).reshape(1, self.obs_res, self.obs_res)
        new_vx = np.clip(10 * (new_vx - 127) + 127, a_min=0, a_max=255).astype(np.uint8)
        new_vy = self.vy_field.to_numpy().reshape(1, self.obs_res_resize * self.n_grid, self.obs_res_resize * self.n_grid)
        new_vy = (255 * (self.robot_state(new_vy) + 10) / 20).squeeze(0)
        new_vy = cv2.resize(
            new_vy,
            (self.obs_res, self.obs_res),
            interpolation=cv2.INTER_AREA,
        ).reshape(1, self.obs_res, self.obs_res)
        new_vy = np.clip(10 * (new_vy - 127) + 127, a_min=0, a_max=255).astype(np.uint8)
        new_state = np.concatenate([new_state, new_vx, new_vy], axis=0)
        self.state = new_state
        # if not os.path.exists("./observation"):
        #     os.makedirs("./observation")
        # cv2.imwrite("./observation/state.png", self.state[0])
        # cv2.imwrite("./observation/vx.png", self.state[1])
        # cv2.imwrite("./observation/vy.png", self.state[2])
        if not np.isnan(self.center_point).any():
            curr_location = self.center_point
            self.prev_location = self.center_point
        else:
            curr_location = self.prev_location
        terminated = False
        reward = 0
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
        print(reward)
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

    def add_voxel(self, x, y, w, h, is_dust=False):
        if not is_dust:
            for _ in range(12000):
                randm_x = np.random.rand()
                randm_y = np.random.rand()
                sign_x = 1 if np.random.rand() > 0.5 else -1
                sign_y = 1 if np.random.rand() > 0.5 else -1
                y_length = np.sqrt((0.5 * w)**2 - (0.5 * w * randm_x)**2)
                self.x_list.append(
                    [
                        x + 0.5 * (1 + sign_x * randm_x) * w + self.offset_x,
                        y + 0.5  * h + sign_y * randm_y * y_length + self.offset_y
                    ]
                )
                self.material_list.append(0)
        else:
            for _ in range(self.dust_particles_num):
                self.x_list.append(
                    [
                        x + np.random.rand() * w + 0.0,
                        y + np.random.rand() * h + 0.16,
                    ]
                )
                self.material_list.append(1)

    def robot_state(self, state):
        mid_index = [
            int((0.0 - self.anchor[None][0]) * self.inv_dx * self.obs_res_resize),
            int((0.4 - self.anchor[None][1]) * self.inv_dx * self.obs_res_resize),
        ]
        up_index = mid_index[1] + int(0.5 * self.obs_real_size * self.obs_res_resize)
        down_index = mid_index[1] - int(0.5 * self.obs_real_size * self.obs_res_resize)
        left_index = mid_index[0] - int(0.5 * self.obs_real_size * self.obs_res_resize)
        right_index = mid_index[0] + int(0.5 * self.obs_real_size * self.obs_res_resize)
        if up_index > self.obs_res_resize * self.n_grid:
            up_index = self.obs_res_resize * self.n_grid
            down_index = up_index - int(1.0 * self.obs_real_size * self.obs_res_resize)
        if down_index < 0:
            down_index = 0
            up_index = down_index + int(1.0 * self.obs_real_size * self.obs_res_resize)
        if left_index < 0:
            left_index = 0
            right_index = left_index + int(1.0 * self.obs_real_size * self.obs_res_resize)
        if right_index > self.obs_res_resize * self.n_grid:
            right_index = self.obs_res_resize * self.n_grid
            left_index = right_index - int(1.0 * self.obs_real_size * self.obs_res_resize)
        state = state[
            :,
            self.n_grid * self.obs_res_resize - up_index : self.n_grid * self.obs_res_resize - down_index,
            left_index:right_index,
        ]
        return state

    def update_grid_actuation(self, action):
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
            base_index = np.array(
                [
                    int((self.center_point[0] - self.anchor[None][0]) * self.inv_dx - 0.5 * final_action_res),
                    int((self.center_point[1] - self.anchor[None][1]) * self.inv_dx - 0.5 * final_action_res),
                ]
            )
        else:
            base_index = np.array(
                [
                    int((self.prev_location[0] - self.anchor[None][0]) * self.inv_dx - 0.5 * final_action_res),
                    int((self.prev_location[1] - self.anchor[None][1]) * self.inv_dx - 0.5 * final_action_res),
                ]
            )
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
    def set_observation_field(self):
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
    def reset_(self):
        for i in range(self.n_particles):
            self.x[i] = self.x_save[i]
            self.v[i] = [0, 0]
            self.material[i] = self.material_save[i]
            if self.material[i] == 0:
                self.p_mass[i] = 2
            else:
                self.p_mass[i] = 2
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
    def substep(self):
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


if __name__ == "__main__":
    ti.init(arch=ti.gpu)
    gui = ti.GUI("Taichi MPM Morphological Maze", res=512, background_color=0x112F41, show_gui=False)
    env = DIG(visualize=True, save_file_name="test")
    env.reset()
    env.render(gui, log=True, record_id=0)
    while True:
        env.step(2 * np.random.rand(env.action_space.shape[0]) - 1)
