"""
Simple wrapper for registering metaworld enviornments
properly with gym.
"""
import gym
import numpy as np


def trim_mw_obs(obs):
    # Remove the double robot observation from the environment.
    # Only stack two object observations
    # this helps keep everything more markovian
    return np.concatenate((obs[:18], obs[22:]), dtype=np.float32)


class MetaWorldSawyerEnv(gym.Env):
    def __init__(self, env_name, seed=False, randomize_hand=True, sparse: bool = False, horizon: int = 250):
        from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

        self.env_name = env_name
        self._env = ALL_V2_ENVIRONMENTS[env_name]()
        self._env._freeze_rand_vec = False
        self._env._partially_observable = False
        self._env._set_task_called = True
        self._seed = seed
        if self._seed:
            self._env.seed(0)  # Seed it at zero for now.
        self.randomize_hand = randomize_hand
        self.sparse = sparse
        assert self._env.observation_space.shape[0] == 39
        low, high = self._env.observation_space.low, self._env.observation_space.high
        self.observation_space = gym.spaces.Box(low=trim_mw_obs(low), high=trim_mw_obs(high), dtype=np.float32)
        self.action_space = self._env.action_space
        self._max_episode_steps = min(horizon, self._env.max_path_length)

    def seed(self, seed=None):
        super().seed(seed=seed)
        if self._seed:
            self._env.seed(0)

    def step(self, action):
        self._episode_steps += 1
        obs, reward, done, info = self._env.step(action)
        if self._episode_steps == self._max_episode_steps:
            done = True
            info["discount"] = 1.0  # Ensure infinite boostrap.
        # Remove history from the observations. It makes it too hard to reset.
        if self.sparse:
            reward = float(info["success"])  # Reward is now if we succeed or fail.
        else:
            reward = reward / 10
        return trim_mw_obs(obs.astype(np.float32)), reward, done, info

    def _get_obs(self):
        return trim_mw_obs(self._env._get_obs())

    def get_state(self):
        joint_state, mocap_state = self._env.get_env_state()
        qpos, qvel = joint_state.qpos, joint_state.qvel
        mocap_pos, mocap_quat = mocap_state
        self._split_shapes = np.cumsum(
            np.array([qpos.shape[0], qvel.shape[0], mocap_pos.shape[1], mocap_quat.shape[1]])
        )
        return np.concatenate([qpos, qvel, mocap_pos[0], mocap_quat[0], self._env._last_rand_vec], axis=0)

    def set_state(self, state):
        joint_state = self._env.sim.get_state()
        if not hasattr(self, "_split_shapes"):
            self.get_state()  # Load the split
        qpos, qvel, mocap_pos, mocap_quat, rand_vec = np.split(state, self._split_shapes, axis=0)
        if not np.all(self._env._last_rand_vec == rand_vec):
            # We need to set the rand vec and then reset
            self._env._freeze_rand_vec = True
            self._env._last_rand_vec = rand_vec
            self._env.reset()
        joint_state.qpos[:] = qpos
        joint_state.qvel[:] = qvel
        self._env.set_env_state((joint_state, (np.expand_dims(mocap_pos, axis=0), np.expand_dims(mocap_quat, axis=0))))
        self._env.sim.forward()

    def reset(self, **kwargs):
        self._episode_steps = 0
        self._env.reset(**kwargs).astype(np.float32)
        if self.randomize_hand:
            # Hand init pos is usually set to self.init_hand_pos
            # We will add some uniform noise to it.
            high = np.array([0.25, 0.15, 0.2], dtype=np.float32)
            hand_init_pos = self.hand_init_pos + np.random.uniform(low=-high, high=high)
            hand_init_pos = np.clip(hand_init_pos, a_min=self._env.mocap_low, a_max=self._env.mocap_high)
            hand_init_pos = np.expand_dims(hand_init_pos, axis=0)
            for _ in range(50):
                self._env.data.set_mocap_pos("mocap", hand_init_pos)
                self._env.data.set_mocap_quat("mocap", np.array([1, 0, 1, 0]))
                self._env.do_simulation([-1, 1], self._env.frame_skip)

        # Get the obs once to reset history.
        self._get_obs()
        return self._get_obs().astype(np.float32)

    def render(self, mode="rgb_array", camera_name="corner2", width=640, height=480):
        assert mode == "rgb_array", "Only RGB array is supported"
        # stack multiple views
        for ctx in self._env.sim.render_contexts:
            ctx.opengl_context.make_context_current()
        return self._env.render(offscreen=True, camera_name=camera_name, resolution=(width, height))

    def __getattr__(self, name):
        return getattr(self._env, name)


class MetaWorldSawyerImageWrapper(gym.Wrapper):
    def __init__(self, env, width=64, height=64, camera="corner2", show_goal=False):
        assert isinstance(
            env.unwrapped, MetaWorldSawyerEnv
        ), "MetaWorld Wrapper must be used with a MetaWorldSawyerEnv class"
        super().__init__(env)
        self._width = width
        self._height = height
        self._camera = camera
        self._show_goal = show_goal
        shape = (3, self._height, self._width)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

    def _get_image(self):
        if not self._show_goal:
            try:
                self.env.unwrapped._set_pos_site("goal", np.inf * self.env.unwrapped._target_pos)
            except ValueError:
                pass  # If we don't have the goal site, just continue.
        img = self.env.render(mode="rgb_array", camera_name=self._camera, width=self._width, height=self._height)
        return img.transpose(2, 0, 1)

    def step(self, action):
        state_obs, reward, done, info = self.env.step(action)
        # Throw away the state-based observation.
        info["state"] = state_obs
        return self._get_image().copy(), reward, done, info

    def reset(self):
        # Zoom in camera corner2 to make it better for control
        # I found this view to work well across a lot of the tasks.
        camera_name = "corner2"
        # Original XYZ is 1.3 -0.2 1.1
        index = self.model.camera_name2id(camera_name)
        self.model.cam_fovy[index] = 20.0  # FOV
        self.model.cam_pos[index][0] = 1.5  # X
        self.model.cam_pos[index][1] = -0.35  # Y
        self.model.cam_pos[index][2] = 1.1  # Z

        self.env.reset()
        return self._get_image().copy()  # Return the image observation


def get_mw_image_env(env_name, **kwargs):
    env = MetaWorldSawyerEnv(env_name, **kwargs)
    return MetaWorldSawyerImageWrapper(env)
