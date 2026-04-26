import gym
import numpy as np

class RobosuiteGymAdapter(gym.Env):
    """Wraps a robosuite OffScreenRenderEnv to look like OpenAI‑Gym."""
    def __init__(self, rs_env):
        self.rs_env = rs_env


        # obs = self.reset()                      # sample to build spaces
        obs, info = self.reset()  # Make sure to unpack correctly



        def make_space(v):
            if v.dtype == np.uint8:          # image: 0‑255
                return gym.spaces.Box(low=0, high=255, shape=v.shape, dtype=np.uint8)
            else:                            # proprio, states: unbounded
                return gym.spaces.Box(low=-np.inf, high=np.inf, shape=v.shape, dtype=v.dtype)

        self.observation_space = gym.spaces.Dict({k: make_space(v) for k, v in obs.items()})


        # self.observation_space = gym.spaces.Dict({
        #     k: gym.spaces.Box(low=-np.inf, high=np.inf, shape=v.shape, dtype=v.dtype)
        #     for k, v in obs.items()
        # })






        try:
            self.action_space = self.rs_env.action_space
        except AttributeError:
            try:
                act_dim = self.rs_env.action_dim
            except AttributeError:
                try:
                    low, high = self.rs_env.action_spec
                    self.action_space = gym.spaces.Box(low, high, dtype=np.float32)
                except AttributeError:
                    # Fallback: derive from MuJoCo model (always present)
                    act_dim = int(self.rs_env.sim.model.nu)    # number of actuators
            if 'action_space' not in self.__dict__:
                self.action_space = gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
                )
                

        # # ----- action space -----
        # try:                                # preferred: use existing Gym space
        #     self.action_space = self.rs_env.action_space
        # except AttributeError:              # rare older builds
        #     try:
        #         act_dim = self.rs_env.action_dim
        #         self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)
        #     except AttributeError:
        #         low, high = self.rs_env.action_spec
        #         self.action_space = gym.spaces.Box(low, high, dtype=np.float32)


        # try:
        #     act_dim = self.rs_env.action_dim                 # old robosuite
        #     low, high = -np.ones(act_dim),  np.ones(act_dim)
        # except AttributeError:
        #     low, high = self.rs_env.action_spec              # new robosuite
        #     act_dim   = low.size
        # # MuJoCo actions are already normalized [-1, 1]
        # self.action_space = gym.spaces.Box(
        #         low=low.astype(np.float32),
        #         high=high.astype(np.float32),
        #         shape=(act_dim,),
        #         dtype=np.float32)


        # act_dim = self.rs_env.action_dim
        # self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)



    def reset(self, **kwargs):
        obs = self.rs_env.reset(**kwargs)
        # print("Observation keys returned from rs_env.reset():", obs.keys())
        return obs, {} #observation, info tuple


    # def reset(self, **kwargs):
    #     return self.rs_env.reset(**kwargs)



    def step(self, action):
        obs, reward, done, info = self.rs_env.step(action)
        # robosuite returns (obs, reward, done, info); Gym 0.26 adds trunc flag – keep False
        return obs, reward, done, False, info

    def render(self, mode="rgb_array"):
        return self.rs_env.render(mode)



    def get_task(self):
        """Returns the language instruction task from the underlying rs_env."""
        if hasattr(self.rs_env, "language_instruction"):
            return {"language_instruction": self.rs_env.language_instruction}
        else:
            raise AttributeError("Underlying rs_env has no 'language_instruction' attribute.")
