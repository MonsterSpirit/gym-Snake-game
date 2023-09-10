from gymnasium.envs.registration import register
from tcs.envs.core import TcsV2Env, ConstructEnv

register(
    id="tcs-v2", entry_point="tcs.envs.core:TcsV2Env", max_episode_steps=2000,
                     kwargs={}
)
