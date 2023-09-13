from tcs import *
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3 import PPO
import torch 

env = make_atari_env("ALE/DemonAttack-v5", n_envs=1)
print(env.lives())
envs = make_atari_env("tcs-v2", env_kwargs={
        "height": 36,
        "width": 36,
    }, n_envs=32)
print(envs)
# envs = ConstructEnv().getEnv(height=36, width=36)

# model = PPO('CnnPolicy', envs,
#             verbose=1,
#             learning_rate=0.00025,
#             gamma=0.99,  # 折扣因子 作用：当折扣因子（gamma）接近1时，在强化学习中，代理会更加重视未来的奖励，即对长期回报的考虑更为重要。
#             gae_lambda=0.95,  # GAE的λ参数 作用：当GAE的λ参数（gae_lambda）接近1时，在强化学习中，对于估计优势函数和计算GAE（Generalized Advantage Estimation）的累积奖励时，更加注重长期的回报。
#             ent_coef=0.01,  # 熵系数 作用：当熵系数（ent_coef）接近1时，在PPO算法中，策略网络更加随机，具有更高的探索性。
#             clip_range=0.1,  # 较小的剪切范围可以使策略更新更加保守，从而更加重视当前分数。
#             vf_coef=0.5,  # 设置值函数系数，值越大函数在总体损失中的影响更大，从而更加重视分数。
#             n_epochs=4,  # 训练时运行的回合数 作用：较大的回合数可以使策略网络更新更加充分，从而更加重视长期回报。
#             batch_size=256,  # 每个训练批次的大小 作用：较大的批次大小可以使训练更加稳定，从而更加重视长期回报。
#             n_steps=128,  # 每个回合的步数 作用：较大的步数可以使策略网络更新更加充分，从而更加重视长期回报。
#             tensorboard_log="./logs/")


# model.learn(total_timesteps=2000)
