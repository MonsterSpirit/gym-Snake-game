# 贪吃色环境

## 安装

```bash
# linux
bash build.sh
# window
./build.bat
```

## 使用
```python
envs = make_vec_env("tcs-v2", env_kwargs={
        "height": 8, # 高
        "width": 8,  # 宽
        "render_mode": "human", # 是否渲染
        "mp4": True, # 是否录制 mp4
        "image_size": 15, # 训练图片，单个方框大小
    }, n_envs=1,
        seed=1, # 随机食物种子
    )
```

<video controls width="400" height="300">
  <source src="./demo.mp4" type="video/mp4">
  <p>Your browser does not support the video tag.</p>
</video>
