from typing import Any, SupportsFloat
import pygame
import random
import numpy as np
import time
import keyboard
import copy
import numpy
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from gymnasium.wrappers import TimeLimit
import cv2


class SnakeEnv:

    def __init__(self, height, width) -> None:
        self.height = height
        self.width = width
        self.observationSpace = [
            [0 for _ in range(self.width)] for _ in range(self.height)]
        pass

    def getValue(self, y, x):
        return self.observationSpace[y][x]

    def setValue(self, y, x, value):
        if self.observationSpace[y][x] != value:
            self.observationSpace[y][x] = value

    def setNull(self, y, x):
        self.observationSpace[y][x] = 0

    def print(self):
        print()
        for y in range(len(self.observationSpace)):
            print(self.observationSpace[y])

    def getDesignationValue(self, value):
        result = []
        for y in range(len(self.observationSpace)):
            for x in range(len(self.observationSpace[y])):
                if self.observationSpace[y][x] == value:
                    result.append((y, x))
        return result

    def getEnvironmentalData(self):
        return self.observationSpace


class Wall:

    def __init__(self, snakeEnv: SnakeEnv) -> None:
        self.value = 1
        self.snakeEnv = snakeEnv
        self.walls = []
        self.generate()
        pass

    def generate(self):
        snakeEnv = self.snakeEnv
        for y in range(snakeEnv.height):
            for x in range(snakeEnv.width):
                if x == 0 or x == snakeEnv.width - 1 or \
                        y == 0 or y == snakeEnv.height - 1:
                    snakeEnv.setValue(y, x, self.value)
                    self.walls.append((y, x))

    def isWall(self, y, x):
        for i in range(len(self.walls)):
            vy, vx = self.walls[i]
            if vy == y and vx == x:
                return True
        return False


class Food:

    def __init__(self, snakeEnv: SnakeEnv, seed=None) -> None:
        self.value = 2
        self.snakeEnv = snakeEnv
        if seed == None:
            seed = int(time.time())
        random.seed(seed)
        self.foodCoordinates = ()
        self.generate()
        pass

    def generate(self, remove=False):
        if remove and len(self.foodCoordinates) == 2:
            y, x = self.foodCoordinates
            self.snakeEnv.setNull(y, x)
        emptyPosition = self.snakeEnv.getDesignationValue(0)
        lenEmptyPosition = len(emptyPosition)
        if lenEmptyPosition <= 0:
            return True
        if lenEmptyPosition == 1:
            y, x = emptyPosition[0]
        else:
            ri = random.randint(0, len(emptyPosition) - 1)
            y, x = emptyPosition[ri]
        self.foodCoordinates = (y, x)
        self.snakeEnv.setValue(y, x, self.value)
        return False

    def getFoodCoordinates(self):
        return self.foodCoordinates

    def isFood(self, y, x):
        vy, vx = self.foodCoordinates
        return vy == y and vx == x

    def rendering(self):
        if len(self.foodCoordinates) == 2:
            y, x = self.foodCoordinates
            self.snakeEnv.setValue(y, x, self.value)


class SnakeBody:

    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

    def __init__(self, snakeEnv: SnakeEnv) -> None:
        self.snakeEnv = snakeEnv
        self.head_value = 3
        self.body_value = 4
        # 长度
        self.len = 1
        # 移动方向
        self.movingDirection = 1
        self.body = [(int(snakeEnv.height / 2), int(snakeEnv.width / 2))]
        self.end = ()
        self.generate()
        pass

    def getBody(self):
        return self.body

    def getHead(self):
        return self.body[0]

    def length(self):
        return len(self.body)

    def setAction(self, action):
        if action == 0:
            return
        if (action % 2 == 0 and action - 1 == self.movingDirection) or \
                (action % 1 == 0 and action + 1 == self.movingDirection):
            return
        self.movingDirection = action

    def lengthen(self):
        if len(self.end) == 2:
            y, x = self.end
            self.body.append((y, x))
            self.snakeEnv.setValue(y, x, self.body_value)
            self.end = ()

    def move(self):
        y, x = self.body[0]
        if self.movingDirection == SnakeBody.UP:
            y -= 1
        elif self.movingDirection == SnakeBody.DOWN:
            y += 1
        elif self.movingDirection == SnakeBody.LEFT:
            x -= 1
        elif self.movingDirection == SnakeBody.RIGHT:
            x += 1
        else:
            raise RuntimeError(f"异常 move : {self.movingDirection}")
        self.body.insert(0, (y, x))
        ey, ex = self.body.pop()
        self.end = (ey, ex)
        self.snakeEnv.setNull(ey, ex)
        self.generate()
        return (y, x)

    def headBodyStack(self):
        hy, hx = self.body[0]
        for i in range(1, len(self.body)):
            y, x = self.body[i]
            if hy == y and hx == x:
                return True
        return False

    def isBody(self, hy, hx):
        for i in range(1, len(self.body)):
            y, x = self.body[i]
            if hy == y and hx == x:
                return True
        return False

    def generate(self):
        y, x = self.body[0]
        self.snakeEnv.setValue(y, x, self.head_value)
        for i in range(1, len(self.body)):
            y, x = self.body[i]
            self.snakeEnv.setValue(y, x, self.body_value)


class TcsV2Env(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, **kwargs) -> None:
        self.image_size = kwargs.get("image_size", 10)
        self.show = False
        self.height = kwargs.get("height", 50)
        self.width = kwargs.get("width", 50)
        self.windowcolor = [(0, 0, 0), (255, 255, 255), (255, 0, 0),
                            (0, 0, 255), None]
        self.WINDOW_WIDTH = self.width * self.image_size
        self.WINDOW_HEIGHT = self.height * self.image_size
        self.WINDOW_BLACK = (0, 0, 0)
        self.obs_type = kwargs.get("obs_type", "rgb")
        self.outputType = kwargs.get("outputType", "Discrete")
        if self.outputType == "Discrete":
            self.action_space = gym.spaces.Discrete(5)
        elif self.outputType == "Box":
            self.action_space = gym.spaces.Box(
                low=0, high=4, shape=(1,), dtype=np.uint8
            )
        if self.obs_type == "rgb":
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(self.height * self.image_size, self.width * self.image_size, 3), dtype=np.uint8)
        if self.obs_type == "data":
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(9,), dtype=np.uint8)
        self.render_mode = kwargs.get("render_mode", None)
        self.mp4 = kwargs.get("mp4", False)
        self.mp4fps = kwargs.get("mp4fps", 30)
        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]
        super().__init__()

    def get_action_meanings(self):
        return ["NOOP", "UP", "DOWN", "LEFT", "RIGHT"]

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        if isinstance(action, np.ndarray):
            action = action[0]
            action = round(action)
        info = {}
        reward = 0
        self.snakebody.setAction(action)
        y, x = self.snakebody.move()
        done = False
        if self.snakebody.isBody(y, x):
            done = True
            info["message"] = "碰到身体"
        if self.wall.isWall(y, x):
            done = True
            info["message"] = "碰撞到墙壁"
        if self.food.isFood(y, x):
            self.snakebody.lengthen()
            if self.food.generate():
                done = True
                vn_step = 1000 - self.num_step
                if vn_step < 0:
                    vn_step = 0
                reward += 100 * (vn_step / 1000)
                info["message"] = "神TM吃满屏了"
                if self.show:
                    time.sleep(3)
            reward += 1
            # self.distanceFood = 20
        # else:
        #     fy, fx = self.food.getFoodCoordinates()
        #     jy = abs(fy - y)
        #     jx = abs(fx - x)
        #     distance = jy + jx
        #     if distance < self.distanceFood:
        #         self.distanceFood = distance
        #         reward += 1 - ((1 / 20) * distance)
        if reward < 0:
            reward = 0
        self.score += reward
        observation = self.environmentalData()
        self.num_step += 1
        return observation, reward, done, False, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self.snakeEnv = SnakeEnv(self.height, self.width)
        self.wall = Wall(self.snakeEnv)
        self.snakebody = SnakeBody(self.snakeEnv)
        self.food = Food(self.snakeEnv, seed=seed)
        observation = self.environmentalData()
        self.score = 0
        self.num_step = 0
        # self.distanceFood = 20
        return observation, {}

    def environmentalData(self):
        if self.obs_type == "rgb":
            observation = self.render()
            return observation
        elif self.obs_type == "data":
            hy, hx = self.snakebody.getHead()
            fy, fx = self.food.getFoodCoordinates()
            hl = self.snakebody.length()
            observation = [hy, hx, fy, fx, hl]
            return observation
        return {}

    def render(self):
        if (self.render_mode == "human") and ("window" not in self.__dict__ or self.window == None):
            pygame.init()
            self.window = pygame.display.set_mode(
                (self.width * 10, self.height * 10))
            pygame.display.set_caption("tcs")
            self.window.fill(self.WINDOW_BLACK)
            self.show = True
        # 创建一个画板
        if "canvas" not in self.__dict__ or self.canvas == None:
            self.canvas = pygame.Surface(
                (self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        if self.show and ("canvasN" not in self.__dict__ or self.canvasN == None):
            self.canvasN = pygame.Surface(
                (self.width * 10, self.height * 10)
            )
        self.canvas.fill(self.WINDOW_BLACK)
        if self.show:
            self.canvasN.fill(self.WINDOW_BLACK)

        observation = self.snakeEnv.getEnvironmentalData()
        for y in range(len(observation)):
            for x in range(len(observation[y])):
                value = observation[y][x]
                if value == 0 or self.windowcolor[value] == None:
                    continue
                pygame.draw.rect(
                    self.canvas, self.windowcolor[value], (x * self.image_size, y * self.image_size, self.image_size, self.image_size))
                if self.show:
                    pygame.draw.rect(
                        self.canvasN, self.windowcolor[value], (
                            x * 10, y * 10, 10, 10)
                    )
        body = self.snakebody.getBody()
        bodylen = len(body)
        if bodylen > 1:
            colosNm = round(105 / (bodylen - 1))
            dcolosNm = 150 + colosNm
            for i in range(bodylen - 2, 0, -1):
                y, x = body[i]
                vvColos = (0, dcolosNm, 0)
                pygame.draw.rect(
                    self.canvas, vvColos, (x * self.image_size, y * self.image_size, self.image_size, self.image_size))
                if self.show:
                    pygame.draw.rect(
                        self.canvasN, vvColos, (
                            x * 10, y * 10, 10, 10)
                    )
                if dcolosNm >= 255:
                    continue
                dcolosNm += colosNm
                if dcolosNm > 255:
                    dcolosNm = 255
            y, x = body[-1]
            vbcolos = (232, 255, 1)
            pygame.draw.rect(
                self.canvas, vbcolos, (x * self.image_size, y * self.image_size, self.image_size, self.image_size))
            if self.show:
                pygame.draw.rect(
                    self.canvasN, vbcolos, (
                        x * 10, y * 10, 10, 10)
                )

        result = np.transpose(
            np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
        )
        if self.render_mode == "human":
            show_image = np.transpose(
                np.array(pygame.surfarray.pixels3d(self.canvasN)), axes=(1, 0, 2)
            )
            self.window.blit(self.canvasN, self.canvasN.get_rect())
            # 清理事件
            pygame.event.pump()
            pygame.display.update()
            # pygame.display.flip()
            # event = pygame.event.poll()
            # # pygame.display.update()
            # if event.type == pygame.QUIT:
            #     pygame.quit()
            #     exit()
            if self.mp4:
                if "outMp4" not in self.__dict__ or self.outMp4 == None:
                    height, width, _ = show_image.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.outMp4 = cv2.VideoWriter(
                        "1.mp4", fourcc, self.mp4fps, (width, height))
                self.outMp4.write(show_image)

        return result

    def close(self):
        del self.snakebody
        del self.wall
        del self.food
        del self.snakeEnv
        del self.score
        if self.mp4:
            self.outMp4.release()
        return super().close()


class ConstructEnv:

    def __init__(self) -> None:
        gym.register(id="tcs-v2", entry_point=TcsV2Env, max_episode_steps=2000,
                     kwargs={})
        pass

    def getDummyVecEnvs(self, num=32, **kwargs) -> DummyVecEnv:
        return DummyVecEnv([lambda: self.getEnv(**kwargs) for _ in range(num)])

    def getSudoVecEnvs(self, num=6, **kwargs) -> SubprocVecEnv:
        return SubprocVecEnv([lambda: self.getEnv(**kwargs) for _ in range(num)])

    def getEnv(self, **kwargs) -> TimeLimit:
        return TimeLimit(gym.make("tcs-v2", **kwargs), max_episode_steps=5000)


if __name__ == "__main__":
    env = ConstructEnv().getEnv(render_mode="human", height=3, width=3)
    env.reset()
    clock = pygame.time.Clock()
    action = 0

    def on_press(key: keyboard.KeyboardEvent):
        global action
        if key.name == "up":
            action = 1
        elif key.name == "down":
            action = 2
        elif key.name == "left":
            action = 3
        elif key.name == "right":
            action = 4

    keyboard.on_press(on_press)

    i = 0
    all_reward = 0
    while True:
        i += 1
        if i % 10 == 0:
            observation, reward, done, a, info = env.step(action)
            all_reward += reward
            print(all_reward)
            if done:
                print("game over")
                exit()
        env.render()

        clock.tick(60)
