import gym

import dataclass

from typing import List
import numpy as np

# https://qiita.com/animegazer/items/4158462e5a3efaba9d7b

@dataclass
class ProductionPlan:
    sozai: float

@dataclass
class State:
    sozai: float
    buhin: float
    seihin: float


@dataclass
class ProductionScheduling:
    state: State

    k1 = 0.5
    k2 = 0.8

    def __init__(self):
        self.state = State(sozai=0, buhin=0, seihin=0)
        pass


    def reset(self):
        pass

    def render(self):
        pass

    def step(self, action: ProductionPlan):
        #(observation, reward, done, info)
        pass




# https://gym.openai.com/docs/
def do_nothing(env):
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample()) # take a random action

def random_action(env):
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break


if __name__=="__main__":
    env = gym.make('CartPole-v0')
    #do_nothing(env)
    random_action(env)
