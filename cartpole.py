import gym

from dataclasses import dataclass

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
        # (observation, reward, done, info)
        pass


class TDGradientAgent:
    pass


# https://gym.openai.com/docs/
def do_nothing(env):
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # take a random action


def random_action(env):
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break


@dataclass
class StringSpaceCartPole(gym.core.Env):
    underlying: gym.core.Env

    def __init__(self, env: gym.core.Env):
        self.underlying = env
        self.action_space = env.action_space

    def reset(self):
        return self.underlying.reset()

    def render(self, mode='human'):
        return self.underlying.render(mode)

    def observation_space(self):
        return "str"

    def metadata(self):
        return self.underlying.metadata

    def reward_range(self):
        return self.underlying.reward_range

    def step(self, action):
        observation, reward, done, info = self.underlying.step(action)
        return self._stringify(observation), reward, done, info

    def _stringify(self, observation) -> str:
        x = observation[0]
        x_v = observation[1]
        theta = observation[2]  # +/-0.42
        theta_v = observation[3]  # +/-inf

        theta = str(np.round((theta / 0.42) * 7))

        theta_v_maxabs = 6
        theta_v = np.round(np.clip(theta_v, -theta_v_maxabs, theta_v_maxabs))

        return "{0}/{1}".format(theta, theta_v)


class OnPolicyQAgent:
    import numpy as np
    from gym.core import Env

    env: Env

    # TODO: actionを選ぶ部分を分離
    # TODO: alpha決定する部分を分離
    maxT: int = 100
    trial: int = 1000

    def __init__(self, env: Env):
        self.env = env

        self.q_table: np.ndarray = np.zeros(())

        pass

    def train(self):
        for _ in range(self.trial):
            for t in range(self.maxT):
                env.render()
                print(observation)
                action = env.action_space.sample()
                observation, reward, done, info = env.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break

        pass


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    print(type(env))
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)

    env = StringSpaceCartPole(env)

    q_agent = OnPolicyQAgent(env)

    for _ in range(10):
        env.reset()
        for t in range(100):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print(observation)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

    #qlearn = OnPolicyQAgent(strCartPole)

    # do_nothing(env)
    # random_action(env)
