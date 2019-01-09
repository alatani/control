import gym


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
