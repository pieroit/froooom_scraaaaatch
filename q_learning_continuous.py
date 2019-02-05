
# environment MountainCar-v0

import numpy as np
import gym
from sklearn.neural_network import MLPRegressor

# load environment
env = gym.make('MountainCar-v0')
print('Observation space', env.observation_space)
print('Action space', env.action_space)

# prepare agent
#learning_rate    = 0.01

# Q model
Q = MLPRegressor([10])

# just to initialize the model (input and output shape)
fake_X = np.zeros([1, env.observation_space.shape[0]])
fake_y = np.zeros([1, env.action_space.n])
Q.partial_fit(fake_X, fake_y)

# learn for many episodes
n_episodes = 5001
for episode in range(n_episodes):

    # restart environment
    state = env.reset()
    cumulative_reward = 0.0
    exploration_prob = 0.1

    done = False
    while not done:

        # show the environment
        if episode % 10 == 0:
            env.render()

        # choose action
        if np.random.random() < exploration_prob:
            # random action
            action = env.action_space.sample()
        else:
            # best action
            action = Q.predict([state])[0]
            action = np.argmax(action)

        # step
        next_state, reward, done, info = env.step(action)

        # overriding reward
        position, velocity = next_state
        reward = abs(velocity)
        if position > 0:
            reward += position
        reward *= 10
        #print(reward)

        cumulative_reward += reward

        # learn
        target         = Q.predict([state])[0]
        #print(target)
        target[action] = reward + ( 0.1 * np.max( Q.predict([next_state])[0] ))
        Q.partial_fit([state], [target])

        # update state
        state = next_state

    print('Episode', episode, 'epsilon', exploration_prob, 'sum of rewards', cumulative_reward)
    #print(Q)
