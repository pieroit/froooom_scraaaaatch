
# environment CliffWalking-v0
#   n_episodes = 50000
#   learning_rate = 0.1
#   epsilon = 0.01

# environment Taxi-v2
#   n_episodes = 10000
#   learning_rate = 0.1
#   epsilon = 0.1

import numpy as np
import gym

# load environment
env = gym.make('CliffWalking-v0')
#env = gym.make('Taxi-v2')
print('Observation space', env.observation_space)
print('Action space', env.action_space)

# prepare agent
learning_rate    = 0.1
Q_table_shape = [env.observation_space.n, env.action_space.n]
Q = np.zeros(Q_table_shape)
print(Q)

# learn for many episodes
exploration_prob = 1
n_episodes = 10001
for episode in range(n_episodes):

    # restart environment
    state = env.reset()
    cumulative_reward = 0.0
    exploration_prob *= 0.999

    done = False
    while not done:

        # show the environment
        if episode % 100 == 0:
            env.render()

        # choose action
        if np.random.random() < exploration_prob:
            # random action
            action = env.action_space.sample()
        else:
            # best action
            action = np.argmax( Q[state, :] )

        # step
        next_state, reward, done, info = env.step(action)
        cumulative_reward += reward
        #print('State', next_state)
        #print('Reward', reward)
        #print('\n')

        # learn
        prediction = Q[state, action]
        target     = reward + np.max(Q[next_state, :])
        error      = target - prediction
        Q[state, action] += learning_rate * error

        # update state
        state = next_state

    print('Episode', episode, 'epsilon', exploration_prob, 'sum of rewards', cumulative_reward)
    #print(Q)
