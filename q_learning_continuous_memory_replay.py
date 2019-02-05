
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
X_memory = np.zeros([1, env.observation_space.shape[0]])
y_memory = np.zeros([1, env.action_space.n])
Q.partial_fit(X_memory, y_memory)

# learn for many episodes
exploration_prob = 1
n_episodes = 5001
for episode in range(n_episodes):

    # restart environment
    state = env.reset()
    cumulative_reward = 0.0
    exploration_prob *= 0.99
    latest_velocity = 0.0

    random_sequence = None

    done = False
    while not done:

        # show the environment
        if episode % 5 == 0:
            env.render()

        # choose action
        if np.random.random() < exploration_prob or random_sequence is not None:

            if random_sequence is None:
                random_sequence = [env.action_space.sample()]

            random_sequence.append(random_sequence[-1])
            action = random_sequence[-1]

            if len(random_sequence) > 50:
                random_sequence = None

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
        if position > 0.45:
            reward += 1
        reward *= 10

        cumulative_reward += reward

        # learn
        target         = Q.predict([state])[0]
        target[action] = reward + ( 0.1 * np.max( Q.predict([next_state])[0] ))
        X_memory = np.append(X_memory, [state], axis=0)
        y_memory = np.append(y_memory, [target], axis=0)
        sample_indexes = np.random.choice(len(X_memory), 10)
        Q.partial_fit([state], [target])
        Q.partial_fit(X_memory[sample_indexes], y_memory[sample_indexes])

        # update state
        state = next_state

    print('Episode', episode, 'epsilon', exploration_prob, 'sum of rewards', cumulative_reward)
    #print(Q)
