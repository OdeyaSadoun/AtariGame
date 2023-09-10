# https://github.com/sudharsan13296/Deep-Reinforcement-Learning-With-Python/blob/master/09.%20%20Deep%20Q%20Network%20and%20its%20Variants/9.03.%20Playing%20Atari%20Games%20using%20DQN.ipynb

#step 1:
import tensorflow as tf
print(tf.__version__)

#step 2:
import random
import gymnasium as gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import cv2


#step 3:
env = gym.make("MsPacman-v4", render_mode="human")

#step 4:
state_size = (88, 80, 1)

#step 5:
action_size = env.action_space.n

#step 6:
color = np.array([210, 164, 74]).mean()



def preprocess_state(state):
    # Convert the state to grayscale
    grayscale_state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)

    # Resize the state to a smaller resolution (e.g., 84x84)
    resized_state = cv2.resize(grayscale_state, (84, 84))

    # Normalize pixel values to a range between 0 and 1
    normalized_state = resized_state / 255.0

    return normalized_state



#step 7:
class DQN:
    def __init__(self, state_size, action_size):

        # define the state size
        self.state_size = state_size

        # define the action size
        self.action_size = action_size

        # define the replay buffer
        self.replay_buffer = deque(maxlen=5000)

        # define the discount factor
        self.gamma = 0.9

        # define the epsilon value
        self.epsilon = 0.8

        # define the update rate at which we want to update the target network
        self.update_rate = 1000

        # define the main network
        self.main_network = self.build_network()

        # define the target network
        self.target_network = self.build_network()

        # copy the weights of the main network to the target network
        self.target_network.set_weights(self.main_network.get_weights())

    # Let's define a function called build_network which is essentially our DQN.

    def build_network(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=4, padding='same', input_shape=self.state_size))
        model.add(Activation('relu'))

        model.add(Conv2D(64, (4, 4), strides=2, padding='same'))
        model.add(Activation('relu'))

        model.add(Conv2D(64, (3, 3), strides=1, padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())

        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam())

        return model

    # We learned that we train DQN by randomly sampling a minibatch of transitions from the
    # replay buffer. So, we define a function called store_transition which stores the transition information
    # into the replay buffer

    def store_transistion(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    # We learned that in DQN, to take care of exploration-exploitation trade off, we select action
    # using the epsilon-greedy policy. So, now we define the function called epsilon_greedy
    # for selecting action using the epsilon-greedy policy.

    def epsilon_greedy(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_size)

        Q_values = self.main_network.predict(state)

        return np.argmax(Q_values[0])

    # train the network
    def train(self, batch_size):

        # sample a mini batch of transition from the replay buffer
        minibatch = random.sample(self.replay_buffer, batch_size)

        # compute the Q value using the target network
        for state, action, reward, next_state, done in minibatch:
            if not done:
                target_Q = (reward + self.gamma * np.amax(self.target_network.predict(next_state)))
            else:
                target_Q = reward

            # compute the Q value using the main network
            Q_values = self.main_network.predict(state)

            Q_values[0][action] = target_Q

            # train the main network
            self.main_network.fit(state, Q_values, epochs=1, verbose=0)

    # update the target network weights by copying from the main network
    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())


#step 8:
num_episodes = 500

#step 9:
num_timesteps = 20000

#step 10:
batch_size = 8

#step 11:
num_screens = 4

#step 12:
dqn = DQN(state_size, action_size)

#step 13:
done = False
time_step = 0

# for each episode
for i in range(num_episodes):

    # set return to 0
    Return = 0

    # preprocess the game screen
    state = preprocess_state(env.reset())

    # for each step in the episode
    for t in range(num_timesteps):

        # render the environment
        env.render()

        # update the time step
        time_step += 1

        # update the target network
        if time_step % dqn.update_rate == 0:
            dqn.update_target_network()

        # select the action
        action = dqn.epsilon_greedy(state)

        # perform the selected action
        next_state, reward, done, _ = env.step(action)

        # preprocess the next state
        next_state = preprocess_state(next_state)

        # store the transition information
        dqn.store_transistion(state, action, reward, next_state, done)

        # update current state to next state
        state = next_state

        # update the return
        Return += reward

        # if the episode is done then print the return
        if done:
            print('Episode: ', i, ',' 'Return', Return)
            break

        # if the number of transistions in the replay buffer is greater than batch size
        # then train the network
        if len(dqn.replay_buffer) > batch_size:
            dqn.train(batch_size)