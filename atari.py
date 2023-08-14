
import random
import gymnasium as gym
import numpy as np
from collections import deque
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
# from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import cv2

# Preprocess the game screen

color = np.array([210, 164, 74]).mean()

class DQN:
    def __init__(self, state_size, action_size, frame_stack_size):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=5000)
        self.gamma = 0.9
        self.epsilon = 0.8
        self.update_rate = 1000
        self.main_network = self.build_network()
        self.target_network = self.build_network()
        self.target_network.set_weights(self.main_network.get_weights())

    def preprocess_state(self, state):
        state_observation = state[0]

        if len(state_observation.shape) == 1:
            # Return the 1-dimensional state_observation as is
            return state_observation
        elif len(state_observation.shape) == 2:
            # Assuming state_observation is a 2-dimensional image
            # Convert the image to 3 channels
            image = np.expand_dims(state_observation, axis=2)
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return gray_image
        else:
            # If the image already has 3 color channels, return it as is
            return state_observation

    def stack_frames(self, state):
        # Preprocess the state and stack it with previous frames
        processed_frame = self.preprocess_state(state)
        if len(self.frame_stack) == 0:
            for _ in range(self.frame_stack_size):
                self.frame_stack.append(processed_frame)
        else:
            self.frame_stack.append(processed_frame)
        stacked_frames = np.stack(self.frame_stack, axis=-1)
        return stacked_frames

    def build_network(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (8, 8), strides=4, padding='same', input_shape=self.state_size))
        model.add(tf.keras.layers.Activation('relu'))

        model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=2, padding='same'))
        model.add(tf.keras.layers.Activation('relu'))

        model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same'))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
        return model

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def epsilon_greedy(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_size)

        # Stack the frames before passing to the network
        stacked_frames = self.stack_frames(state)

        Q_values = self.main_network.predict(np.expand_dims(stacked_frames, axis=0))
        return np.argmax(Q_values[0])

    def train(self, batch_size):
        minibatch = random.sample(self.replay_buffer, batch_size)

        # Initialize arrays to store batched states and targets
        batch_states = []
        batch_targets = []

        for state, action, reward, next_state, done in minibatch:
            # Stack frames for current state and next state
            state_frames = self.stack_frames(state)
            next_state_frames = self.stack_frames(next_state)

            # Predict Q-values for current state and target Q-values for next state
            current_Q_values = self.main_network.predict(np.expand_dims(state_frames, axis=0))
            next_state_target_Q = self.target_network.predict(np.expand_dims(next_state_frames, axis=0))

            # Calculate target Q-value using the Bellman equation
            if done:
                target_Q = reward
            else:
                target_Q = reward + self.gamma * np.amax(next_state_target_Q)

            # Update the Q-value for the selected action
            current_Q_values[0][action] = target_Q

            batch_states.append(state_frames)
            batch_targets.append(current_Q_values)

        # Convert the lists to numpy arrays
        batch_states = np.array(batch_states)
        batch_targets = np.array(batch_targets)

        # Train the network using the batched data
        self.main_network.fit(batch_states, batch_targets, epochs=1, verbose=0)

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

# Create the environment
env = gym.make('BreakoutNoFrameskip-v4')


# env = gym.make("ALE/MsPacman-v5", render_mode="human")
state_size = (88, 80, 1)
action_size = env.action_space.n

# Create the DQN agent
frame_stack_size = 4
dqn = DQN(state_size, action_size, frame_stack_size)

num_episodes = 500
num_timesteps = 20000
batch_size = 8

time_step = 0

# Loop through episodes
# Loop through episodes
for i in range(num_episodes):
    Return = 0
    state = env.reset()[0]

    # Use the preprocess_state method from the DQN class
    state = dqn.preprocess_state(state)

    # Loop through timesteps
    for t in range(num_timesteps):
        env.render()
        time_step += 1

        if time_step % dqn.update_rate == 0:
            dqn.update_target_network()

        action = dqn.epsilon_greedy(state)

        next_state, reward, done, *_ = env.step(action)

        # Use the preprocess_state method from the DQN class
        next_state = dqn.preprocess_state(next_state)

        dqn.store_transition(state, action, reward, next_state, done)
        state = next_state
        Return += reward

        if done:
            print('Episode: ', i, ', Return', Return)
            break

        if len(dqn.replay_buffer) > batch_size:
            dqn.train(batch_size)

env.close()
