# def run_simple():
#     import gymnasium as gym
#     env = gym.make("ALE/Asteroids-v5", render_mode="human")
#     observation, info = env.reset()
#
#     for _ in range(1000):
#         action = env.action_space.sample()  # agent policy that uses the observation and info
#         observation, reward, terminated, truncated, info = env.step(action)
#
#         if terminated or truncated:
#             observation, info = env.reset()
#
#     env.close()

#
# import gymnasium as gym
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models, optimizers
#
# # Create the Atari environment
# env = gym.make("SpaceInvaders-v4")
# num_actions = env.action_space.n
#
#
# # Define the DQN model
# def build_dqn(input_shape, num_actions):
#     model = models.Sequential()
#     model.add(layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_shape))
#     model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(512, activation='relu'))
#     model.add(layers.Dense(num_actions))
#     return model
#
#
# input_shape = env.observation_space.shape
# dqn_model = build_dqn(input_shape, num_actions)
#
#
# # Define the replay buffer
# class ReplayBuffer:
#     def __init__(self, buffer_size):
#         self.buffer_size = buffer_size
#         self.buffer = []
#         self.index = 0
#
#     def add(self, state, action, reward, next_state, done):
#         experience = (state, action, reward, next_state, done)
#         if len(self.buffer) < self.buffer_size:
#             self.buffer.append(experience)
#         else:
#             self.buffer[self.index] = experience
#         self.index = (self.index + 1) % self.buffer_size
#
#     def sample(self, batch_size):
#         buffer_array = np.array(self.buffer)
#         samples = np.random.choice(buffer_array, batch_size, replace=False)
#         states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
#         return states, actions, rewards, next_states, dones
#
#
# # Define hyperparameters
# epsilon_initial = 1.0
# epsilon_decay = 0.995
# epsilon_min = 0.1
# batch_size = 32
# gamma = 0.99
# learning_rate = 0.001
# target_update_frequency = 1000
#
# # Define the optimizer and loss function
# optimizer = optimizers.Adam(learning_rate)
# loss_fn = tf.losses.MeanSquaredError()
#
# # Initialize the DQN target model
# dqn_target_model = build_dqn(input_shape, num_actions)
# dqn_target_model.set_weights(dqn_model.get_weights())
#
# # Initialize the replay buffer
# replay_buffer = ReplayBuffer(buffer_size=100000)
#
# # Training loop
# num_episodes = 1000
#
# for episode in range(num_episodes):
#     state = env.reset()
#     total_reward = 0
#     done = False
#
#     while not done:
#         # Epsilon-greedy exploration strategy
#         epsilon = max(epsilon_initial * epsilon_decay ** episode, epsilon_min)
#         if np.random.rand() < epsilon:
#             action = env.action_space.sample()
#         else:
#             q_values = dqn_model.predict(np.expand_dims(state, axis=0))
#             action = np.argmax(q_values)
#
#         next_state, reward, done, *_ = env.step(action)
#         total_reward += reward
#
#         replay_buffer.add(state, action, reward, next_state, done)
#         state = next_state
#
#         # Update the DQN model
#         if len(replay_buffer.buffer) >= batch_size:
#             states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
#
#             with tf.GradientTape() as tape:
#                 q_values = dqn_model(states)
#                 next_q_values_target = dqn_target_model(next_states)
#                 target_q_values = rewards + gamma * np.max(next_q_values_target, axis=1) * (1 - dones)
#                 mask = tf.one_hot(actions, num_actions)
#                 predicted_q_values = tf.reduce_sum(q_values * mask, axis=1)
#                 loss = loss_fn(target_q_values, predicted_q_values)
#
#             grads = tape.gradient(loss, dqn_model.trainable_variables)
#             optimizer.apply_gradients(zip(grads, dqn_model.trainable_variables))
#
#         # Update the target model
#         if episode % target_update_frequency == 0:
#             dqn_target_model.set_weights(dqn_model.get_weights())
#
#     print(f"Episode {episode + 1} - Total Reward: {total_reward}")
#
# # Close the environment
# env.close()


import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()