#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import pacman
import gym
# from matplotlib import pylab
# import random
import numpy as np
# from collections import deque
# import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential


# In[2]:


class pacman:
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 1
        self.learning_rate = 0.01
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
        # build the model
        self.model = Sequential()
        self.model.add(Dense(48, input_dim=self.state_size, activation='relu'))
        self.model.add(Dense(48, activation='relu'))
        self.model.add(Dense(self.action_size, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate))
        self.model.summary()

    def store_episode_info(self, state, action, reward):
        actions = np.zeros([self.action_size])
        actions[action] = 1
        self.episode_actions.append(np.array(actions).astype('float32'))
        self.episode_states.append(state)
        self.episode_rewards.append(reward)

    def generate_a_random_action(self, state):
        state = state.reshape([1, state.shape[0]])
        prediction_probabilities = self.model.predict(state, batch_size=1).flatten()
        action = np.random.choice(self.action_size, 1, p=prediction_probabilities)[0]
        return action

    def discount_rewards(self, episode_rewards):
        discounted_rewards = np.zeros_like(episode_rewards)
        running_add = 0
        for t in reversed(range(0, episode_rewards.size)):
            if episode_rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + episode_rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards


#     def load(self, file_name):
#         self.model.load_weights(file_name)

#     def save(self, file_name):
#         self.model.save_weights(file_name)


# In[ ]:


scores_list = []
if __name__ == "__main__":
    env = gym.make("MsPacman-ram-v0")
    state = env.reset()
    score = 0
    episode = 0

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    player = pacman(state_size, action_size)
    try:
        player.model.load_weights('MsPacman-ram-v0.h5')
    except:
        pass
    
    while True:
        env.render()
        action = player.generate_a_random_action(state)
        next_state, reward, done, info = env.step(action)
        score += reward
        player.store_episode_info(state, action, reward)
        state = next_state

        if done:
            episode += 1
            #train here
            episode_actions = np.vstack(player.episode_actions)
            episode_rewards = np.vstack(player.episode_rewards)
            episode_rewards = player.discount_rewards(episode_rewards)
            episode_rewards = episode_rewards / np.std(episode_rewards)
            episode_actions *= -episode_rewards 
            player.model.train_on_batch(np.squeeze(np.vstack([player.episode_states])), np.squeeze(np.vstack([episode_actions])))
            
            # clean the memory
            player.episode_states, player.episode_actions, player.episode_rewards = [], [], []
            print('Episode: %d - Score: %f.' % (episode, score))
            scores_list.append(score)
            score = 0
            state = env.reset()
            if episode > 1 and episode % 5 == 0:
                player.model.save_weights('MsPacman-ram-v0.h5')


# In[ ]:


with open('your_file.txt', 'w') as f:
    for item in scores_list:
        f.write("%s\n" % item)


# In[ ]:




