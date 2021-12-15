from collections import deque
import random
import time

import numpy as np
import gym  # Contains the game we want to play
from tensorflow.keras.models import Sequential  # To compose multiple Layers
from tensorflow.keras.layers import Dense  # Fully-Connected layer
from tensorflow.keras.layers import Activation  # Activation functions
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model,load_model

env_name = 'CartPole-v1'
env = gym.make(env_name)  # create the environment

saved_model=load_model("CartPole.h5")

observation = env.reset()
Score=0
time.sleep(5)

for counter in range(300):
    env.render()
    time.sleep(0.05)
    # TODO: Get discretized observation
    action = np.argmax(saved_model.predict(observation.reshape([1,4])))
    
    # TODO: Perform the action 
    observation, reward, done, info = env.step(action) # Finally perform the action
    Score=Score+1
    
    if done:
        print("Score: ")
        print(Score)
        break
env.close()