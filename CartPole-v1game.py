import random
import time
import numpy as np
import gym 
from tensorflow.keras.models import load_model

env_name = 'CartPole-v1'
env = gym.make(env_name)  # create the environment

saved_model=load_model("CartPole-v1_model.h5")

observation = env.reset()

for counter in range(300):
    env.render()
    time.sleep(0.05)
   
    action = np.argmax(saved_model.predict(observation.reshape([1,4])))
  
    observation, reward, done, info = env.step(action)
 
    if done:
        break
env.close()
