from collections import deque
import random

import numpy as np
import gym  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense  
from tensorflow.keras.layers import Activation 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model,load_model

env_name = 'CartPole-v1'
env = gym.make(env_name)

num_actions = env.action_space.n
num_observations = env.observation_space.shape[0]
print(f"There are {num_actions} possible actions and {num_observations} observations")

model = Sequential()

model.add(Dense(16, input_shape=(1, num_observations)))
model.add(Activation('relu'))

model.add(Dense(32))
model.add(Activation('relu'))


model.add(Dense(num_actions))
model.add(Activation('linear'))

print(model.summary())

target_model = clone_model(model)

EPOCHS = 400

epsilon = 1.0
EPSILON_REDUCE = 0.995  
LEARNING_RATE = 0.001 
GAMMA = 0.95

def epsilon_greedy_action_selection(model, epsilon, observation):
    if np.random.random() > epsilon:
        prediction = model.predict(observation) 
        action = np.argmax(prediction)  
    else:
        action = np.random.randint(0, env.action_space.n) 
    return action

replay_buffer = deque(maxlen=20000)
update_target_model = 10

def replay(replay_buffer, batch_size, model, target_model):
    
    if len(replay_buffer) < batch_size: 
        return
    
    samples = random.sample(replay_buffer, batch_size)  
    
    
    target_batch = []  
    
    # Efficient way to handle the sample by using the zip functionality
    zipped_samples = list(zip(*samples))  
    states, actions, rewards, new_states, dones = zipped_samples  
    
    # Predict targets for all states from the sample
    targets = target_model.predict(np.array(states))
    
    # Predict Q-Values for all new states from the sample
    q_values = model.predict(np.array(new_states))  
    
    # Now we loop over all predicted values to compute the actual targets
    for i in range(batch_size):  
        
        # Take the maximum Q-Value for each sample
        q_value = max(q_values[i][0])  
        
        # Store the ith target in order to update it according to the formula
        target = targets[i].copy()  
        if dones[i]:
            target[0][actions[i]] = rewards[i]
        else:
            target[0][actions[i]] = rewards[i] + q_value * GAMMA
        target_batch.append(target)

    # Fit the model based on the states and the updated targets for 1 epoch
    model.fit(np.array(states), np.array(target_batch), epochs=1, verbose=0)  

def update_model_handler(epoch, update_target_model, model, target_model):
    if epoch > 0 and epoch % update_target_model == 0:
        target_model.set_weights(model.get_weights())

model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))


best_so_far = 0
for epoch in range(EPOCHS):
    observation = env.reset()  # Get inital state
    
    # Keras expects the input to be of shape [1, X] thus we have to reshape
    observation = observation.reshape([1, 4])  
    done = False  
    
    points = 0
    while not done:  # as long current run is active
        
        # Select action acc. to strategy
        action = epsilon_greedy_action_selection(model, epsilon, observation)
        
        # Perform action and get next state
        next_observation, reward, done, info = env.step(action)  
        next_observation = next_observation.reshape([1, 4])  # Reshape!!
        replay_buffer.append((observation, action, reward, next_observation, done))  # Update the replay buffer
        observation = next_observation  # update the observation
        points+=1

        # Most important step! Training the model by replaying
        replay(replay_buffer, 32, model, target_model)

    
    epsilon *= EPSILON_REDUCE 
    
    
    update_model_handler(epoch, update_target_model, model, target_model)
    
    if points > best_so_far:
        best_so_far = points
    if epoch %25 == 0:
        print(f"{epoch}: Points reached: {points} - epsilon: {epsilon} - Best: {best_so_far}")


observation = env.reset()
for counter in range(300):
    env.render()
    
    action = np.argmax(model.predict(observation.reshape([1,4])))
     
    observation, reward, done, info = env.step(action) 
    
    if done:
        print(f"done")
        break
env.close()