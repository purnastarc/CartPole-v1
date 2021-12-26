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

# model construction
model = Sequential()

model.add(Dense(16, input_shape=(1, num_observations)))
model.add(Activation('relu'))

model.add(Dense(32))
model.add(Activation('relu'))


model.add(Dense(num_actions))
model.add(Activation('linear'))

target_model = clone_model(model)

# learning parameters
EPOCHS = 400
epsilon = 1.0
EPSILON_REDUCE = 0.995  
LEARNING_RATE = 0.001 
GAMMA = 0.95


# stochastic policy
def epsilon_greedy_action_selection(model, epsilon, observation):
    if np.random.random() > epsilon:
        prediction = model.predict(observation) 
        action = np.argmax(prediction)  
    else:
        action = np.random.randint(0, env.action_space.n) 
    return action

replay_buffer = deque(maxlen=20000)
update_target_model = 10


# experince replay buffer function
def replay(replay_buffer, batch_size, model, target_model):
    
    if len(replay_buffer) < batch_size: 
        return
    
    samples = random.sample(replay_buffer, batch_size)  
    
    
    target_batch = []  
    
    zipped_samples = list(zip(*samples))  
    states, actions, rewards, new_states, dones = zipped_samples  
    
    targets = target_model.predict(np.array(states))
    
   
    q_values = model.predict(np.array(new_states))  
    
    for i in range(batch_size):  
       
        q_value = max(q_values[i][0])  
        
        target = targets[i].copy()  
        if dones[i]:
            target[0][actions[i]] = rewards[i]
        else:
            target[0][actions[i]] = rewards[i] + q_value * GAMMA
        target_batch.append(target)

    model.fit(np.array(states), np.array(target_batch), epochs=1, verbose=0)  
    

 # copying the weights from Q-network to Target-network
def update_model_handler(epoch, update_target_model, model, target_model):
    if epoch > 0 and epoch % update_target_model == 0:
        target_model.set_weights(model.get_weights())

 # compile the model
model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))

# training the Q-network model
best_so_far = 0
for epoch in range(EPOCHS):
    observation = env.reset() 
    
    
    observation = observation.reshape([1, 4])  
    done = False  
    
    points = 0
    while not done: 
        
        # Select action acc. to strategy
        action = epsilon_greedy_action_selection(model, epsilon, observation)
        
        # Perform action and get next state
        next_observation, reward, done, info = env.step(action)  
        next_observation = next_observation.reshape([1, 4])
        replay_buffer.append((observation, action, reward, next_observation, done))
        observation = next_observation 
        points+=1

      
        replay(replay_buffer, 32, model, target_model)

    
    epsilon *= EPSILON_REDUCE 
    
    
    update_model_handler(epoch, update_target_model, model, target_model)
    
    if points > best_so_far:
        best_so_far = points
    if epoch %25 == 0:
        print(f"{epoch}: Points reached: {points} - epsilon: {epsilon} - Best: {best_so_far}")
        
        
# save the model
model.save("CartPole-v1_model", save_format="h5")
