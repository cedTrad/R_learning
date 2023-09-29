import gym
import time

env = gym.make("Taxi-v3", render_mode = "human")

class RandomAgent:
    
    def __init__(self, env):
        self.env = env
    
    
    def get_action(self, state):
        return self.env.action_space.sample()
    


agent = RandomAgent(env)


env.reset()
state = 123
env.s = state 

epochs = 0
penalties = 0
reward = 0

# store framess to latter plot them
frames = []

done = False

while not done:
    time.sleep(0.2)
    action= agent.get_action(state)
    state, reward, done, info, _ = env.step(action)
    
    env.render()
    
    if reward == -10:
        penalties += 1
    
    frames.append({
        "state" : state,
        "action" : action,
        "reward" : reward
    }
                  )
    print(f"state : {state}  --- action : {action} --- reward : {reward} --- done : {done}")
    
    epochs += 1

print("Timesteps taken : {}".format(epochs))
print("Penalties incurred : {}".format(penalties))




# Let's generate histograms to quantify performance

from tqdm import tqdm

n_episodes = 100

# for plotting metrics
Timesteps_per_episode = []
penalties_per_episode = []


for i in tqdm(range(0, n_episodes)):
    state = env.reset()
    
    epochs, penalties, reward = 0, 0, 0
    done = False
    
    while not done:
        
        action = agent.get_action(state)
        next_state, reward, done, info, _ = env.step(action)
        
        if reward == -10:
            penalties += 1
        
        state = next_state
        epochs += 1
        
    timesteps_per_episode.append(epochs)
    penalties_per_episode.append(penalties)
    
    