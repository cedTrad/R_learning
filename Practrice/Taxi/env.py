import gym
env = gym.make("Taxi-v3", render_mode = "human")

print("action space : ",env.action_space)

print(f"State Space : {env.observation_space}")



# Rewards
state = 123
action = 0

print(" env.P[state][action][0] ", env.P[state][action][0])


env.reset()

env.s = 123
env.render(mode='human')
