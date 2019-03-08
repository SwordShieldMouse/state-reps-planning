from algs import *

env = gym.make("CartPole-v0")

## we will try to product the next state of the cart pole given the current state and the action taken
episodes = 100
trials = 1
gamma = 0.99
n_predictions = 10

for trial in range(trials):
    print("starting trial {}".format(trial + 1))
    for episode in range(episodes):
        train_prediction(env = env, gamma = gamma, episodes = episodes, n_predictions = n_predictions)
