from algs import *

env = gym.make("CartPole-v0")

envs = [env]

## we will try to product the next state of the cart pole given the current state and the action taken
episodes = 100
trials = 1
gamma = 0.99
n_predictions = 10
lr = 1e-3

returns = {"fixed-lr": [], "modification": [], "episode": [], "return": []}
episode_ixs = [i + 1 for i in range(episodes)]

for trial in range(trials):
    print("starting trial {}".format(trial + 1))

    returns["return"] += test_adaptive_lr(env = env, gamma = gamma, episodes = episodes, lr = lr)
    returns["episode"] + episode_ixs
    returns["modification"] += ["adaptive_lr"] * episodes
    returns["fixed-lr"] += [lr] * episodes
    #train_prediction(env = env, gamma = gamma, episodes = episodes, n_predictions = n_predictions)

# plot what we have
df = pd.DataFrame(returns)
seaborn.lineplot(x = "episode", y = "return", data = df, hue = "fixed-lr", style = 'modification', legend = 'full')
plt.show()
