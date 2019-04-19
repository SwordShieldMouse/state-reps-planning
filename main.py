from algs import *
from training_algs import *

env = gym.make("CartPole-v0")

envs = [env]

## we will try to product the next state of the cart pole given the current state and the action taken
episodes = 500
trials = 5
gamma = 0.99
lambda_t = 0.5
n_predictions = 10
#lr = 1e-4
lrs = [1e-4 * (2 ** i) for i in range(0, 4)]

RUN_GVF = True
RUN_ADAPTIVE_LR = False

returns = {"fixed-lr": [], "modification": [], "episode": [], "return": []}
gvf_returns = {"lr": [], "episode": [], "return": []}
episode_ixs = [i + 1 for i in range(episodes)]

if RUN_GVF == True:
    for lr in lrs:
        for trial in range(trials):
            print("starting trial {}".format(trial + 1))

            gvf_returns["return"] += train_off_policy_gvf(env = env, gamma = gamma, episodes = episodes, lr = lr)
            gvf_returns["episode"] += episode_ixs
            #returns["modification"] += [modification] * episodes
            gvf_returns["lr"] += [lr] * episodes

    df = pd.DataFrame(gvf_returns)
    seaborn.lineplot(x = "episode", y = "return", data = df, hue = "lr", legend = 'full')
    plt.show()

if RUN_ADAPTIVE_LR == True:
    for modification in ("adaptive_lr", "none"):
        print("working on modification = {}".format(modification))
        for trial in range(trials):
            print("starting trial {}".format(trial + 1))

            if modification == "adaptive_lr":
                returns["return"] += test_adaptive_lr(env = env, gamma = gamma, episodes = episodes, lr = lr)
            elif modification == "none":
                returns["return"] += vanilla_actor_critic(env = env, gamma = gamma, episodes = episodes, lr = lr)
            returns["episode"] += episode_ixs
            returns["modification"] += [modification] * episodes
            returns["fixed-lr"] += [lr] * episodes
        #train_prediction(env = env, gamma = gamma, episodes = episodes, n_predictions = n_predictions)

    # plot what we have
    df = pd.DataFrame(returns)
    seaborn.lineplot(x = "episode", y = "return", data = df, hue = "fixed-lr", style = 'modification', legend = 'full')
    plt.show()
