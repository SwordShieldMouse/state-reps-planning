from includes import *

## try implementing alphazero's version of MCTS
class Tree():
    # will be a tree with two types of nodes: state nodes or action nodes
    def __init__(self, prior_prob, value, type, name = "root", children = None, parent = None):
        self.name = name # the feature vector of state or action
        self.type = type # whether state or action node
        self.children = []
        self.parent = None

        self.visits = 0
        self.prior_prob = 0
        self.value = 0

        # add children nodes
        if children != None:
            self.children = children
        if parent != None:
            self.parent = parent

## TODO: implements a PSR
class PSR():
    def __init__(self, q):
        self.q = q

## TOD: implementation of a GVF
class GVF():
    def __init__(self, dim_state, dim_action):
        self.cumulant = Cumulant()
        self.termination = Termination()


## implements the cumulant function of a GVF (i.e., the pseudo-reward)
# GVFs take a while to learn, so it probably isn't helpful to backpropagate cumulant and termination at every time step
class Cumulant():
    def __init__(self, dim_state, dim_action):
        self.layer = nn.Linear()

## For predicting the next state, given current state and action
# holds its own step-size, which changes based on how good the predictor is
class Predict(nn.Module):
    def __init__(self, dim_state, dim_action, lr = 1e-4):
        super(Predict, self).__init__()
        self.lr = lr
        self.layer = nn.Linear(dim_state + dim_action, dim_state)
        #print(dim_state + dim_action)

    def forward(self, s, a):
        x = torch.cat([s, a])
        #print(s.shape, a.shape, x.shape)
        return self.layer(x)

    def get_lr(self):
        return self.lr

    def set_lr(self, lr):
        self.lr = lr

# learning rate that adapts according to an objective
class Adaptive_LR(nn.Module):
    # gives a array of learning rates, one for each weight considered
    def __init__(self, wgt_size):
        # lr should depend on current weights and gradient of whatever function we are minimizing, at the current state
        super(Adaptive_LR, self).__init__()

        # Q: does it make sense to initialize learning rates?

        # use sigmoid to squash between 0 and 1
        self.layers = nn.Sequential(
            nn.Linear(2 * wgt_size, wgt_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        # should take in current weights and gradient; returns one learning rate for each weight
        return self.layers(x)

class Policy(nn.Module):
    def __init__(self, dim_state, dim_action):
        super(Policy, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(dim_state, dim_action),
            nn.LogSoftmax(dim = -1)
        )

    def forward(self, x):
        return self.layers(x)

class Value(nn.Module):
    def __init__(self, dim_state):
        super(Value, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(dim_state, 1)
        )

    def forward(self, x):
        return self.layers(x)


def test_adaptive_lr(envs, gamma, episodes, lr = 1e-3):
    for env in envs:
        dim_action = env.action_space.n
        dim_state = env.observation_space.shape[0]

        policy = Policy(dim_state, dim_action)
        value = Value(dim_state)
        # need a learning rate for every parameter in each model
        policy_params_size = sum(p.numel() for p in policy.parameters())
        value_params_size = sum(p.numel() for p in value.parameters())

        # remember we also need lr's for the adaptive lr's
        policy_lrs = Adaptive_LR(policy_params_size)
        value_lrs = Adaptive_LR(value_params_size)

        for episode in range(episodes):
            obs = env.reset()
            done = False

            prev_TD_error = None

            ix = 0
            final_return = 0

            last_td_error = torch.Tensor([0]).to(device)

            while done != True:
                env.render()

                log_probs = policy(torch.Tensor(obs).to(device))
                m = Categorical(logits = log_probs)
                #print("time = {}".format(t), log_probs, obs)
                action = m.sample()

                prev_value = value(torch.Tensor(obs).to(device))
                prev_obs = obs

                obs, reward, done, info = env.step(action.item())

                final_return += reward * (gamma ** ix)

                policy.zero_grad()
                value.zero_grad()

                # Idea for updating learning rates:
                    # rates are updated according to squared TD error of NEXT iteration
                    # basically, lrs affect the next time step, so it's natural to make a cost function based on what happens afterwards
                    # essentially, we are minimizing an expected TD error, where the states are drawn from the usual state distribution and the time subscripts are also drawn from a distribution
                    # so at each iteration, we first optimize the step-sizes with respect to the TD-error from the last time step

                # actor-critic
                delta = reward + gamma * value(torch.Tensor(obs).to(device)).detach() - prev_value.detach() # detach since we do semi-gradient TD
                policy_loss = -(gamma ** ix) * delta * m.log_prob(action)
                value_loss = delta * value(torch.Tensor(prev_obs).to(device))


                # TD-error gives us an idea of where we should have had the previous step-sizes
                lr_loss = last_td_error ** 2

                # need to keep track of this since we want to optimize the lr for the next iteration based on the actual TD error and not the one calculated with already updated weights
                last_td_error = reward + gamma * value(torch.Tensor(obs).to(device)) - prev_value.clone()

                policy_loss.backward(retain_graph = True)
                value_loss.backward(retain_graph = True)
                lr_loss.backward()

                ix += 1 # tracker for discounting


                # optimize policy and value function
                with torch.no_grad():
                    for ix, param in enum(policy.parameters()):
                        param -= policy_lrs[ix](torch.cat([param, param.grad])) * param.grad
                    for ix, param in enum(value.parameters()):
                        param -= value_lrs[ix](torch.cat([param, param.grad])) * param.grad
                    for param in policy_lrs.parameters():
                        param -= lr * param.grad
                    for param in value_lrs.parameters():
                        param -= lr * param.grad

            print("total return of {} at end of episode {}".format(final_return, episode + 1))

def train_prediction(env, gamma, n_predictions, episodes):
    # how do we evaluate whether our predictions are good?
        # could have learning curves of all the predictions
    dim_action = env.action_space.n
    dim_state = env.observation_space.shape[0]
    #print(dim_action, dim_state)

    predicts = [Predict(dim_state, dim_action) for _ in range(n_predictions)]

    predict_errors = {}
    for i in range(n_predictions):
        predict_errors[i] = []

    for episode in range(episodes):
        state = env.reset()
        done = False

        while done != True:
            # just take random actions because we're only interested in prediction
            action = env.action_space.sample()

            # encode action as a one-hot vector
            action_vec = torch.zeros(dim_action).to(device)
            action_vec[action] = 1.0

            predictions = [predict(torch.Tensor(state).to(device), action_vec) for predict in predicts]

            state, reward, done, info = env.step(action)

            # evaluate how well our predictions did
            # our reward function is 1/2 * (state - prediction)^2, so our gradient would be -nabla prediction * (state - prediction)
            for ix, predict in enumerate(predicts):
                predict.zero_grad()
                loss = torch.sum((torch.Tensor(state).to(device) - predictions[ix]) ** 2)
                predict_errors[i].append(loss.item())
                #print(loss.item())
                loss.backward()

                lr = predict.get_lr()
                with torch.no_grad():
                    for param in predict.parameters():
                        param -= lr * param.grad

                # if our error for a particular prediction is high, set the lr high
                # if our error has been low for some time, set the lr low because we don't want to destabilise
                # Could also try adaptive learning rate of t / E[x^t x] as suggested in the book
                # TODO: should probably prove that this type of step-size adjustment gives convergence or has some desirable properties
                    # probably involves characterising slow convergence

                # should probably include situations where we don't touch the learning rate at all
                #if np.mean(predict_errors[ix][-1:-10]) < 5.0:
                    # arbitrary placeholder threshold
                #    predict.set_lr(lr / 2)
                #else:
                #    predict.set_lr(lr * 2)

                # if the error has been high for a very long time, consider reinitializing it to one of the ones with better error?

                # if the error has been low for a very long time, perturb a bit so that we are still improving if necessary
    return predict_errors
