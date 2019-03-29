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
        # mb should combined LR and policy into one network that is trained?
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

def vanilla_actor_critic(env, gamma, episodes, lr = 1e-3):
    dim_action = env.action_space.n
    dim_state = env.observation_space.shape[0]

    policy = Policy(dim_state, dim_action)
    value = Value(dim_state)

    policy_optim = torch.optim.SGD(policy.parameters(), lr = lr)
    value_optim = torch.optim.SGD(value.parameters(), lr = lr)

    returns = [] # holds returns for each episode

    for episode in range(episodes):
        obs = env.reset()
        done = False

        #prev_TD_error = None

        timestep = 0
        final_return = 0

        #last_td_error = torch.Tensor([0]).to(device)

        while done != True:
            env.render()

            log_probs = policy(torch.Tensor(obs).to(device))
            m = Categorical(logits = log_probs)
            #print("time = {}".format(t), log_probs, obs)
            action = m.sample()

            prev_value = value(torch.Tensor(obs).to(device))
            prev_obs = torch.Tensor(obs).to(device)

            obs, reward, done, info = env.step(action.item())

            obs = torch.Tensor(obs).to(device)


            final_return += reward * (gamma ** timestep)


            # zero grad again because we used them in the simulation
            policy_optim.zero_grad()
            value_optim.zero_grad()

            delta = reward + gamma * value(obs).detach() - value(prev_obs).detach()
            policy_loss = -(gamma ** timestep) * delta * m.log_prob(action)
            value_loss = -delta * value(torch.Tensor(prev_obs).to(device))


            policy_loss.backward(retain_graph = True)
            value_loss.backward()

            timestep += 1 # tracker for discounting


            policy_optim.step()
            value_optim.step()

        #print("total return of {} at end of episode {}".format(final_return, episode + 1))
        returns.append(final_return)
    return returns


def test_adaptive_lr(env, gamma, episodes, lr = 1e-3):
    dim_action = env.action_space.n
    dim_state = env.observation_space.shape[0]

    policy = Policy(dim_state, dim_action)
    value = Value(dim_state)

    sim_policy = Policy(dim_state, dim_action)
    sim_value = Value(dim_state)

    # make sure weights of sim and not are the same
    sim_policy.load_state_dict(policy.state_dict())
    sim_value.load_state_dict(value.state_dict())

    # need a learning rate for every parameter in each model
    policy_params_size = sum(p.numel() for p in policy.parameters())
    value_params_size = sum(p.numel() for p in value.parameters())

    returns = [] # holds returns for each episode

    # remember we also need lr's for the adaptive lr's
    policy_lrs = []
    value_lrs = []
    for p in policy.parameters():
        policy_lrs.append(Adaptive_LR(p.numel()))
    for p in value.parameters():
        value_lrs.append(Adaptive_LR(p.numel()))

    adaptive_lr_sample = {"episode": [], "index": [], "param_name": [], "lr": []} # holds an adaptive lr as it evolves through the episodes; for evaluation

    for episode in range(episodes):
        obs = env.reset()
        done = False

        timestep = 0
        final_return = 0

        while done != True:
            env.render()

            log_probs = policy(torch.Tensor(obs).to(device))
            m = Categorical(logits = log_probs)
            #print("time = {}".format(t), log_probs, obs)
            action = m.sample()

            prev_value = value(torch.Tensor(obs).to(device))
            prev_obs = torch.Tensor(obs).to(device)

            obs, reward, done, info = env.step(action.item())

            obs = torch.Tensor(obs).to(device)


            final_return += reward * (gamma ** timestep)

            policy.zero_grad()
            value.zero_grad()

            for lrs in policy_lrs:
                lrs.zero_grad()
            for lrs in value_lrs:
                lrs.zero_grad()

            # Idea for updating learning rates:
                # rates are updated according to squared TD error of NEXT iteration
                # basically, lrs affect the next time step, so it's natural to make a cost function based on what happens afterwards
                # essentially, we are minimizing an expected TD error, where the states are drawn from the usual state distribution and the time subscripts are also drawn from a distribution
                # so at each iteration, we first optimize the step-sizes with respect to the TD-error from the last time step
                # this is essentially a line search
                # does it make sense to do this with a trust region?
            # if we had a planner, we could optimize step-sizes so that the TD error of the next state/action is minimized

            # before we actually update our policy and value weights, simulate an update with our variable step-sizes to the policy and value, and see what the TD-error is one the next iteration, with the same states. Run descent on this squared TD error to optimize the variable step-sizes before we do anything tot he value and policy weights.

            # simulated update according to current TD-error
            sim_delta = reward + gamma * value(torch.Tensor(obs).to(device)).detach() - prev_value.detach() # detach since we do semi-gradient TD
            #sim_policy_loss = -(gamma ** ix) * sim_delta * m.log_prob(action)
            sim_value_loss = sim_delta * value(torch.Tensor(prev_obs).to(device))

            #sim_policy_loss.backward()
            sim_value_loss.backward()

            # update the simulation networks with the gradient of the current TD error
            # save the resulting params into tensors and later we use them in getting the TD error

            value_params = []
            value_grads = []
            for ix, param in enumerate(value.parameters()):
                value_grads.append(param.grad)
                value_params.append(param.data)
            #print(value_grads)
            # get the TD error of the simulation
            sim_delta = reward + gamma * sim_value(obs).detach() - sim_value(prev_obs).detach() # detach since we do semi-gradient TD
            # need to reroll probability
                # complication: have now updated the policy, but action was taken with respect to previous policy
                # do we use the log prob of the simulated policy assuming this action was taken?
            sim_log_probs = sim_policy(prev_obs)
            sim_m = Categorical(logits = sim_log_probs)

            # losses of the simulation
            # TD loss is (r + gamma * (v(curr_state, old_weight) - alpha(old_weight, grad) * grad * curr_state) - (v(old_state, old_weight) - alpha(old_weight, grad) * grad * old_state)

            param_and_grad0 = torch.cat([value_params[0], value_grads[0]], dim = -1)
            param_and_grad1 = torch.cat([value_params[1], value_grads[1]], dim = -1)

            #print(torch.mul(value_lrs[0](param_and_grad0), value_grads[0]).shape)

            updated_value_curr = torch.dot( torch.mul(value_lrs[0](param_and_grad0), value_grads[0])[0], obs ) + torch.mul(value_lrs[1](param_and_grad1), value_grads[1])
            updated_value_prev = torch.dot( torch.mul(value_lrs[0](param_and_grad0), value_grads[0])[0], prev_obs ) + torch.mul(value_lrs[1](param_and_grad1), value_grads[1])
            sim_value_loss = (reward + gamma * (value(obs).detach() - updated_value_curr) - (value(prev_obs).detach() - updated_value_prev))

            #sim_policy_loss.backward(retain_graph = True)
            sim_value_loss.backward()
            # gradient step of the step-sizes
            with torch.no_grad():
                for lrs in value_lrs:
                    for param in lrs.parameters():
                        #print(lr)
                        param -= lr * param.grad
                        #print(param[0].detach().numpy()[0])
                #for lrs in policy_lrs:
                #    for param in lrs.parameters():
                #        param -= lr * param.grad

            # zero grad again because we used them in the simulation
            policy.zero_grad()
            value.zero_grad()

            # zero all lr grads
            for lrs in policy_lrs:
                lrs.zero_grad()
            for lrs in value_lrs:
                lrs.zero_grad()

            delta = reward + gamma * value(obs).detach() - value(prev_obs).detach()
            policy_loss = -(gamma ** timestep) * delta * m.log_prob(action)
            value_loss = -delta * value(torch.Tensor(prev_obs).to(device))

            policy_loss.backward(retain_graph = True)
            value_loss.backward()

            #if ix > 0:
            #    lr_loss.backward()

            timestep += 1 # tracker for discounting


            # optimize policy and value function
            with torch.no_grad():
                for ix, param in enumerate(policy.parameters()):
                    #print(torch.cat([param.view(1, -1), param.grad.view(1, -1)], dim = -1).shape)
                    weight_and_grad = torch.cat([param.view(1, -1), param.grad.view(1, -1)], dim = -1)
                    #print(weight_and_grad.shape)
                    #param.data -= torch.mul(policy_lrs[ix](weight_and_grad).view(param.grad.shape), param.grad)
                    param -= lr * param.grad
                for ix, (name, param) in enumerate(value.named_parameters()):
                    weight_and_grad = torch.cat([param.view(1, -1), param.grad.view(1, -1)], dim = -1)
                    param.data -= torch.mul(value_lrs[ix](weight_and_grad).view(param.grad.shape), param.grad)
                    #print(value_lrs[ix](weight_and_grad)[0].detach().numpy()[0])
                    for lr_ix, lr in enumerate(value_lrs[ix](weight_and_grad)[0].detach().numpy()):
                        adaptive_lr_sample["param_name"].append(name)
                        adaptive_lr_sample["index"].append(lr_ix)
                        adaptive_lr_sample["lr"].append(lr)
                        adaptive_lr_sample["episode"].append(episode)
                    #adaptive_lr_sample[ix].append(value_lrs[ix](weight_and_grad)[0].detach().numpy()[0])
                    #adaptive_lr_sample[name].append(name)


        #print("total return of {} at end of episode {}".format(final_return, episode + 1))
        returns.append(final_return)
    df = pd.DataFrame(adaptive_lr_sample)
    #print(df)

    ## uncomment the following to plot the lr's 
    #seaborn.lineplot(x = "episode", y = "lr", style = "param_name", hue = "index", data = df, legend = "full")
    #plt.show()
    return returns

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
