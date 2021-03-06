from includes import *
from algs import *

def gvf_trace_update(optim, gvf, cumulant, termination, obs, action, prev_obs, lambda_t, prev_z):
    optim.zero_grad()
    delta = cumulant(obs, action) + (1 - termination(obs).detach()) * gvf(obs).detach() - gvf(prev_obs).detach()
    #z = (1 - termination(obs).detach()) * lambda_t * prev_z + gvf(obs)
    z = gvf(obs)
    loss = -delta * z
    loss.backward()

    optim.step()

    return z.detach()

def gvf_off_policy_update(optim, gvf, cumulant, termination, obs, action, prev_obs, importance_ratio = 1):
    optim.zero_grad()
    delta = cumulant + (1 - termination) * gvf(obs).detach() - gvf(prev_obs).detach()
    loss = -importance_ratio * delta * gvf(obs)
    loss.backward()

    optim.step()

def aggregate_gvfs(gvfs, obs):
    # returns a tensor that is obs applied to each element of gvfs
    return torch.Tensor([gvf(obs) for gvf in gvfs]).to(device)

def train_off_policy_gvf(env, gamma, lr, episodes):
    dim_action = env.action_space.n
    dim_state = env.observation_space.shape[0]

    # off-policy
    # learning multiple target policies from one behaviour policy
    # the target policies are the policies for GVFs, which are then used as features for the behaviour policy

    random_policy = Policy(dim_state, dim_action)
    random_gvf = Time_GVF(dim_state) # gvf corresponding to random policy
    random_gvf_optim = torch.optim.SGD(random_gvf.parameters(), lr = lr)

    left_gvf = Time_GVF(dim_state) # GVF for policy of going left
    left_gvf_optim = torch.optim.SGD(left_gvf.parameters(), lr = lr)

    right_gvf = Time_GVF(dim_state) # GVF for policy of going right
    right_gvf_optim = torch.optim.SGD(right_gvf.parameters(), lr = lr)


    policy = Policy(dim_state, dim_action)
    policy_optim = torch.optim.SGD(policy.parameters(), lr = lr)


    # termination factor is just 0 unless we are at a terminal state
    returns = []



    for episode in range(episodes):
        obs = torch.Tensor(env.reset()).to(device)
        done = False

        # cumulant is timestep
        timestep = 0

        final_return = 0

        actions = []
        rewards = []
        states = [torch.Tensor(obs).to(device)]

        while done != True:
            env.render()

            log_probs = policy(obs)
            m = Categorical(logits = log_probs)
            action = m.sample()

            prev_obs = torch.Tensor(obs).to(device)

            obs, reward, done, info = env.step(action.item())

            rewards.append(reward)
            actions.append(action.item())

            obs = torch.Tensor(obs).to(device)

            states.append(obs)

            final_return += reward * (gamma ** timestep)

            # optimize gvf's online
            # if episode is not ended, cumulant is 1
            # termination factor is 0 ==> the pseudo-discount factor is 1 so we do not discount

            # NOTE: this is hardcoded for cartpole
            # TODO: have one GVF corresponding to taking an action exclusively

            true_prob = torch.exp(log_probs[action.item()].detach())
            left_prob = 1 if action.item() == 0 else 0
            right_prob = 1 if action.item() == 1 else 0
            random_prob = random_policy(prev_obs).detach()[action.item()]

            gvf_off_policy_update(left_gvf_optim, left_gvf, 1, 0, obs, actions[-1], prev_obs, left_prob / true_prob)
            gvf_off_policy_update(right_gvf_optim, right_gvf, 1, 0, obs, actions[-1], prev_obs, right_prob / true_prob)
            gvf_off_policy_update(random_gvf_optim, random_gvf, 1, 0, obs, actions[-1], prev_obs, random_prob / true_prob)

            timestep += 1


        # REINFORCE
        for i in range(len(actions)):
            # optimize policy
            G = 0
            if i <= len(rewards) - 1:
                G = sum([(gamma ** j) * rewards[j] for j in range(i, len(rewards))])

            # don't need to use log since we already output logits
            elig_vec = policy(states[i])[actions[i]] # evaluate the gradient with the current params

            loss = -G * elig_vec
            policy_optim.zero_grad()
            if i + 1 == len(actions):
                loss.backward()
            else:
                loss.backward(retain_graph = True)
            policy_optim.step()
    return returns

def train_time_gvf(env, gamma, lr, episodes):
    dim_action = env.action_space.n
    dim_state = env.observation_space.shape[0]

    # only learn one GVF: from a state and given an action, predict the time until failure/episode ending (basically predicts falling)
    # will have to be monte carlo method, or at least that is most reliable

    # just use a fixed policy for now
    # TODO: in the future find way to make this adaptive, maybe with off-policy? have a set of policies that we would like to predict
    gvf = Time_GVF(dim_state)
    gvf_optim = torch.optim.SGD(gvf.parameters(), lr = lr)

    # TODO: should also try to learn off-policy predictions; this can probably help with the fact that our policy is always changing
        # how do options fit in?


    policy = Policy(dim_state, dim_action)
    policy_optim = torch.optim.SGD(policy.parameters(), lr = lr)


    # termination factor is just 1 unless we are at a terminal state

    returns = []

    # for assessing accuracy of gvf later
    gvf_assess = {"episode": [], "gvf_pred": [], "true_time": [], "timestep": []}
    gvf_errors = {"episode": [], "error": []} # holds mean squared error of gvf prediction from actual time till end at a given step

    # for plots
    fig = plt.figure(figsize = (15, 10))


    for episode in range(episodes):
        gvf_error = 0
        obs = torch.Tensor(env.reset()).to(device)
        done = False

        gvf_preds = []

        # cumulant is timestep
        timestep = 0

        final_return = 0

        actions = []
        rewards = []
        states = [torch.Tensor(obs).to(device)]

        while done != True:
            env.render()

            gvf_assess["episode"].append(episode)
            gvf_assess["gvf_pred"].append(gvf(obs).detach().item())
            gvf_assess["timestep"].append(timestep)

            gvf_preds.append(gvf(obs).detach().item())


            log_probs = policy(obs)
            m = Categorical(logits = log_probs)
            action = m.sample()

            prev_obs = torch.Tensor(obs).to(device)

            obs, reward, done, info = env.step(action.item())

            rewards.append(reward)
            actions.append(action.item())

            obs = torch.Tensor(obs).to(device)

            states.append(obs)

            final_return += reward * (gamma ** timestep)
            timestep += 1

        gvf_assess["episode"].append(episode)
        gvf_assess["gvf_pred"].append(gvf(obs).detach().item())
        gvf_assess["timestep"].append(timestep)

        gvf_preds.append(gvf(obs).detach().item())

        # average squared error
        gvf_error = np.sqrt(sum([(gvf_preds[i] - (timestep - i)) ** 2 for i in range(timestep + 1)])) / (timestep + 1)

        gvf_errors["episode"].append(episode)
        gvf_errors["error"].append(gvf_error)

        # REINFORCE
        for i in range(len(actions)):
            # optimize policy
            G = 0
            if i <= len(rewards) - 1:
                G = sum([(gamma ** j) * rewards[j] for j in range(i, len(rewards))])

            # don't need to use log since we already output logits
            elig_vec = policy(states[i])[actions[i]] # evaluate the gradient with the current params

            loss = -G * elig_vec
            policy_optim.zero_grad()
            if i + 1 == len(actions):
                loss.backward()
            else:
                loss.backward(retain_graph = True)
            policy_optim.step()

        # optimize GVFs
        # recall cumulant is just 1 at non-terminal state
        # termination factor is 0 until we hit a terminal state, so the return is given below
        # for each state in the buffer, the return at that stage is simply len(buffer) - 1 - index of state since the terminal state is in the buffer
        cumulants = [len(states) - 1 - ix for ix in range(len(states))]

        gvf_assess["true_time"] += cumulants

        # possible problem: cumulants can grow very large. can we normalize?
        for i in range(len(states)):
            gvf_optim.zero_grad()
            delta = cumulants[i] - gvf(states[i]).detach()
            gvf_loss = -delta * gvf(states[i])
            gvf_loss.backward()
            gvf_optim.step()

        # plot accuracy of gvf for final episode

        #plt.show()
        returns.append(final_return)

    # show combined plots of accuracy of gvf
    #gvf_assess = pd.DataFrame(gvf_assess)
    #seaborn.lineplot(x = "timestep", y = "gvf_pred", data = gvf_assess.loc[gvf_assess["episode"] == episode - 1], label = "gvf")
    #seaborn.lineplot(x = "timestep", y = "true_time", data = gvf_assess.loc[gvf_assess["episode"] == episode - 1], label = "truth")

    #gvf_errors = pd.DataFrame(gvf_errors)
    #seaborn.lineplot(x = "episode", y = "error", data = gvf_errors)

    #plt.show()
    return returns



def train_gvf(env, gamma, lambda_t, episodes, lr):
    dim_action = env.action_space.n
    dim_state = env.observation_space.shape[0]

    # each gvf will be a state feature
    # number of states is dim_state just to be simple so that we can plug into the policy
    gvfs = [Value(dim_state) for i in range(dim_state)]
    terminations = [Termination(dim_state) for i in range(dim_state)]
    cumulants = [Cumulant(dim_state, dim_action) for i in range(dim_state)]
    gvf_optims = [torch.optim.SGD(gvfs[i].parameters(), lr = lr) for i in range(dim_state)]

    # try a general value function network (RNN with GVFs?)
    gvf = Value(dim_state)
    termination = Termination(dim_state)
    cumulant = Cumulant(dim_state, dim_action) # also try features themselves as cumulants
    policy = Policy(dim_state, dim_action)

    policy_optim = torch.optim.SGD(policy.parameters(), lr = lr)

    optim = torch.optim.SGD(gvf.parameters(), lr = lr)

    returns = []


    for episode in range(episodes):
        obs = torch.Tensor(env.reset()).to(device)
        done = False

        timestep = 0
        final_return = 0

        actions = []
        rewards = []
        states = [torch.Tensor(obs).to(device)]

        prev_z = torch.Tensor([0] * dim_state).to(device)

        while done != True:
            env.render()

            # QUESTION: Is there some way to only use gvf's that have been determined to be accurate?
                # regularization?

            # how to determine accuracy of a GVF?
            log_probs = policy(aggregate_gvfs(gvfs, obs))
            m = Categorical(logits = log_probs)
            action = m.sample()

            prev_obs = torch.Tensor(obs).to(device)

            obs, reward, done, info = env.step(action.item())

            rewards.append(reward)
            actions.append(action.item())

            obs = torch.Tensor(obs).to(device)

            states.append(obs)

            final_return += reward * (gamma ** timestep)


            # how to use GVFs for a policy?
            # mb have GVFs as features to start?
            #prev_z = gvf_update(optim, gvf, cumulant, termination, obs, torch.Tensor([action]).to(device), prev_obs, lambda_t, prev_z)

            # for now, don't do eligibity traces
            #gvf_update(optim, gvf, cumulant, termination, obs, torch.Tensor([action]).to(device), prev_obs, lambda_t, prev_z)
            for i in range(dim_state):
                gvf_trace_update(gvf_optims[i], gvfs[i], cumulants[i], terminations[i], obs, torch.Tensor([action]).to(device), prev_obs, lambda_t, prev_z)

        # REINFORCE
        for i in range(len(actions)):
            G = 0
            if i <= len(rewards) - 1:
                G = sum([(gamma ** j) * rewards[j] for j in range(i, len(rewards))])

            # don't need to use log since we already output logits
            elig_vec = policy(states[i])[actions[i]] # evaluate the gradient with the current params

            loss = -G * elig_vec
            #assert np.isnan(G.detach()) == False, "G is nan"
            #assert torch.sum(torch.isnan(elig_vec)) == 0, "elig vec is nan"
            #assert np.isnan(loss.detach()) == False, "loss is nan"
            policy_optim.zero_grad()
            if i + 1 == len(actions):
                loss.backward()
            else:
                loss.backward(retain_graph = True)
            policy_optim.step()

        returns.append(final_return)
    return returns


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
