from includes import *

## implements a PSR
class PSR():
    def __init__(self, q):
        self.q = q

## implementation of a GVF
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
