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


## For GVFs
class Cumulant(nn.Module):
    def __init__(self, dim_state, dim_action):
        # for now, have cumulant depend upon (s, a) since we don't have access to next state s' if we're doing this online
        super(Cumulant, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_state + 1, 1), # add one since action is one-dimensional
            nn.Sigmoid()
        )

    def forward(self, s, a):
        #print(torch.cat([s, a], dim = -1))
        return self.layers(torch.cat([s, a], dim = -1))

class Termination(nn.Module):
    # gives a termination factor given a state
    # we don't necessarily want to optimize this at every time step, but we subclass nn.Module just in case
    def __init__(self, dim_state):
        super(Termination, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(dim_state, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # REMEMBER: round gives zero gradient, so change if we want the pseudotermination factor to change
        return torch.round(self.layers(x))

# TODO: GAN for planning


# TODO: Flow-based model for planning
# allocate more layers/resources depending on model error?



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
