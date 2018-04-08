from scipy.sparse import dok_matrix

import numpy as np
import random
from itertools import starmap, repeat, product, islice, ifilter, izip

ftype = np.float32

import math

class MDP:
    def __init__(self, _startingstate, _transitions, _rewards, _gamma):

        try:
            states = sorted(
                {st for (s, a, s2) in _transitions.iterkeys() for st in (s, s2)}
            )
            actions = sorted(
                {a for (s, a, s2) in _transitions.iterkeys()}
            )

            stateInd = {s: i for i, s in enumerate(states)}
            actionInd = {a: i for i, a in enumerate(actions)}

            assert set(_startingstate).issubset(stateInd.keys()), \
            "initial states are not subset of total states"

            self.startingStateInd = [stateInd[x] for x in _startingstate]

            assert set(_rewards.keys()).issubset(list(product(states, actions))),\
                "states appearing in rewards should also appear in transitions"

        except ValueError, TypeError:

            print("transitions or rewards do not have the correct structure")
            raise


        self.states = states
        self.actions = actions
        self.nstates = len(states)
        self.nactions = len(actions)

        self.stateInd = stateInd
        self.actionInd = actionInd
        self.rewards = _rewards
        self.gamma = _gamma

        """for generating the transition function as a |S||A|x|S|"""
        #empty sparse matrix for transition function
        transitions = np.array([[dok_matrix((1, self.nstates), dtype=ftype) for _ in self.actions] for _ in self.states], dtype=object)

        for (s, a, s2), p in _transitions.iteritems():
            si, ai, si2 = self.stateInd[s], self.actionInd[a], self.stateInd[s2]
            transitions[si, ai][0, si2] = p

        for s, a in product(range(self.nstates), range(self.nactions)):
            transitions[s,a] = transitions[s,a].tocsr()
            assert 0.99 <= transitions[s,a].sum() <= 1.01, "probability transitions should sum up to 1"+ str(transitions[s,a])

        self.transitions = transitions

        # E_test = np.zeros((nstates*nactions, nstates), dtype=ftype)
        E = dok_matrix((self.nstates * self.nactions, self.nstates), dtype=ftype)

        for s in range(self.nstates):
            for a in range(self.nactions):
                E[s*self.nactions+a, :] = [self.transitions[s,a][0,i] for i in range(self.nstates) ]
                #E[s * self.nactions + a, s] -= 1
                E[s * self.nactions + a, s] -= 1.0/self.gamma

        self.E = E
        self.alpha = [np.float32(1.0/self.nstates)]*self.nstates

    def display_mdp(self):

        print 'sates = ', self.states
        print 'actions = ', self.actions
        print 'gamma =', self.gamma
        print 'rewards', self.rewards

        for s in range(self.nstates):
            for a in range(self.nactions):
                print [ 'P('+ str(i) + '|'+ str(s) + ',' + str(a) +') ='+ str(self.transitions[s, a][0, i]) for i in range(self.nstates)]

        pass

    def modify_mdp(self):
        for i in range(self.nstates):
            self.alpha[i] = 0.0
        self.alpha[0] = 1.0

        # self.transitions[0, 0][0, 0] = 0.33
        # self.transitions[0, 0][0, 1] = 0.66
        # #******
        # self.transitions[0, 1][0, 0] = 0.05
        # self.transitions[0, 1][0, 2] = 0.95
        # # # ******
        # self.transitions[1, 0][0, 0] = 0.66
        # self.transitions[1, 0][0, 1] = 0.33
        # self.transitions[1, 0][0, 2] = 0.0
        # # # ******
        # self.transitions[1, 1][0, 1] = 0.9
        # self.transitions[1, 1][0, 2] = 0.1
        # # # ******
        # # self.transitions[2, 0][0, 0] = 0.66
        # # self.transitions[2, 0][0, 2] = 0.33
        # # # ******
        # self.transitions[2, 1][0, 1] = 0.66
        # self.transitions[2, 1][0, 2] = 0.33


        pass

def general_random_mdp(n_states, n_actions, _gamma, _reward_lb, _reward_up):
    """ Builds a random MDP.
        Each state has ceil(log(nstates)) successors.
        Reward are random values between 0 and 1
    """
    nsuccessors = int(math.ceil(math.log1p(n_states)))
    gauss_iter = starmap(random.gauss,repeat((0.5,0.5)))
    _t = {}
    _r = {}

    for s, a in product(range(n_states), range(n_actions)):
        next_states = random.sample(range(n_states), nsuccessors)
        probas = np.fromiter(islice(ifilter(lambda x: 0 < x < 1 ,gauss_iter),nsuccessors), ftype)

        _t.update({(s,a,s2):p for s2,p in izip(next_states, probas/sum(probas))})
        #_r.update({(s,a):r for r in np.random.uniform(-600.,600,1)})
        _r.update({(s, a): r for r in np.random.uniform(_reward_lb, _reward_up, 1)})

    return MDP(
        _startingstate= set(range(n_states)),
        _transitions= _t,
        _rewards= _r ,
        _gamma = 0.95)

    pass


def general_random_mdp_rounded(n_states, n_actions, _gamma, _reward_lb, _reward_up):
    """ Builds a random MDP.
        Each state has ceil(log(nstates)) successors.
        Reward are random values between 0 and 1
    """
    nsuccessors = int(math.ceil(math.log1p(n_states)))
    _t = {}
    _r = {}

    for s, a in product(range(n_states), range(n_actions)):
        next_states = random.sample(range(n_states), nsuccessors)

        while True:
            t = random.sample([0.0, 0.33, 0.5, 0.66, 1.0], nsuccessors)
            if sum(t) < 1.01 and sum(t) > 0.99:
                probas =  t
                break

        _t.update({(s,a,s2):p for s2,p in izip(next_states, probas)})
        #_r.update({(s,a):r for r in np.random.uniform(-600.,600,1)})
        _r.update({(s, a): r for r in np.random.uniform(_reward_lb, _reward_up, 1)})

    print '_t', _t
    return MDP(
        _startingstate= set(range(n_states)),
        _transitions= _t,
        _rewards= _r ,
        _gamma = 0.95)

    pass

def grid_MDP(rows, columns, start=None, goal=None):

    actions = {0:'w', 1:'nw', 2: 'n', 3: 'ne', 4: 'e', 5: 'se', 6: 's', 7: 'sw', 8: 'stay'}
    actions_grid = {'w':[-1, 0], 'nw':[-1,1], 'n':(0, 1), 'ne':(1,1), 'e': (1,0), 'se':(1,-1), 's':(0,-1), 'sw':(-1,-1), 'stay':(0,0)}
    state_grid = np.zeros((rows, columns))

    n_states = rows*columns
    _t = {}
    _r = {}


    x, y, x_, y_ = None, None, None, None
    if start is None:
        x = random.choice(range(rows))
        y = random.choice(range(columns))
        start = [x, y]

    if goal is None:
        while True:
            x_ = random.choice(range(rows))
            y_ = random.choice(range(columns))
            if x_ != x and y != y_:
                goal = (x_, y_)
                break


    for i in range(rows):
        for j in range(columns):
            s = i*rows+columns
            _t.update({(s, a, s2) for a in actions.iterkeys()})

            pass

    pass

grid_MDP(3,3)

