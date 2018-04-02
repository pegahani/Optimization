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
            assert 0.99 <= transitions[s,a].sum() <= 1.01, "probability transitions should sum up to 1"

        self.transitions = transitions

        # E_test = np.zeros((nstates*nactions, nstates), dtype=ftype)
        E = dok_matrix((self.nstates * self.nactions, self.nstates), dtype=ftype)

        for s in range(self.nstates):
            for a in range(self.nactions):
                E[s*self.nactions+a, :] = [self.transitions[s,a][0,i] for i in range(self.nstates) ]
                E[s * self.nactions + a, s] -= 1

        self.E = E
        self.alpha = [np.float32(1.0/self.nstates)]*self.nstates

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
