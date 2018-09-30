import math
import random
from itertools import starmap, product, islice, ifilter, repeat, izip
from scipy.sparse import dok_matrix

import numpy as np
from collections import defaultdict

from toolz import first

ftype = np.float32

class VV_MDP:
    def __init__(self,
                 _startingstate,
                 _transitions,  # dictionary of key:values   (s,a,s):proba
                 _rewards,  # dictionary of key:values   s: vector of rewards
                 _gamma=.9):
        try:
            states = sorted(
                {st for (s, a, s2) in _transitions.iterkeys() for st in (s, s2)}
            )
            actions = sorted(
                {a for (s, a, s2) in _transitions.iterkeys()}
            )

            n , na = len(states) , len(actions)

            stateInd = {s: i for i, s in enumerate(states)}
            actionInd = {a: i for i, a in enumerate(actions)}

            assert set(_startingstate).issubset(stateInd.keys()), \
                "initial states are not subset of total states"

            self.startingStateInd = [stateInd[x] for x in _startingstate]

            #d = len(_rewards[first(_rewards.iterkeys())])
            #assert all(d == len(np.array(v,dtype=ftype)) for v in _rewards.itervalues()),\
            #       "incorrect reward vectors"
            #assert set(_rewards.keys()).issubset(states) ,\
            #       "states appearing in rewards should also appear in transitions"

        except ValueError,TypeError:

            print("transitions or rewards do not have the correct structure")
            raise

        self.states = states
        self.actions = actions
        self.transitions = _transitions
        self.stateInd = stateInd
        self.actionInd = actionInd


    def _get_Actions(self, _state):
        return set(self.actionInd[a] for (s,a,s2) in self.transitions if s == _state)



def random_mdp(n_states, n_actions, _r=None):
    """ Builds a random MDP.
        Each state has ceil(log(nstates)) successors.
        Reward vectors are permutations of [1,0,...,0]
    """

    nsuccessors = int(math.ceil(math.log1p(n_states)))
    gauss_iter = starmap(random.gauss,repeat((0.5,0.5)))
    _t = {}

    for s,a in product(range(n_states), range(n_actions)):
        next_states = random.sample(range(n_states), nsuccessors)
        probas =  np.fromiter(islice(ifilter(lambda x: 0 < x < 1 ,gauss_iter),nsuccessors), ftype)

        _t.update({(s,a,s2):p for s2,p in izip(next_states, probas/sum(probas))})

    #if _r is None:
    #    _r = {i:np.random.permutation([1]+[0]*(len(_lambda)-1)) for i in range(n_states)}

    #assert len(_r[0])==len(_lambda),"Reward vectors should have same length as lambda"

    return VV_MDP(
        _startingstate= set(range(n_states)),
        _transitions= _t,
        _rewards= _r ,
        _gamma= 0.95)


