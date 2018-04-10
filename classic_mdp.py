from scipy.sparse import dok_matrix

import numpy as np
import random
from operator import add
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
    actions_grid = {'w':[-1, 0], 'nw':[-1,1], 'n':[0, 1], 'ne':[1,1], 'e': [1,0], 'se':[1,-1], 's':[0,-1], 'sw':[-1,-1], 'stay':[0,0]}
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
                goal = [x_, y_]
                break

    #random.uniform(0.0, 0.2)
    for i in range(rows):
        for j in range(columns):
            s = i*rows+j
            next_states = {}

            p_ = {}
            for a in actions.iterkeys():
                dir = actions_grid[actions[a]]
                res = [i+dir[0], j+dir[1]]

                check = False
                if (0 <= res[0] <= rows-1) and (0<= res[1] <= columns-1):
                    check = True
                    next_states[a] = res

                if check is True:
                    p_[a] = random.uniform(0.0, 0.2)
                    _t.update({(s, a, next_states[a][0]*rows+next_states[a][1]): 1-p_[a]})

            for a in p_.iterkeys():
                next = random.choice([k for k in next_states.iterkeys() if k != a ])
                next_s = next_states[next]
                _t.update({(s, a, next_s[0]*rows+next_s[1]): p_[a]})

    unknow_rewards = []

    grid_size = rows*columns
    print grid_size

    # number of states with unknown rewards
    num_states_ur = random.randint(int(40.0*grid_size/100.0), int(60*grid_size/100.0))
    while len(unknow_rewards)< num_states_ur:
        elem = random.randint(0,grid_size-1)
        if elem not in unknow_rewards:
            unknow_rewards.append(elem)

    # rewards for states with unknown rewards
    for s in unknow_rewards:
        for a in range(9):
            _r.update({(s, a): r for r in np.random.uniform(35, 50, 1)})

    #reward for the goal state
    for a in range(9):
        _r.update({(goal[0]*rows+goal[1], a): np.random.uniform(80, 1000, 1)})

    #the rest of states
    rest_states = [x for x in range(rows*columns) if x not in unknow_rewards+[goal[0]*rows+goal[1]] ]

    for s in rest_states:
        for a in range(9):
            _r.update({(s, a): 50.0})

    return MDP(
        _startingstate= set(range(grid_size)),
        _transitions= _t,
        _rewards= _r ,
        _gamma = 0.95)

def index_level_to_state(level, index, half_level):

     if level <= half_level:
        return sum(pow(2, i) for i in range(level))+ index
     else:
         return  sum(pow(2, i) for i in range(half_level+1)) + sum(pow(2, i) for i in range(half_level-1, 2*half_level-level, -1))+ index

def state_to_index_level(state):

    pass

def to_left_child(state, half_level):



    pass

def to_right_child(state, half_level):
    pass

def get_father(state, half_level):
    pass

def diamond_mdp(half_level):

    n_states = sum(pow(2, i) for i in range(half_level+1)) + sum(pow(2, i) for i in range(half_level))
    state_level_index = {}
    pass

#%%%%%%%%%%%%%%%%%%%
def create_diamond_MDP(half_level):
    if half_level == 2:
        return diamond_mdp_2()

def diamond_mdp_2():

    n_states= 9
    n_actions = 3

    _t = {}
    _r = {}

    state_dict = {0:(0, 0), 1: (1,0), 2: (1,1), 3: (2,0), 4: (2,1), 5: (2,2), 6:(2,3), 7: (3,0), 8: (3,1), 9: (4,0)}
    actions = {0: 'a0', 1: 'a1', 2: 'a2'}

    # action a0
    _t.update({(0, 0, 1):0.5, (0,0,2): 0.5})
    _t.update({(1,0,3): 0.5, (1, 0, 4):0.5, (2, 0, 5): 0.5, (2, 0, 6): 0.5})
    _t.update({(3,0,7):0.5, (3, 0,8): 0.5, (4,0,7):0.5, (4, 0,8): 0.5, (5,0,7):0.5, (5, 0,8): 0.5, (6,0,7):0.5, (6, 0,8): 0.5 })
    _t.update({(7, 0, 9):1.0, (8, 0, 9): 1.0})

    # action a1
    _t.update({(0, 1, 1):0.3, (0,1,0): 0.7, (1,1 ,3 ):0.3, (1,1 ,0 ):0.7, (2, 1, 5):0.3, (2, 1,0 ):0.7})
    _t.update({(3,1 ,8 ):0.3, (3, 1, 1):0.7, (4, 1,7 ):0.3, (4,1 ,1 ):0.7, (5,1 ,7 ):0.3, (5,1 ,2 ):0.7, (6,1 ,8 ):0.3, (6, 1,2 ):0.7})
    _t.update({(7,1 ,9 ):0.3, (7,1 ,3 ):0.7, (8, 1, 9):0.3, (8,1 ,5 ):0.7})

    # action a2
    _t.update({(0, 2, 2):0.7, (0,2,0): 0.3, (1,2 ,4 ):0.7, (1,2 ,0 ):0.3, (2, 2, 6):0.7, (2, 2,0 ):0.3})
    _t.update({(3,2 ,7 ):0.7, (3, 2, 1):0.3, (4, 2,8 ):0.7, (4,2 ,1 ):0.3, (5,2 ,8 ):0.7, (5,2 ,2 ):0.3, (6,2 ,7 ):0.7, (6, 2,2 ):0.3})
    _t.update({(7,2 ,9 ):0.7, (7,2 ,4 ):0.3, (8, 2, 9):0.7, (8,2 ,6 ):0.3})

    

    pass



#grid_MDP(3,3)
create_diamond_MDP(2)
