from scipy.sparse import dok_matrix

import numpy as np
import random
from operator import add
from itertools import starmap, repeat, product, islice, ifilter, izip

ftype = np.float32

import math

class MDP:
    def __init__(self, _startingstate, _transitions, _rewards_bounds, _gamma, _alpha = None):

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

            assert set(_rewards_bounds.keys()).issubset(list(product(states, actions))), \
                "states appearing in rewards should also appear in transitions"

            # assert set(_rewards.keys()).issubset(list(product(states, actions))),\
            #     "states appearing in rewards should also appear in transitions"

        except ValueError, TypeError:

            print("transitions or rewards do not have the correct structure")
            raise


        self.states = states
        self.actions = actions
        self.nstates = len(states)
        self.nactions = len(actions)

        self.stateInd = stateInd
        self.actionInd = actionInd
        #self.rewards = _rewards
        self.rewards_bounds = _rewards_bounds
        self.gamma = _gamma

        """for generating the transition function as a |S||A|x|S|"""
        #empty sparse matrix for transition function
        transitions = np.array([[dok_matrix((1, self.nstates), dtype=ftype) for _ in self.actions] for _ in self.states], dtype=object)

        for (s, a, s2), p in _transitions.iteritems():
            si, ai, si2 = self.stateInd[s], self.actionInd[a], self.stateInd[s2]
            transitions[si, ai][0, si2] = p

        #for s, a in product(range(self.nstates), range(self.nactions)):
        #    transitions[s,a] = transitions[s,a].tocsr()
        #    assert 0.99 <= transitions[s,a].sum() <= 1.01, "probability transitions should sum up to 1"+ str(transitions[s,a])

        self.transitions = transitions

        # E_test = np.zeros((nstates*nactions, nstates), dtype=ftype)
        E = dok_matrix((self.nstates * self.nactions, self.nstates), dtype=ftype)

        for s in range(self.nstates):
            for a in range(self.nactions):
                E[s*self.nactions+a, :] = [self.transitions[s,a][0,i] for i in range(self.nstates) ]
                #E[s * self.nactions + a, s] -= 1
                E[s * self.nactions + a, s] -= 1.0/self.gamma

        self.E = E
        if _alpha is None:
            self.alpha = [np.float32(1.0/self.nstates)]*self.nstates
        else:
            self.alpha = _alpha

    def display_mdp(self):

        print 'sates = ', self.states
        print 'actions = ', self.actions
        print 'gamma =', self.gamma
        print 'rewards', self.rewards_bounds

        for s in range(self.nstates):
            for a in range(self.nactions):
                print [ 'P('+ str(i) + '|'+ str(s) + ',' + str(a) +') ='+ str(self.transitions[s, a][0, i]) for i in range(self.nstates)]

        pass

    def modify_mdp(self):
        for i in range(self.nstates):
            self.alpha[i] = 0.0
        self.alpha[0] = 1.0
        pass

def general_random_mdp(n_states, n_actions, _gamma, _reward_lb, _reward_up, reward_on_state):
    """ Builds a random MDP.
        Each state has ceil(log(nstates)) successors.
        Reward are random values between 0 and 1
    """
    nsuccessors = int(math.ceil(math.log1p(n_states)))
    gauss_iter = starmap(random.gauss,repeat((0.5,0.5)))
    _t = {}
    _r = {}

    for s in range(n_states):
        rewards = np.random.uniform(_reward_lb, _reward_up, 2)
        for a in range(n_actions):
            next_states = random.sample(range(n_states), nsuccessors)
            probas = np.fromiter(islice(ifilter(lambda x: 0 < x < 1, gauss_iter), nsuccessors), ftype)

            _t.update({(s, a, s2): p for s2, p in izip(next_states, probas / sum(probas))})

            if reward_on_state:
                lb = np.float32(min(rewards))
                up = np.float32( max(rewards))
                _r.update({(s, a): [lb, up]})
            else:
                #_r.update({(s,a):r for r in np.random.uniform(-600.,600,1)})
                #_r.update({(s, a): r for r in np.random.uniform(_reward_lb, _reward_up, 1)})
                _r.update({(s, a): [_reward_lb, _reward_up]})


    return MDP(
        _startingstate= set(range(n_states)),
        _transitions= _t,
        _rewards_bounds= _r ,
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
        _r.update({(s, a): [_reward_lb, _reward_up]})

    print '_t', _t
    return MDP(
        _startingstate= set(range(n_states)),
        _transitions= _t,
        _rewards_bounds= _r ,
        _gamma = 0.95)

    pass

######### GRID MDP ###############
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

#%%%%%%%DIAMOND%%%%%%%%%%%%
def create_diamond_MDP(half_level,probab_low , probab_high,  reward_type):
    if half_level == 2:
        return diamond_mdp_2(probab_low , probab_high,  reward_type)
    if half_level == 3:
        return diamond_mdp_3(probab_low , probab_high,  reward_type)

def diamond_mdp_3(probab_low , probab_high,  reward_type):
    n_states = 22
    n_actions = 3

    state_dict = {0:(0, 0), 1: (1,0), 2: (1,1), 3: (2,0), 4: (2,1), 5: (2,2), 6:(2,3), 7: (3,0), 8: (3,1), 9: (3,2), 10: (3,3), 11: (3,4), 12: (3,5),
    13: (3,6), 14: (3,7), 15: (4,0), 16: (4,1), 17: (4,2), 18: (4,3), 19: (5,0), 20: (5,1), 26 : (6,0)}

    actions = {0: 'a0', 1: 'a1', 2: 'a2'}
    _t = {}

    #action a_0
    _t.update({(0, 0, 1):0.5, (0,0,2): 0.5})
    _t.update({(1,0,3): 0.5, (1, 0, 4):0.5, (2, 0, 5): 0.5, (2, 0, 6): 0.5})
    _t.update({(3,0,7):0.5, (3, 0,8): 0.5, (4,0,9):0.5, (4, 0,10): 0.5, (5,0,11):0.5, (5, 0,12): 0.5, (6,0,13):0.5, (6, 0,14): 0.5 })
    _t.update({(7,0,15):0.5, (7, 0,16): 0.5, (8,0,15):0.5, (8, 0,16): 0.5, (9,0,15):0.5, (9, 0,16): 0.5, (10,0,15):0.5, (10, 0,16): 0.5 })
    _t.update({(11,0,17):0.5, (11, 0,18): 0.5, (12,0,17):0.5, (12, 0,18): 0.5, (13,0,17):0.5, (13, 0,18): 0.5, (14,0,17):0.5, (17, 0,18): 0.5 })
    _t.update({(15,0,19):0.5, (15, 0,20): 0.5, (16,0,19):0.5, (16, 0,20): 0.5, (17,0,19):0.5, (17, 0,20): 0.5, (18,0,19):0.5, (18, 0,20): 0.5 })
    _t.update({(19, 0, 21):1.0, (20, 0, 21): 1.0})

    # action a1
    _t.update({(0, 1, 1):probab_low, (0,1,0): probab_high, (1,1 ,3 ):probab_low, (1,1 ,0 ):probab_high, (2, 1, 5):probab_low, (2, 1,0 ):probab_high})
    _t.update({(3,1 ,7 ):probab_low, (3, 1, 1):probab_high, (4, 1,9 ):probab_low, (4,1 ,1 ):probab_high, (5,1 ,11 ):probab_low, (5,1 ,2 ):probab_high, (6,1 ,13):probab_low, (6, 1,2 ):probab_high})
    _t.update({(7,1 ,18 ):probab_low, (7, 1, 3):probab_high, (8, 1,15 ):probab_low, (8,1 ,3 ):probab_high, (9,1 ,15 ):probab_low, (9,1 ,4 ):probab_high, (10,1 ,16):probab_low, (10, 1,4 ):probab_high})
    _t.update({(11, 1, 16):probab_low, (11,1,5): probab_high, (12,1 ,17 ):probab_low, (12,1 ,15 ):probab_high, (13, 1, 17):probab_low, (13, 1,6 ):probab_high, (14, 1, 18):probab_low, (14, 1,6 ):probab_high})
    _t.update({(15,1 ,20 ):probab_low, (15, 1, 8):probab_high, (16, 1,19 ):probab_low, (16,1 ,10 ):probab_high, (17,1 ,19 ):probab_low, (17,1 ,12 ):probab_high, (18,1 ,20):probab_low, (18, 1,14 ):probab_high})
    _t.update({(19,1 ,21 ):probab_low, (19,1,15 ):probab_high, (20, 1, 21):probab_low, (20,1 ,17):probab_high})

    # action a2
    _t.update({(0, 2, 2):probab_high, (0,2,0): probab_low, (1,2 ,4 ):probab_high, (1,2 ,0 ):probab_low, (2, 2, 6):probab_high, (2, 2,0 ):probab_low})
    _t.update({(3,2 ,8 ):probab_high, (3, 2, 1):probab_low, (4, 2,10):probab_high, (4,2 ,1):probab_low, (5,2 ,12 ):probab_high, (5,2 ,2 ):probab_low, (6,2 ,14 ):probab_high, (6, 2,2 ):probab_low})
    _t.update({(7,2 ,15 ):probab_high, (7, 2, 3):probab_low, (8, 2,16):probab_high, (8,2 ,3):probab_low, (9,2 ,16 ):probab_high, (9,2 ,4 ):probab_low, (10,2 ,17 ):probab_high, (10, 2,4 ):probab_low})
    _t.update({(11,2 ,17 ):probab_high, (11, 2, 5):probab_low, (12, 2,18):probab_high, (12,2 ,5):probab_low, (13,2 ,18 ):probab_high, (13,2 ,6 ):probab_low, (14,2 ,15):probab_high, (14, 2,6 ):probab_low})
    _t.update({(15,2 ,19 ):probab_high, (15, 2, 7):probab_low, (16, 2,20):probab_high, (16,2 ,9):probab_low, (17,2 ,20 ):probab_high, (17,2 ,11 ):probab_low, (18,2 ,19):probab_high, (18, 2,13):probab_low})
    _t.update({(19,2 ,21 ):probab_high, (19, 2, 16):probab_low, (20, 2,21):probab_high, (20,2 ,18):probab_low})


    _r_bounds = {}

    for s in range(7)+range(15,22,1):
        _r_bounds.update({(s, 0): [0.0, 0.0], (s, 1): [0.0, 0.0], (s, 2): [0.0, 0.0]})

    if reward_type == 0:
        # # U states

        for s in range(7,15, 1):
            _r_bounds.update({(s, 0): [-0.06, 0.06], (s, 1): [-0.06, 0.06], (s, 2): [-0.06, 0.06]})

        # goal
        _r_bounds.update({(21, 0): [0.06, 0.1], (21, 1): [0.06, 0.1], (21, 2): [0.06, 0.1]})

    else:
        print "THATS IT"
    
    print "len ", len(_r_bounds)
    print "_r_bounds \n" , _r_bounds        
    
    return MDP(
        _startingstate= set(range(n_states)),
        _transitions= _t,
        _rewards_bounds= _r_bounds,
        _gamma = 0.5, _alpha= [1.0]+21*[0.0])

def diamond_mdp_2(probab_low , probab_high,  reward_type):

    n_states= 9
    n_actions = 3

    _t = {}


    state_dict = {0:(0, 0), 1: (1,0), 2: (1,1), 3: (2,0), 4: (2,1), 5: (2,2), 6:(2,3), 7: (3,0), 8: (3,1), 9: (4,0)}
    actions = {0: 'a0', 1: 'a1', 2: 'a2'}

    # action a0
    _t.update({(0, 0, 1):0.5, (0,0,2): 0.5})
    _t.update({(1,0,3): 0.5, (1, 0, 4):0.5, (2, 0, 5): 0.5, (2, 0, 6): 0.5})
    _t.update({(3,0,7):0.5, (3, 0,8): 0.5, (4,0,7):0.5, (4, 0,8): 0.5, (5,0,7):0.5, (5, 0,8): 0.5, (6,0,7):0.5, (6, 0,8): 0.5 })
    _t.update({(7, 0, 9):1.0, (8, 0, 9): 1.0})

    # action a1
    _t.update({(0, 1, 1):probab_low, (0,1,0): probab_high, (1,1 ,3 ):probab_low, (1,1 ,0 ):probab_high, (2, 1, 5):probab_low, (2, 1,0 ):probab_high})
    _t.update({(3,1 ,8 ):probab_low, (3, 1, 1):probab_high, (4, 1,7 ):probab_low, (4,1 ,1 ):probab_high, (5,1 ,7 ):probab_low, (5,1 ,2 ):probab_high, (6,1 ,8 ):probab_low, (6, 1,2 ):probab_high})
    _t.update({(7,1 ,9 ):probab_low, (7,1 ,3 ):probab_high, (8, 1, 9):probab_low, (8,1 ,5 ):probab_high})

    # action a2
    _t.update({(0, 2, 2):probab_high, (0,2,0): probab_low, (1,2 ,4 ):probab_high, (1,2 ,0 ):probab_low, (2, 2, 6):probab_high, (2, 2,0 ):probab_low})
    _t.update({(3,2 ,7 ):probab_high, (3, 2, 1):probab_low, (4, 2,8 ):probab_high, (4,2 ,1 ):probab_low, (5,2 ,8 ):probab_high, (5,2 ,2 ):probab_low, (6,2 ,7 ):probab_high, (6, 2,2 ):probab_low})
    _t.update({(7,2 ,9 ):probab_high, (7,2 ,4 ):probab_low, (8, 2, 9):probab_high, (8,2 ,6 ):probab_low})

    _r_bounds = {}
    _r_bounds.update({(0,0):[0.0,0.0],(0,1): [0.0,0.0], (0,2): [0.0,0.0]})
    _r_bounds.update({(1,0):[0.0,0.0],(1,1): [0.0,0.0], (1,2): [0.0,0.0]})
    _r_bounds.update({(2,0):[0.0,0.0],(2,1): [0.0,0.0], (2,2): [0.0,0.0]})

    if reward_type == 0:

        # # U states

        _r_bounds.update({(3,0):  [-600, 600], (3, 1): [-600, 600], (3, 2): [-600, 600]} )
        _r_bounds.update({(4, 0): [-600, 600], (4, 1): [-600, 600], (4, 2): [-600, 600]})
        _r_bounds.update({(5, 0): [-600, 600], (5, 1): [-600, 600], (5, 2): [-600, 600]})
        _r_bounds.update({(6, 0): [-600, 600], (6, 1): [-600, 600], (6, 2): [-600, 600]})

        # goal

        _r_bounds.update({(9, 0): [600, 1000], (9, 1): [600, 1000], (9, 2): [600, 1000]})

    if reward_type == 1:

        # U states

        _r_bounds.update({(3, 0): [random.uniform(-600, 0), random.uniform(0, 600)],
                          (3, 1): [random.uniform(-600, 0), random.uniform(0, 600)],
                          (3, 2): [random.uniform(-600, 0), random.uniform(0, 600)]})
        _r_bounds.update({(4, 0): [random.uniform(-600, 0), random.uniform(0, 600)],
                          (4, 1): [random.uniform(-600, 0), random.uniform(0, 600)],
                          (4, 2): [random.uniform(-600, 0), random.uniform(0, 600)]})
        _r_bounds.update({(5, 0): [random.uniform(-600, 0), random.uniform(0, 600)],
                          (5, 1): [random.uniform(-600, 0), random.uniform(0, 600)],
                          (5, 2): [random.uniform(-600, 0), random.uniform(0, 600)]})
        _r_bounds.update({(6, 0): [random.uniform(-600, 0), random.uniform(0, 600)],
                          (6, 1): [random.uniform(-600, 0), random.uniform(0, 600)],
                          (6, 2): [random.uniform(-600, 0), random.uniform(0, 600)]})

        #goal

        _r_bounds.update({(9, 0): [random.uniform(600, 800), random.uniform(800, 10000)],
                          (9, 1): [random.uniform(600, 800), random.uniform(800, 10000)],
                          (9, 2): [random.uniform(600, 800), random.uniform(800, 10000)]})

    if reward_type == 2:

        # U states
        rew_bound = [random.uniform(-600, 0), random.uniform(0, 600)]
        _r_bounds.update({(3, 0): rew_bound,
                          (3, 1): rew_bound,
                          (3, 2): rew_bound})

        rew_bound = [random.uniform(-600, 0), random.uniform(0, 600)]
        _r_bounds.update({(4, 0): rew_bound,
                          (4, 1): rew_bound,
                          (4, 2): rew_bound})

        rew_bound = [random.uniform(-600, 0), random.uniform(0, 600)]
        _r_bounds.update({(5, 0): rew_bound,
                         (5, 1): rew_bound,
                         (5, 2): rew_bound})

        rew_bound = [random.uniform(-600, 0), random.uniform(0, 600)]

        _r_bounds.update({(6, 0): rew_bound ,
                          (6, 1): rew_bound ,
                          (6, 2): rew_bound })

        #goal

        goal_bound = [random.uniform(600, 800), random.uniform(800, 10000)]
        _r_bounds.update({(9, 0): goal_bound,
                          (9, 1): goal_bound,
                          (9, 2): goal_bound})

    if reward_type == 3:

        # U states

        _r_bounds.update({(3, 0): [random.uniform(-600, 0), random.uniform(0, 600)],
                          (3, 1): [random.uniform(-600, 0), random.uniform(0, 600)],
                          (3, 2): [random.uniform(-600, 0), random.uniform(0, 600)]})
        _r_bounds.update({(4, 0): [random.uniform(-600, 0), random.uniform(0, 600)],
                          (4, 1): [random.uniform(-600, 0), random.uniform(0, 600)],
                          (4, 2): [random.uniform(-600, 0), random.uniform(0, 600)]})
        _r_bounds.update({(5, 0): [random.uniform(-600, 0), random.uniform(0, 600)],
                          (5, 1): [random.uniform(-600, 0), random.uniform(0, 600)],
                          (5, 2): [random.uniform(-600, 0), random.uniform(0, 600)]})
        _r_bounds.update({(6, 0): [random.uniform(-600, 0), random.uniform(0, 600)],
                          (6, 1): [random.uniform(-600, 0), random.uniform(0, 600)],
                          (6, 2): [random.uniform(-600, 0), random.uniform(0, 600)]})

        #goal

        _r_bounds.update({(9, 0): [600, 1000], (9, 1): [600, 1000], (9, 2): [600, 1000]})


    _r_bounds.update({(7,0):[0.0,0.0],(7,1): [0.0,0.0], (7,2): [0.0,0.0]})
    _r_bounds.update({(8,0):[0.0,0.0],(8,1): [0.0,0.0], (8,2): [0.0,0.0]})


    return MDP(
        _startingstate= set(range(n_states)),
        _transitions= _t,
        _rewards_bounds= _r_bounds,
        _gamma = 0.95, _alpha= [1.0]+9*[0.0])


# %%%%%%%%%%%%% counter example %%%%%%%%%%%%%
def mdp_counter_example(T0, T1, A, B, C):

    n_states= 4
    n_actions = 3

    _t = {}
    _r = {}

    _t.update({(2,0,0):1.0, (2, 1, 1): 1.0, (2,2,0):T0, (2,2,1):T1, (0,0,3):1.0, (1,0,3):1.0})
    _r.update({(0,0):[-A,A],                 (0,1): [-10000.0, -10000.0], (0,2): [-10000.0, -10000.0],
               (1,0):[-A+B, A+B],            (1,1): [-10000.0, -10000.0], (1,2): [-10000.0, -10000.0],
               (2,0):[0.0, 0.0],             (2,1): [0.0, 0.0],           (2,2): [C, C],
               (3,0):[-10000.0, -10000.0],   (3,1): [-10000.0, -10000.0], (3,2): [-10000.0, -10000.0]
               })
    _alpha = [0.0, 0.0, 1.0, 0.0]

    return MDP(
        _startingstate= set(range(n_states)),
        _transitions= _t,
        _rewards_bounds= _r,
        _gamma = 0.9999, _alpha= _alpha)

