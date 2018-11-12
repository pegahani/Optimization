import scipy
from scipy.sparse import dok_matrix
from sys import maxint
import random
from itertools import product, izip, count, repeat, starmap, islice, ifilter
import math
from toolz import first
from collections import defaultdict
import numpy as np
from heapq import heappush, heappop

# try:
#     from scipy.sparse import csr_matrix, dok_matrix
#     from scipy.spatial.distance import cityblock as l1distance
# except:
#     from sparse_mat import dok_matrix,csr_matrix,l1distance

ftype = np.float32
#np.set_printoptions(threshold='nan')

class VVMdp:

    def __init__(self,
                 _startingstate,
                 _transitions,  # dictionary of key:values   (s,a,s):proba
                 _rewards,  # dictionary of key:values   s: vector of rewards
                 _gamma=.9, _lambda = None):

        self._lambda = _lambda

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

            d = len(_rewards[first(_rewards.iterkeys())])
            assert all(d == len(np.array(v,dtype=ftype)) for v in _rewards.itervalues()),\
                   "incorrect reward vectors"
            assert set(_rewards.keys()).issubset(states) ,\
                   "states appearing in rewards should also appear in transitions"

        except ValueError,TypeError:

            print("transitions or rewards do not have the correct structure")
            raise


        # convert rewards to nstates x d matrix
        rewards = np.zeros((n,d), dtype=ftype)
        for s, rv in _rewards.iteritems():
            rewards[stateInd[s], :] = rv

        #this finds maximum of sum of reward vectors among all row of rewards! why does he compute it? i don't know
        self.rmax = np.max( [sum(abs(rewards[s,:])) for s in range(n)] )

        # Convert Transitions to nstates x nactions array of sparse 1 x nstates matrices
        # sparse matrices are build as DOK matrices and then converted to CSR format
        # build reverse_transition dictionary of state:set of states
        transitions = np.array(
            [[ dok_matrix((1, n), dtype=ftype) for _ in actions ] for _ in states ],
            dtype=object
        )

        rev_transitions = defaultdict(set)

        for (s, a, s2), p in _transitions.iteritems():
            si, ai, si2 = stateInd[s], actionInd[a], stateInd[s2]
            transitions[si,ai][0, si2] = p
            rev_transitions[si2].add(si)

        for s, a in product(range(n), range(na)):
            transitions[s,a] = transitions[s,a].tocsr()
            assert 0.99 <= transitions[s,a].sum() <= 1.01, "probability transitions should sum up to 1"

        # autoprobability[s,a] = P(s|s,a)
        self.auto_probability = np.array( [[transitions[s,a][0,s] for a in range(na)] for s in range(n)] ,dtype=ftype )

        # copy local variables in object variables
        self.states , self.actions , self.nstates , self.nactions, self.d = states,actions,n,na,d
        self.stateInd,self.actionInd = stateInd,actionInd
        self.rewards , self.transitions, self.rev_transitions = rewards , transitions, rev_transitions
        self.gamma = _gamma

        #E_test = np.zeros((n*na, n), dtype=ftype)
        E_test = dok_matrix((n*na, n), dtype=ftype)

        for s in range(n):
            for a in range(na):
                E_test[s*na+a, :] = [ transitions[s,a][0,i] for i in range(n) ]

        self.E_test = E_test

    def T(self, state, action):
        """Transition model.  From a state and an action, return all
        of (state , probability) pairs."""
        _tr = self.transitions[state,action]
        return izip(_tr.indices, _tr.data)

    def set_Lambda(self,l):
        self.Lambda = np.array(l,dtype=ftype)

    def get_lambda(self):
        return self.Lambda

    def expected_vec_utility(self,s,a, Uvec):
        "The expected vector utility of doing a in state s, according to the MDP and U."
        # Uvec is a (nxd) matrix
        return np.sum( (p*Uvec[s2] for s2,p in self.T(s,a)) )

    def expected_dot_utility(self,s,a,Uvec):
        # assumes self.Lambda numpy array exists
        # Uvec is a (nxd) matrix
        return sum( (p*(Uvec[s2].dot(self.Lambda)) for s2,p in self.T(s,a)) )

    def expected_scalar_utility(self,s,a,U):
        # U is a n-dimensional vector
        return self.transitions[s,a].dot(U)

    def get_vec_Q(self,s,a,Uvec):
        # Uvec is a (nxd) matrix
        return self.rewards[s] + self.gamma * self.expected_vec_utility(s,a,Uvec)

    #evaluate policy and modify nxd matrix Uvec after all row's modification
    def policy_evaluation(self, epsilon, policy, k, Uvec):
        #_Uvec is of dimension nxd
        n, d = self.nstates, self.d
        gamma , R , expected_scalar_utility = self.gamma , self.rewards , self.expected_scalar_utility

        _uvec= np.zeros( (n,d) , dtype=ftype)

        for t in range(k):

            delta = 0.0
            for s in range(n):
                # Choose the action
                act = random.choice(policy[s])
                # Compute the update
                _uvec[s] = R[s] + gamma * self.expected_vec_utility(s,act,Uvec)

                delta = max(delta,  scipy.spatial.distance.cityblock(_uvec[s] , Uvec[s], w=None) ) #l1distance(_uvec[s] , Uvec[s]) )

            for s in range(n):
                Uvec[s] = _uvec[s]

            if delta < epsilon * (1 - gamma) / gamma:
                return Uvec
        return Uvec

    def update_matrix(self, policy_p,_Uvec_nd):

        """
        This function receives an updated policy after considering advantages and the old nxd matrix
        and it returns the updated matrix related to new policy
        :param policy_p: a given policy
        :param _Uvec_nd: nxd matrix before implementing new policy
        :return: nxd matrix after improvement
        """

        n , d  = self.nstates, self.d
        gamma , R  = self.gamma , self.rewards

        _uvec_nd= np.zeros((n,d) , dtype=ftype)

        for s in range(n):
                act = random.choice(policy_p[s])

                # Compute the update
                _uvec_nd[s,:] = R[s] + gamma * self.expected_vec_utility(s, act, _Uvec_nd)

        return _uvec_nd

    def value_iteration(self, epsilon=0.001,policy=None,k=100000,_Uvec=None, _stationary= True):
        "Solving an MDP by value iteration. [Fig. 17.4]"
        n , na, d , Lambda = self.nstates , self.nactions, self.d , self.Lambda
        gamma , R , expected_scalar_utility = self.gamma , self.rewards , self.expected_scalar_utility

        Udot = np.zeros( n , dtype=ftype)
        _uvec= np.zeros( d , dtype=ftype)
        #Rdot = np.array( [R[s].dot(Lambda) for s in range(n)] ,dtype=ftype)
        Uvec = np.zeros( (n,d) , dtype=ftype)
        if _Uvec != None:
            Uvec[:] = _Uvec

        Q    = np.zeros( na , dtype=ftype )

        for t in range(k):

            delta = 0.0
            for s in range(n):

                # Choose the action
                if policy != None:
                    if _stationary:
                        act = random.choice(policy[s])
                    else:
                        act = policy[s]

                else:
                    Q[:]    = [expected_scalar_utility(s, a, Udot) for a in range(na)]
                    act     = np.argmax(Q)

                # Compute the update
                _uvec[:] = R[s] + gamma * self.expected_vec_utility(s,act,Uvec)
                _udot    = Lambda.dot(_uvec)

                if policy != None:
                    delta = max(delta, l1distance(_uvec , Uvec[s]) )
                else:
                    #print "old delta=",delta," , new delta=",max(delta, abs(_udot-Udot[s]) )
                    #print "_udot=",_udot,"  ,  Udot[s]=",Udot[s]
                    #print "_uvec=",_uvec,"  ,  Uvec[s]=",Uvec[s]
                    #print

                    delta = max(delta, abs(_udot-Udot[s]) )

                Uvec[s] = _uvec
                Udot[s] = _udot

            if delta < epsilon * (1 - gamma) / gamma:
                return Uvec
        return Uvec

    def best_action(self,s,U):
        # U is a (nxd) matrix
        # Lambda has to be defined
        return np.argmax( [self.expected_dot_utility(s,a,U) for a in range(self.nactions)] )

    def best_policy(self, U):
        """Given an MDP and a (nxd) utility function U, determine the best policy,
        as a mapping from state to action. (Equation 17.4)"""
        pi = np.zeros((self.nstates),np.int)
        for s in range(self.nstates):
            pi[s] = self.best_action(s,U)
        return pi

    def readable_policy(self,pi):
        return {self.states[s]:self.actions[a] for s,a in pi.iteritems()}

    def policy_iteration(self,_Uvec=None):
        "Solve an MDP by policy iteration [Fig. 17.7]"
        if _Uvec == None:
            U = np.zeros( (self.nstates,self.d) , dtype=ftype)
        else:
            U = _Uvec

        pi = {s:random.randint(0,self.nactions-1) for s in range(self.nstates)}
        while True:
            U = self.value_iteration(epsilon=0.0,policy=pi, k=20,_Uvec=U, _stationary=False)
            unchanged = True
            for s in range(self.nstates):
                a = self.best_action(s,U)
                if a != pi[s]:
                    pi[s] = a
                    unchanged = False
            if unchanged:
                return U

    def prioritized_sweeping_policy_evaluation(self,pi, U1, k=maxint , epsilon=0.001):
        """Return an updated utility mapping U from each state in the MDP to its
        utility, using an approximation (modified policy iteration)."""
        R, gamma ,expect_vec_u = self.rewards, self.gamma , self.expected_vec_utility
        h = []

        for s in range(self.nstates):
            heappush( h , (-self.rmax-random.uniform(0,1),s) )
            print('after push')
            print(h)

        for i in count(0):
            U = U1.copy()

            (delta,s) = heappop(h)
            print('after pop')
            print(h)

            U1[s] = R[s] + gamma * expect_vec_u(s,pi[s],U)

            delta = scipy.spatial.distance.cityblock(U1[s], U[s], w = None) #(U1[s],U[s])

            if i > k or delta < epsilon * (1 - gamma) / gamma:
                return U

    def initial_states_distribution(self):
        n = self.nstates
        _init_states = np.zeros(n, dtype=ftype)

        init_n = len(self.startingStateInd)

        for i in range(n):
            if i in self.startingStateInd:
                _init_states[i] = ftype(1)/ftype(init_n)
                #_init_states[i] = 1.0
            else:
                _init_states[i] = 0.0

        return _init_states

    def calculate_advantages(self, _Uvec, pi, _initialDistribution):

        n , na, d , Lambda = self.nstates, self.nactions, self.d, self.Lambda
        advantage_array = np.zeros((self.nstates*self.nactions, self.d), dtype=ftype)

        #advantage_dic = {}

        init_distribution = self.initial_states_distribution()
        counter = 0

        for s in range(n):
            for a in range(na):
                advantage_value = self.get_vec_Q(s,a,_Uvec)-_Uvec[s]
                if _initialDistribution:
                    advantage_array[counter:counter+1] = init_distribution[s]*( advantage_value)
                else:
                    advantage_array[counter:counter+1] = advantage_value
                #advantage_dic[(s,a)] = advantage_array[counter:counter+1]
                counter+=1

        return advantage_array

    def calculate_advantages_labels(self, _matrix_nd, _IsInitialDistribution):
        """
        This function get a matrix and finds all |S|x|A| advantages
        :param _matrix_nd: a matrix of dimension nxd which is required to calculate advantages
        :param _IsInitialDistribution: if initial distribution should be considered in advantage calculation or not
        :return: a dictionary of all advantages for our MDP. keys are pairs and values are advantages vectros
                for instance: for state s and action a and d= 3 we have: (s,a): [0.1,0.2,0.4]
        """
        n, na = self.nstates, self.nactions

        advantage_dic = {}
        init_distribution = self.initial_states_distribution()

        for s in range(n):
            for a in range(na):
                advantage_d = self.get_vec_Q(s,a,_matrix_nd)-_matrix_nd[s]
                if _IsInitialDistribution:
                    advantage_dic[(s,a)] = init_distribution[s]*( advantage_d)
                else:
                    advantage_dic[(s,a)]= advantage_d

        return advantage_dic

def make_grid_VVMDP(n=10):
    _t =       { ((i,j),'v',(min(i+1,n-1),j)):0.9 for i,j in product(range(n),range(n)) }
    _t.update( { ((i,j),'v',(max(i-1,0),j)):0.1 for i,j in product(range(n),range(n)) } )
    _t.update( { ((i,j),'^',(max(i-1,0),j)):0.9 for i,j in product(range(n),range(n)) } )
    _t.update( { ((i,j),'^',(min(i+1,n-1),j)):0.1 for i,j in product(range(n),range(n)) } )
    _t.update( { ((i,j),'>',(i,min(j+1,n-1))):0.9 for i,j in product(range(n),range(n)) } )
    _t.update( { ((i,j),'>',(i,max(j-1,0))):0.1 for i,j in product(range(n),range(n)) } )
    _t.update( { ((i,j),'<',(i,max(j-1,0))):0.9 for i,j in product(range(n),range(n)) } )
    _t.update( { ((i,j),'<',(i,min(j+1,n-1))):0.1 for i,j in product(range(n),range(n)) } )
    _t.update( { ((i,j),'X',(i,j)):1 for i,j in product(range(n),range(n)) } )

    _r = { (i,j):[0.0,0.0] for i,j in product(range(n),range(n))}
    _r[(n-1,0)] = [1.0,0.0]
    _r[(0,n-1)] = [0.0,1.0]
    _r[(n-1,n-1)] = [1.0,1.0]


    gridMdp = VVMdp (
        _startingstate= {(0,0)},
        _transitions= _t,
        _rewards=_r
    )
    gridMdp.set_Lambda( [1,0] )
    return gridMdp


def show_grid_policy(vvmdp,pi,n=10):
    print(np.matrix([[vvmdp.actions[pi[vvmdp.stateInd[(i,j)]]] for j in range(n)] for i in range(n)]))


#******************************** my code for simulating a VVmdp ****************************
"this part simulate an mdp based on our general form"
def generate_guassian(element_numbers):
    values = []
    sum = 0
    while len(values) < element_numbers-1:
        value = random.gauss(0.5, 0.5)
        if sum + value <= 1 and 0 <= value <= 1:
            values.append(value)
            sum += value
    values.append(1.0-sum)
    return values

def generate_probability(n_states, n_actions):
    #number of accessible states for any (s,a) pair
    my_lenth = int(math.ceil(math.log1p(n_states)))
    random_chosen_states_list = {(s,a):random.sample(range(n_states), my_lenth) for s,a in product(range(n_states), range(n_actions)) }
    _t = {}
    for s,a in product(range(n_states), range(n_actions)):
        _prob_list = []
        guassi_result = generate_guassian(my_lenth)
        for sprime in range(n_states):
            if sprime not in random_chosen_states_list[(s,a)]:
                _prob_list.append(0)
            else:
                _prob_list.append(guassi_result.pop())
        _t[(s,a)] = _prob_list
    return _t

#this function generates _n_d dimensional vector with 1 in an arbitrary position
def generate_random_vector(_n_d):
        my_list = list(repeat(0,_n_d))
        position = random.randint(0, _n_d-1)
        #my_list[position] = 1
        my_list[position] = np.float32(random.random())
        return my_list

#this function choose random reward function for states in a d dimensional space
def generate_random_reward_function(n_states, n_actions ,n_d):
    _r= {}
    for i in range(n_states):
        _r[i]=  generate_random_vector(n_d)
    return _r

def make_simulate_mdp(n_states, n_actions, _lambda, _r):
    #verify random chosen states for any (s,a) pair

    generated_probability =  generate_probability(n_states, n_actions)
    _t = {(s,a,sprime):generated_probability[(s,a)][sprime] for s,a,sprime in product(range(n_states), range(n_actions), range(n_states))}
    # _r = {i:[0.0,0.0] for i in range(n_states)}
    # _r[n_states-1] = [1.0,1.0]

    if _r is None:
        _r = generate_random_reward_function(n_states, n_actions, len(_lambda))

    simulate_mdp = VVMdp(
        _startingstate= set(range(n_states)),
        _transitions= _t,
        _rewards= _r ,
        _gamma= 0.95
    )

    assert len(_r[0])==len(_lambda), \
                "reward elements' lengths and lambda length are not equal! they should be equal"

    simulate_mdp.set_Lambda(_lambda)

    return simulate_mdp


#***********Yann code**************
def make_simulate_mdp_Yann(n_states, n_actions, _lambda, _r=None):
    """ Builds a random MDP.
        Each state has ceil(log(nstates)) successors.
        Reward vectors are permutations of [1,0,...,0]
    """

    nsuccessors = int(math.ceil(math.log1p(n_states)))
    gauss_iter  = starmap(random.gauss,repeat((0.5,0.5)))
    _t = {}

    for s,a in product(range(n_states), range(n_actions)):

        next_states = random.sample(range(n_states), nsuccessors)
        probas =  np.fromiter(islice(ifilter(lambda x: 0 < x < 1, gauss_iter),nsuccessors), ftype)

        _t.update(  {(s,a,s2):p for s2,p in izip(next_states, probas/sum(probas) )  }  )

    if _r is None:
        _r = {i:np.random.permutation([1]+[0]*(len(_lambda)-1)) for i in range(n_states)}

    assert len(_r[0])==len(_lambda),"Reward vectors should have same length as lambda"

    return VVMdp(
        _startingstate= set(range(n_states)),
        _transitions= _t,
        _rewards= _r ,
        _gamma= 0.95,
        _lambda=_lambda)

#***********Yann code**************







#******************************** my code for simulating a VVmdp ****************************

def test_VVMDP():
    monMdp = VVMdp(
        _startingstate='buro',
        _transitions={
            ('buro', 'bouger', 'couloir'): 0.4,
            ('buro', 'bouger', 'buro'): 0.6,
            ('buro', 'rester', 'buro'): 1,
            ('couloir', 'bouger', 'couloir'): 1,
            ('couloir', 'bouger', 'buro'): 0,
            ('couloir', 'rester', 'couloir'): 1,
            ('couloir', 'rester', 'buro'): 0
        },
        _rewards={
            'buro': [0.1, 0.0],
            'couloir': [0.1, 0.0]
        }
    )

    import pprint

    print("--state indices and state set--")
    pprint.pprint(monMdp.stateInd)
    pprint.pprint(monMdp.states)
    print("--action indices and action set--")
    pprint.pprint(monMdp.actionInd)
    pprint.pprint(monMdp.actions)
    print("--nstates,nactions,d--")
    print(monMdp.nstates, monMdp.nactions, monMdp.d)
    print("--rewards--")
    pprint.pprint(monMdp.rewards)
    print("--transitions--")
    pprint.pprint([[M.toarray() for M in L] for L in monMdp.transitions])
    print("--reverse transitions--")
    pprint.pprint(monMdp.rev_transitions)
    print("-- testing T --")
    pprint.pprint(list(monMdp.T(0, 0)))
    #U = np.zeros((2,2),dtype=ftype)
    #U[1] = [3,4]
    #return monMdp, U

    ########################################

    # gridMdp = make_grid_VVMDP()
    #
    # print "---value iteration---"
    #
    # U = gridMdp.policy_iteration()
    # print "--- value iteration ended ---"
    # pi = gridMdp.best_policy(U)
    # show_grid_policy(gridMdp,pi)

def show_test_policy(test, pi):
    print([test.actions[pi[test.stateInd['buro']]], test.actions[pi[test.stateInd['couloir']]]])
    #print(test.actions[ pi(test.stateInd['couloir']) ])

def make_test_VVMDP():
        monMdp = VVMdp(
        _startingstate={'buro'},
        _transitions={
            ('buro', 'bouger', 'couloir'): 0.4,
            ('buro', 'bouger', 'buro'): 0.6,
            ('buro', 'rester', 'buro'): 1,
            ('couloir', 'bouger', 'couloir'): 1,
            ('couloir', 'bouger', 'buro'): 0,
            ('couloir', 'rester', 'couloir'): 1,
            ('couloir', 'rester', 'buro'): 0
        },
        _rewards={
            ('buro', 'bouger'): [0.1, 0.0],
            ('buro', 'rester'): [0.0, 1.0],
            ('couloir', 'bouger'): [0.1, 0.0],
            ('couloir','rester'): [0.1, 0.0]
        }
    )

        monMdp.set_Lambda( [1, 0] )
        return monMdp