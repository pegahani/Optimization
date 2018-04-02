from itertools import product
from scipy.optimize.zeros import results_c

import cplex
import numpy as np
import pickle
import time

import sys

from manage_stack import Stack

sys.path.insert(0, '/export/home/alizadeh/Documents/Research/Aomar/Bruno+Emiliano/CODE/Optimization')
sys.stdout.flush()
import classic_mdp

sys.setrecursionlimit(1000000)


class minmax_regret:

    def __init__(self, _mdp, _reward_bounds):
        self.mdp = _mdp
        """_reward_bounds is a |S|x|A| size vector with two dimensional vector elements [lb, ub] where lb <= r_sa <= ub """
        self.reward_bounds = [_reward_bounds]* (_mdp.nstates*_mdp.nactions)
        self.stack = Stack()

        self.EPSI_cutoff =0.001
        self.EPSI_violation_slave =0.000001
        self.EPSI_integrality_check=0.00001

        #BB infos
        self.UB = cplex.infinity
        self.ROOT_LB= -cplex.infinity
        self.best_f = []
        self.ROOT_tot_cuts = 0
        self.BB_tot_nodes = 0
        self.BB_nodes_pruned = 0
        self.BB_current_level = 0

        self.BEST_sto_policy = []
        self.BEST_det_policy = []

        self.MASTER_tot_cuts = 0

        self.controesempio = False

        self.TIME_master = 0
        self.TIME_slave = 0
        self.TIME_root = 0
        self.TIME_limit = 0
        self.TIME_limit_reached = False

        self.verbosity = 2

        self.output = ""


    def best_V(self, epsilon=0.001):

        ns = self.mdp.nstates
        na = self.mdp.nactions

        rewards, transition, gamma = self.mdp.rewards, self.mdp.transitions, self.mdp.gamma
        V1 = np.zeros(ns)

        while True:
            V = V1.copy()
            delta = 0
            for _s in range(ns):
                V1[_s] = max(self.reward_bounds[_s*na + _a][1] + gamma * sum(transition[_s,_a][0,_s2] * V[_s2] for _s2 in range(ns)) for _a in range(na))
                delta = max(delta, abs(V1[_s] - V[_s]))

            if delta < epsilon * 1.0/(1.0-gamma): #(1 - gamma) / gamma:
                return V.tolist()

    def worst_Q(self, epsilon=0.001):

        ns = self.mdp.nstates
        na = self.mdp.nactions

        Q1 = np.zeros(ns*na)
        V1 = np.zeros(ns)
        rewards, transition, gamma = self.mdp.rewards, self.mdp.transitions, self.mdp.gamma

        while True:
            Q = Q1.copy()
            V = V1.copy()

            delta = 0
            for _s in range(ns):
                for _a in range(na):
                    Q1[_s*na+_a] = self.reward_bounds[_s*na + _a][0] + gamma * sum(transition[_s,_a][0,_s2] * V[_s2] for _s2 in range(ns))
                    delta = max(delta, abs(Q1[_s*na+_a] - Q[_s*na+_a]))

                V1[_s] = max(Q1[_s*na:_s*na+na])

            if delta < epsilon * 1.0/(1.0-gamma):  #(1 - gamma) / gamma:
                return Q.tolist()

    def get_bigM(self):

        ns, na = self.mdp.nstates, self.mdp.nactions

        V = self.best_V()
        Q = self.worst_Q()
        big_M = []

        for _s in range(ns):
            big_M = big_M + [ V[_s]-Q[_s*na+_a] for _a in range(na)]

        return big_M

    def make_master(self):#, GEN = None):
        """

        :param GEN:
        :return:
        """
        """initialize Master program as a minimization problem"""
        self.master = cplex.Cplex()

        self.master.objective.set_sense(self.master.objective.sense.minimize)
        self.master.variables.add(names = ['delta']+['f_'+str(i)+'_'+str(j) for (i,j) in product(range(self.mdp.nstates),range(self.mdp.nactions))])
        self.master.objective.set_linear(0, 1.0)

        constr, rhs = [], []

        tempo = (self.mdp.E).todense()
        E_transpose = np.multiply(self.mdp.gamma, tempo.transpose())

        for i in xrange(E_transpose.shape[0]):
            which_coeff = [j+1 for j in range(self.mdp.nstates*self.mdp.nactions)]
            constr.append( [which_coeff, E_transpose[i,:].tolist()[0] ] )

        self.master.linear_constraints.add(lin_expr= constr,
                                           #rhs=[1*i for i in self.mdp.alpha]) # WARNING :  using -alpha instead of alpha
                                           rhs=[-1*i for i in self.mdp.alpha])
                                           
        if self.verbosity >=2:
            # print self.mdp.transitions.shape
            # print self.mdp.transitions
            # for s in range(self.mdp.nstates):
            #     for a in range(self.mdp.nactions):
            #         print [self.mdp.transitions[s,a][0,_s] for _s in range(self.mdp.nstates)]
            #     print
            #
            # print "**********"
            #
            # print "E_transpose  : ", tempo
            print "self.mdp.gamma : ", self.mdp.gamma
            print "self.mdp.alpha : ", self.mdp.alpha

        self.master.write("master.lp")

        return self.master

    def make_slave(self, f=None):

        ns = self.mdp.nstates
        na = self.mdp.nactions

        """initialize Slave program as a maximization problem"""
        self.slave = cplex.Cplex()
        self.slave.objective.set_sense(self.slave.objective.sense.maximize)
        self.slave.variables.add(names = ['Q_'+str(i)+'_'+str(j) for (i,j) in product(range(self.mdp.nstates),range(self.mdp.nactions))] +
                                          ['V_' + str(i) for i in range(self.mdp.nstates)]+
                                          ['I_' + str(i) + '_' + str(j) for (i, j) in product(range(self.mdp.nstates), range(self.mdp.nactions))]+
                                          ['r_' + str(i) + '_' + str(j) for (i, j) in product(range(self.mdp.nstates), range(self.mdp.nactions))])

        counter = self.mdp.nstates*self.mdp.nactions

        for i in range(ns):
            self.slave.objective.set_linear(int(counter+i), float(self.mdp.alpha[i]))


        transitions = self.mdp.transitions

        #Q_a = r_a + gamma P_a V
        for _action in range(na):
            constr, rhs = [], []
            for _state in range(ns):
                coeff, index = [], []
                #Q
                index.append(_state*na + _action)
                coeff.append(-1.0)
                #r
                index.append(ns*na + ns + ns*na + _state*na+_action)
                coeff.append(1.0)

                #V
                for _state2 in range(ns):
                    index.append(ns*na+ _state2)
                    coeff.append( np.float(self.mdp.gamma * transitions[_state, _action][0, _state2]) )

                self.slave.linear_constraints.add(lin_expr = [cplex.SparsePair(ind= index, val=coeff)], senses=["E"], rhs=[0.0])

        #V => Q_a
        for _a in range(na):
            for _s in range(ns):
                coeff, index = [], []
                #Q
                index.append(_s*na + _a)
                coeff.append(-1.0)
                #V
                index.append(ns*na+ _s)
                coeff.append(1.0)

                self.slave.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=index, val=coeff)],senses=["G"], rhs=[0.0])



        # V <= (1-I_a)M_a + Q_a
        big_M = [1e8]*ns*na
        #big_M = self.get_bigM() #of size ns*na
        for _a in range(na):
            for _s in range(ns):
                coeff, index = [], []
                # Q
                index.append(_s * na + _a)
                coeff.append(-1.0)
                # V
                index.append(ns * na + _s)
                coeff.append(1.0)
                # I
                index.append(ns*na+ns+_s * na + _a)
                coeff.append(big_M[_s * na + _a])

                self.slave.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=index, val=coeff)], senses=["L"],
                                                  rhs=[big_M[_s * na + _a]])


        # sum_a i_a = 1
        for _s in range(ns):
            coeff, index = [], []
            for _a in range(na):
                # I
                index.append(ns*na+ns+_s * na + _a)
                coeff.append(1)
            self.slave.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=index, val=coeff)], senses=["E"], rhs=[1.0])

        #bound on r variables
        for _s in range(ns):
            self.slave.variables.set_lower_bounds(ns*na+ _s, -cplex.infinity)
            for _a in range(na):
                index = _s*na+_a
                index_I = ns*na+ ns + index
                # r bounds
                self.slave.variables.set_lower_bounds(ns*na + ns + na*ns + index, self.reward_bounds[index][0])
                self.slave.variables.set_upper_bounds(ns*na + ns + na*ns + index, self.reward_bounds[index][1])
                # I binaries
                self.slave.variables.set_types(index_I, self.slave.variables.type.binary)
                # Q bounds
                self.slave.variables.set_lower_bounds(index,-cplex.infinity)


        self.slave.write("slave.lp")

        pass

    def update_slave(self, result_master):

        ns = self.mdp.nstates
        na = self.mdp.nactions

        for _s in range(self.mdp.nstates):
            for _a in range(self.mdp.nactions):
                self.slave.objective.set_linear(ns*na + ns + ns*na + _s*na+_a, -1*result_master[1+_s*na+_a])

        if self.verbosity >=2:
            self.slave.write("slave_update.lp")

        pass

    def update_master(self, result_slave):

        ns = self.mdp.nstates
        na = self.mdp.nactions

        V = result_slave[ns*na: ns*na+ns]
        r =  result_slave[ns*na+ns+ns*na: ns*na+ns+ns*na+ns*na]
        rhs_val = np.dot(self.mdp.alpha, V)

        self.master.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=range(1+ns*na), val=[1.0]+r)], senses=["G"], rhs=[rhs_val])
        if self.verbosity >=2:
            self.master.write('master_update.lp')

        pass

    def solve_deterministic_opt_stack(self, TL, epsilon):

        nstate, naction = self.mdp.nstates, self.mdp.nactions


        self.make_slave()
        self.make_master()

        self.UB = cplex.infinity

        self.BB_nodes_pruned = 0
        self.BB_current_level = 0

        self.MASTER_tot_cuts = 0

        self.TIME_slave = 0
        self.TIME_master = 0
        self.TIME_limit = TL

        new_item = {'id':1, 'level':1, 'LB': -cplex.infinity, 'fixing': (nstate*naction)*[-1]}
        self.stack.push(new_item)

        while not self.stack.isEmpty():
            pop_item = self.stack.pop()
            tempo_fixing = pop_item['fixing']
            self.fix_stack(tempo_fixing)
            if (self.TIME_limit < (self.TIME_master + self.TIME_slave)):
                if self.TIME_limit_reached == False:
                    if __debug__:
                        if self.verbosity >= 1:
                            print "TIME LIMIT of ", self.TIME_limit, " seconds reached!!!"
                    self.TIME_limit_reached = True
                break

            it_counter = 0
            self.BB_tot_nodes += 1
            self.BB_current_level += 1

            if self.BB_tot_nodes == 1:
                start_t = time.time()

            results_master = self.solve_stochastic_opt(epsilon, alone=False, upper_bound=self.UB)
            f, delta = results_master[1:], results_master[0]

            self.master.write('master_debug_stack.lp')


            if self.BB_tot_nodes == 1:
                end_t = time.time()
                self.TIME_root += (end_t - start_t)
                self.BEST_sto_policy = results_master
                self.ROOT_LB = results_master[0]
                self.ROOT_tot_cuts = self.MASTER_tot_cuts
            if __debug__:
                if (self.verbosity == 1 and self.BB_tot_nodes % 100 == 0) or self.verbosity >= 2:
                    print "node ", self.BB_tot_nodes, " lv ", self.BB_current_level, " nd prnd ", self.BB_nodes_pruned, "UB ", self.UB, " LB ", \
                    results_master[
                        0], " tot cuts ", self.MASTER_tot_cuts, " T_M ", self.TIME_master, " T_S ", self.TIME_slave
                    # self.output += "node "+ ","+ str(self.BB_tot_nodes) + ',' + " lv " + ',' + str(self.BB_current_level) + ',' + " nd prnd " + ',' + str(self.BB_nodes_pruned)+ ','\
                    #    "UB "+ str(self.UB) + ',' + " LB " + ',' + str(results_master[0]) + ',' + " tot cuts " + ',' + str(self.MASTER_tot_cuts) + ',' +  " T_M " + ',' +\
                    #               str(self.TIME_master) + ',' + " T_S " + ',' + str(self.TIME_slave) + ';'

            print "node ", self.BB_tot_nodes, " lv ", self.BB_current_level, " nd prnd ", self.BB_nodes_pruned, "UB ", self.UB, " LB ", \
                results_master[
                    0], " tot cuts ", self.MASTER_tot_cuts, " T_M ", self.TIME_master, " T_S ", self.TIME_slave

            """check if LB > UB"""
            if (delta >= self.UB):
                if __debug__:
                    if self.verbosity >= 2:
                        print "cut because of the bound"
                        print "new node - end"
                self.BB_nodes_pruned += 1
                self.BB_current_level -= 1
            else:
                result = self.f_is_deterministic(f)
                (f_out, stochastic_state_actions) = result #self.f_is_deterministic(f)
                """check if solution is integer"""
                if f_out:
                    self.UB = delta
                    self.best_f = f
                    self.BEST_det_policy = results_master
                    if __debug__:
                        if self.verbosity >= 2:
                            print "deterministic policy - update UB"
                            print "new node - end"
                    self.BB_current_level -= 1

                else:
                    # fix state action in the master with the maximum \pi(s,a)
                    assert stochastic_state_actions != None, \
                        "this policy is not stochastic"

                    """find new stocastic f"""
                    best_probability_index = np.argmax([item[2] for item in stochastic_state_actions])
                    best_state_action = stochastic_state_actions.pop(best_probability_index)[0:2]

                    _state = best_state_action[0]
                    _action = best_state_action[1]

                    new_fixing = pop_item['fixing']

                    # the right node
                    right_fixing = np.copy(new_fixing)
                    right_fixing[_state*naction+_action] = 0

                    right_item = {'id':1, 'level':pop_item['level']+1, 'LB': delta, 'fixing': right_fixing}


                    # the left node
                    left_fixing = np.copy(new_fixing)
                    for a in xrange(naction):
                        if a != _action:
                            left_fixing[_state*naction+a] = 0

                    left_item = {'id':1, 'level':pop_item['level']+1, 'LB': delta, 'fixing': left_fixing}

                    # add right_item
                    self.stack.push(right_item)
                    #add left_item
                    self.stack.push(left_item)






        print "RESULTS de MERDE ", self.UB, self.BEST_det_policy
        BEST_sto_policy_binary = [0 if e<1e-8 else 1 for e in self.BEST_sto_policy[1:]]
        BEST_det_policy_binary=[0  if e<1e-8 else 1 for e in self.BEST_det_policy[1:]]
        
        HEUR_det_policy_binary=[]
        ns = self.mdp.nstates
        na = self.mdp.nactions
        
        for i in range(ns):
            best_val = 0.0
            best_ind = -1
            for j in range(na):
                if self.BEST_sto_policy[1+ i*na+j] > best_val:
                    best_val= self.BEST_sto_policy[1+ i*na+j]
                    best_ind = j
            for j in range(na):
                if j == best_ind:
                    HEUR_det_policy_binary.append(1)
                else:
                    HEUR_det_policy_binary.append(0)
        
#        print "BEST_sto_policy ", self.BEST_sto_policy[1:]
#        print "BEST_det_policy ", self.BEST_det_policy[1:]
#        print "HEUR_det", HEUR_det_policy_binary
#        
#        print 'BEST_sto', BEST_sto_policy_binary
#        print 'BEST_det', BEST_det_policy_binary

        self.controesempio = False        
        for i in range(len(HEUR_det_policy_binary)):
            if (HEUR_det_policy_binary[i] != BEST_det_policy_binary[i]):
                self.controesempio = True
                break
#        print "controesempio ", self.controesempio
                
        
        pass

    def fix_stack(self, fixing):
        """
        it updates the bound accordning to fixing value
        :param fixing:
        :return:
        """
        """fixing elemnets have threee options: -1: free 0: value equal to 0 """
        for i in xrange(len(fixing)):
            if int(fixing[i]) == 0:
                self.master.variables.set_upper_bounds(1+ i, 0.0)
            else:
                self.master.variables.set_upper_bounds(1+ i, cplex.infinity)
        pass

    def solve_deterministic_opt(self, TL, epsilon):
        self.make_slave()
        self.make_master()

        self.UB = cplex.infinity

        self.BB_nodes_pruned = 0
        self.BB_current_level = 0

        self.MASTER_tot_cuts = 0

        self.TIME_slave = 0
        self.TIME_master = 0
        self.TIME_limit = TL        
        
        self.inner_bb(epsilon)

        if self.verbosity >=2:
            print "best sto policy: "
            print self.BEST_sto_policy

            print "best det policy: "
            print self.BEST_det_policy
        
        self.output += str(self.BEST_sto_policy) + ';'
        self.output += str(self.BEST_det_policy) + ';'


    def inner_bb(self, epsilon):
        if __debug__:
            if self.verbosity >= 2:
                print "-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_"
                print "-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_"
                print "new node - begin"
        
        if (self.TIME_limit< (self.TIME_master+self.TIME_slave)):
            if self.TIME_limit_reached == False:
                if __debug__:
                    if self.verbosity >= 1:
                        print "TIME LIMIT of ", self.TIME_limit, " seconds reached!!!"
                self.TIME_limit_reached = True
            return

        
        it_counter = 0
        self.BB_tot_nodes += 1
        self.BB_current_level+=1

        if self.BB_tot_nodes == 1:
            start_t = time.time()

        #solve master with cuts
        results_master = self.solve_stochastic_opt(epsilon,alone = False,upper_bound = self.UB)

        try:
            results_master[0] < cplex.infinity-1
        except ValueError:
            sys.exit("Error message")
            #print "WARNING -----> infeasible master"


        if self.BB_tot_nodes == 1:
            end_t = time.time()
            self.TIME_root += (end_t - start_t)
            self.BEST_sto_policy = results_master
            self.ROOT_LB = results_master[0]
            self.ROOT_tot_cuts = self.MASTER_tot_cuts
        if __debug__:
            if (self.verbosity == 1 and self.BB_tot_nodes % 100 == 0) or self.verbosity >= 2:
                print "node ", self.BB_tot_nodes, " lv ", self.BB_current_level, " nd prnd ", self.BB_nodes_pruned, "UB ", self.UB, " LB ", results_master[0], " tot cuts ", self.MASTER_tot_cuts , " T_M ", self.TIME_master, " T_S ", self.TIME_slave
                #self.output += "node "+ ","+ str(self.BB_tot_nodes) + ',' + " lv " + ',' + str(self.BB_current_level) + ',' + " nd prnd " + ',' + str(self.BB_nodes_pruned)+ ','\
                #    "UB "+ str(self.UB) + ',' + " LB " + ',' + str(results_master[0]) + ',' + " tot cuts " + ',' + str(self.MASTER_tot_cuts) + ',' +  " T_M " + ',' +\
                #               str(self.TIME_master) + ',' + " T_S " + ',' + str(self.TIME_slave) + ';'

        print "node ", self.BB_tot_nodes, " lv ", self.BB_current_level, " nd prnd ", self.BB_nodes_pruned, "UB ", self.UB, " LB ", \
        results_master[0], " tot cuts ", self.MASTER_tot_cuts, " T_M ", self.TIME_master, " T_S ", self.TIME_slave
        self.master.write('master_debug.lp')
        print results_master
        print "----"
        raw_input('PAUSA')



        #get f and delta from master
        f, delta = results_master[1:], results_master[0]

        if __debug__:
            if self.verbosity >= 2:
                print "new node - lb : " , delta, " UB : ", self.UB
                print "results_master : ", results_master
        if (delta >= self.UB):
            if __debug__:
                if self.verbosity >= 2:
                    print "cut because of the bound"
                    print "new node - end"
            self.BB_nodes_pruned += 1
            self.BB_current_level -= 1
            return
        else:
            (f_out, stochastic_state_actions) = self.f_is_deterministic(f)
            if f_out:
                self.UB = delta
                self.best_f = f
                self.BEST_det_policy = results_master
                if __debug__:
                    if self.verbosity >= 2:
                        print "deterministic policy - update UB"
                        print "new node - end"
                self.BB_current_level -= 1
                return

            else:
                # fix state action in the master with the maximum \pi(s,a)
                assert stochastic_state_actions != None, \
                    "this policy is not stochastic"

                best_probability_index = np.argmax([item[2] for item in stochastic_state_actions])
                best_state_action = stochastic_state_actions.pop(best_probability_index)[0:2]
                # fixing all actions for selected state s to 0 except for the selected action a
                self.fix_f_master(best_state_action, is_fix= True)
                # call branch and bound on the child of f_selected_stae,selected_action

                # the left node
                if __debug__:
                    if self.verbosity >= 2:
                        print "create child node left - fix (state,action) in solution : " , best_state_action
                print "create child node left - fix (state,action) in solution : " , best_state_action
                raw_input('PAUSA')
                self.inner_bb(epsilon)

                # free f_selected_s,selected_a bounds
                self.fix_f_master(best_state_action, is_fix= False)

                # the right node
                if __debug__:
                    if self.verbosity >= 2:
                        print "create child node righ - exclude (state,action) : " , best_state_action
                print "create child node righ - exclude (state,action) : ", best_state_action
                raw_input('PAUSA')
                self.inner_bb(epsilon)

                self.free_f_master(best_state_action)
                if __debug__:
                    if self.verbosity >= 2:
                        print "new node - end"

        self.BB_current_level -= 1
        return

    def solve_stochastic_opt(self, epsilon, alone=True, upper_bound = None):
        if alone :
            self.make_master()
            self.make_slave()


        if self.verbosity <= 2:
            self.master.set_log_stream(None)
            self.master.set_error_stream(None)
            self.master.set_warning_stream(None)
            self.master.set_results_stream(None)
            self.slave.set_log_stream(None)
            self.slave.set_error_stream(None)
            self.slave.set_warning_stream(None)
            self.slave.set_results_stream(None)

        self.slave.parameters.simplex.tolerances.optimality =1e-12
        if __debug__:
            if self.verbosity >= 2:
                print "solve stochastic policy - begin"
        it_counter =0
        while True :
            it_counter+=1
            # solve master

            if self.verbosity >= 3:
                print "********************************"
                print "********* iteration ", it_counter
                print "********************************"
                print "********************************"
                print "******** MASTER BEGIN **********"
                print "********************************"

            start_t = time.time()
            self.master.solve()
            end_t = time.time()

            if self.verbosity >= 3:
                print "********************************"
                print "******** MASTER END  ***********"
                print "********************************"


            
            self.TIME_master += (end_t - start_t)
            
                
            status_master = self.master.solution.get_status()
            if __debug__:
                if self.verbosity >= 2:
                    print ("status master ", status_master)
                    
            
            #checking if the master is infeasible
            if (status_master== 3 or status_master==103):
                print ("master infeasible")
                raw_input('PAUSA')
                return [cplex.infinity]

            else:
                if (status_master!= 1):
                    print ("********************************")
                    print ("********************************")
                    print ("WARNING!!! unknown master status")
                    print ("********************************")
                    print ("********************************")

                
            
            
            result_master = self.master.solution.get_values()
            objective_master = self.master.solution.get_objective_value()

            if __debug__:
                if self.verbosity >= 2:
                    print ("it ", it_counter, " obj m : ",  objective_master)

            if (alone == False):
                if (objective_master> upper_bound):
                    break

            # update slave with f*
            self.update_slave(result_master)

            # solve slave
            """add upper cutoff to slave program"""
            if __debug__:
                if self.verbosity >= 2:
                    print ("add upper cutoff rm:", result_master[0], " rm with epsi : ",  (1+epsilon)*(result_master[0]))
            
            upper_cutoff = max( 1.0, (1+self.EPSI_cutoff)*(result_master[0]) )
            self.slave.parameters.mip.limits.solutions.set(1) #upperobj = 1.0 #+ (1+epsilon)*(result_master[0])
            self.slave.parameters.mip.tolerances.lowercutoff.set(upper_cutoff)
            self.slave.parameters.timelimit.set(10)


            if self.verbosity >= 3:
                print "********************************"
                print "******** SLAVE BEGIN ***********"
                print "********************************"

            start_t = time.time()
            self.slave.solve()
            end_t = time.time()
            
            if self.verbosity >= 3:
                print "********************************"
                print "********  SLAVE END  ***********"
                print "********************************"

            self.TIME_slave += (end_t - start_t)

            status_slave = self.slave.solution.get_status()

            if __debug__:
                if self.verbosity >= 2:
                    print "status slave : " , status_slave
            if status_slave == 108 or status_slave == 119: #if it finds no soloution
                if True == True:
                    if __debug__:
                        if self.verbosity >=2:
                            print "alone == TRUE"
                    self.slave.parameters.mip.limits.solutions.reset()
                    self.slave.parameters.mip.tolerances.lowercutoff.reset()
                    self.slave.parameters.timelimit.reset()

                    ns = self.mdp.nstates
                    na = self.mdp.nactions
                    self.slave.variables.set_lower_bounds(ns * na + 0, -cplex.infinity)
                    self.slave.solve()
                    
                else:
                    #print "alone == FALSE"
                    break

            # get opt value from slave
            result_slave = self.slave.solution.get_values()
            objective_slave = self.slave.solution.get_objective_value()
            # print 'result slave', result_slave
            # print 'objective slave', objective_slave
            found_new_cut = False
            
            if result_master[0] > self.EPSI_violation_slave: #result_master[0] is greater than zero
                if (objective_slave/result_master[0]) > (1.0+self.EPSI_violation_slave):
                    found_new_cut = True
            else: 
                if result_master[0] > -self.EPSI_violation_slave: #result_master[0] is equal to zero
                    if (objective_slave) > (result_master[0]+self.EPSI_violation_slave):
                        found_new_cut = True 
                else: #result_master[0] is lower than to zero
                    if (objective_slave/result_master[0]) < (1.0-self.EPSI_violation_slave):
                        found_new_cut = True 

            if found_new_cut == True:
                if self.verbosity >= 2:
                    print "add cut alpha.V*-r*f* <= delta"                
                self.update_master(result_slave)
                self.MASTER_tot_cuts += 1
            else:
                break



        if __debug__:
            if self.verbosity >= 2:
                print "solve stochastic policy - end"
        return  result_master

    def f_is_deterministic(self, f):

        ns = self.mdp.nstates
        na = self.mdp.nactions
        
        stochastic_state_actions = []
        if self.verbosity >= 5:        
            print "selected policy"
            for i in range(ns):
                print "*****"
                for j in range(na):
                    print f[i*na+j]
                
        #raw_input('PAUSA')
        
        for i in range(ns):
            sum_f = 0.0
            zero_counter = 0
            action_holder = []
            for j in range(na):
                if ((f[i * na + j] > -self.EPSI_integrality_check) and (f[i * na + j] < self.EPSI_integrality_check)):
                #if f[i * na + j] == 0.0:
                    zero_counter += 1
                else:
                    action_holder.append(j)
                    sum_f += f[i * na + j]
            if zero_counter < (na -1):
                [stochastic_state_actions.append((i, j, f[i * na + j]/sum_f)) for j in action_holder]
                if self.verbosity >= 5:
                    print stochastic_state_actions                    
                
                return (False, stochastic_state_actions)

        return (True, None)


    def show_f(self, var):

        for i in range(self.mdp.nstates):
            line = [var[i * self.mdp.nactions + j] for j in range(self.mdp.nactions)]
            print line
            if len([_zero for _zero in line if _zero == 0.0]) < (self.mdp.nactions -1):
                print '*****YES!!!*****'
                return  '*****YES!!!*****'

        print '*****NO!!*****'
        return '*****NO!!*****'

    def fix_f_master(self, best_state_action, is_fix):
        """

        :param best_state_action:
        :param is_fix: if is_fix the function fix new constraints if no it free the related constraitns to best_state_action
        :return:
        """
        na = self.mdp.nactions

        _state = best_state_action[0]
        _action = best_state_action[1]

        _list = range(na)
        _list.remove(_action)

        for _a in _list:
            if is_fix:
                self.master.variables.set_upper_bounds(1+ _state*na + _a, 0.0)
            else:
                self.master.variables.set_upper_bounds(1 + _state * na + _a, cplex.infinity)
        if not is_fix:
            self.master.variables.set_upper_bounds(1 + _state * na + _action, 0.0)
        self.master.write('master_fix_f.lp')
        return

    def free_f_master(self, best_state_action):
        _state = best_state_action[0]
        _action = best_state_action[1]
        self.master.variables.set_upper_bounds(1 + _state * self.mdp.nactions + _action, cplex.infinity)
        return


def load_mdp(state, action, gamma, _id, _reward_lb, _reward_up):
    """
    Creates a new mdp, initialize related global variables and saves what is needed for reuse
    :type _id: string e.g. 80-1 to save in param80-1.dmp"""

    mdp = classic_mdp.general_random_mdp(state, action, gamma,_reward_lb = _reward_lb, _reward_up = _reward_up)

    # if not _id is None:
    #name = "param_" + str(_id) + ".dmp"
    name = './Models/mdp_' + str(state) + '_' + str(action) + '_' + str(gamma) + '_' + str(_id) + '_'+ str(_reward_lb) + '_' + str(_reward_up) + ".dmp"
    pp = pickle.Pickler(open(name, 'w'))
    pp.dump(mdp)

    pass

def reload_mdp(state, action, gamma, _id, _reward_lb, _reward_up):#(_id):
    """
    Reloads a saved mdp and initialize related global variables
    :type _id: string e.g. 80-1 to reload param80-1.dmp
    """
    #name = "param_" + str(_id) + ".dmp"
    name = './Models/mdp_' + str(state) + '_' + str(action) + '_' + str(gamma) + '_' + str(_id) + '_' + str(_reward_lb) + '_' + str(_reward_up) + ".dmp"
    pup = pickle.Unpickler(open(name, 'r'))
    mdp = pup.load()

    return mdp

#load_mdp(4, 4, 0.9, 'test')
#_mdp = reload_mdp('test_stochastic')
#_mdp = reload_mdp('test_4_4')
#print("stop generating mdp")
# minmax = minmax_regret(_mdp, [-1, 1])
#minmax.solve_stochastic_opt(0.01)
#minmax.solve_deterministic_opt(60, 0.01)