import pylab as plt
import sys
import json

import pickle
import time

import my_mdp
from classic_mdp import general_random_mdp, create_diamond_MDP
from minmax_regret import minmax_regret, load_mdp, reload_mdp


class Display_mdp:

    def __init__(self, _nstate = None, _naction = None,  _file = None ):

        self.nstate = _nstate
        self.naction = _naction
        self.file = _file


    def load(self, _id):
        """
        Creates a new mdp, initialize related global variables and saves what is needed for reuse
        :type _id: string e.g. 80-1 to save in param80-1.dmp
        """
        self.mdp = my_mdp.random_mdp(self.nstate, self.naction, None)

        # if not _id is None:
        name = "param_" + str(_id) + ".dmp"
        pp = pickle.Pickler(open(name, 'w'))
        pp.dump(self.nstate)
        pp.dump(self.naction)
        pp.dump(self.mdp)

        pass

    def reload(self, _id):
        """
        Reloads a saved mdp and initialize related global variables
        :type _id: string e.g. 80-1 to reload param80-1.dmp
        """
        name = "param_" + str(_id) + ".dmp"
        pup = pickle.Unpickler(open(name, 'r'))

        self.nstate = pup.load()
        print('states', self.nstate)
        self.naction = pup.load()
        print('actions', self.naction)
        self.mdp = pup.load()
        print('mdp', self.mdp)

    def save_mdp(self):
        file = open(self.file, "w")
        file.write(str(self.nstate) + '\t' + str(self.naction) + '\n')

        for i in self.mdp.stateInd.iterkeys():

            actions_list = self.mdp._get_Actions(i)
            file.write('state = ' + str(i) + '\n')
            for _ac in actions_list:
                file.write('action = ' + str(_ac) + '\t')
                next_states_probabilities = [(self.mdp.stateInd[s2], self.mdp.transitions[(s, a, s2)]) for (s, a, s2) in
                                             self.mdp.transitions.iterkeys() if (self.mdp.stateInd[s] == i and self.mdp.actionInd[a] == _ac)]
                file.write(str(next_states_probabilities) + '\n')
                file.write('\n')


def main():
    # h = Display_mdp( _file= '../MDPs/ex2.txt')#, _nstate= 4, _naction= 5)
    # #h.load(1)
    # h.reload(1)
    # h.save_mdp()

    text_file = open("./results/diamond.txt", "w")

    for reward_type in [0,1,2,3]:
            for prob_low in range (5,50,5):
                if reward_type == 0:
                    seed = 0
                    print reward_type, prob_low, seed
                    mdp = create_diamond_MDP(2, prob_low / 100.0, 1.0 - prob_low / 100.0, reward_type)
                    # load_mdp(3, 2, 0.9, 1,-1.0, 1.0)
                    # mdp = reload_mdp(3, 2, 0.9, 1,-1.0, 1.0)

                    minmax = minmax_regret(mdp)
                    minmax.solve_deterministic_opt_stack(36000, 0.01, False)

                    print minmax.BEST_det_policy
                    print minmax.BB_tot_nodes

                    print "minimax regert deterministic optimal   ", minmax.UB
                    print "minimax regert stochastic optimal      ", minmax.ROOT_LB
                    print "minimax regert heuristic deterministic ", minmax.UB_HEUR

                    minmax.solve_deterministic_opt_stack(36000, 0.01, False)

                    # if (minmax.controesempio ==True and ( ( minmax.UB_HEUR - minmax.UB ) / minmax.UB ) > 0.01 ):
                    text_file.write(str(reward_type) + ';' + str(prob_low) + ';' + str(seed) + ';')
                    text_file.write(str(minmax.UB)  # optimal deterministic policy value
                                    + ';' + str(minmax.UB_HEUR)  # heuristic deterministic policy value
                                    + ';' + str(minmax.ROOT_LB)  # optimal stochastic policy value
                                    + ';' + str(minmax.BB_tot_nodes)  # total number of BB nodes
                                    + ';' + str(minmax.BB_nodes_pruned)  # total nodes pruned for BB
                                    + ';' + str(minmax.MASTER_tot_cuts)  # total cuts on BB nodes
                                    + ';' + str(
                        minmax.ROOT_tot_cuts)  # total number of constraints <g,r> added to solve master program
                                    + ';' + str(minmax.TIME_master)  # total computing time for master
                                    + ";" + str(minmax.TIME_slave)  # total computing time for slave
                                    + ';' + str(minmax.TIME_root)  # total computing time for benders decomposition
                                    + ';' + str(
                        minmax.TIME_limit_reached)  # flag equal to true if time limit is reached
                                    + ';' + str(
                        minmax.controesempio)  # is the optimal stochastic solution different from the heuristic?
                                    + '; FALSE'  #
                                    )
                    text_file.write(minmax.output)
                    text_file.write('\n')
                    text_file.flush()
                    sys.stdout.flush()

                else:
                    for seed in range(0,5):
                        print reward_type , prob_low, seed
                        mdp = create_diamond_MDP(2,prob_low/100.0, 1.0-prob_low/100.0,reward_type)
                        #load_mdp(3, 2, 0.9, 1,-1.0, 1.0)
                        #mdp = reload_mdp(3, 2, 0.9, 1,-1.0, 1.0)

                        minmax = minmax_regret(mdp)
                        minmax.solve_deterministic_opt_stack(36000, 0.01, False)

                        print minmax.BEST_det_policy
                        print minmax.BB_tot_nodes

                        print "minimax regert deterministic optimal   ", minmax.UB
                        print "minimax regert stochastic optimal      ", minmax.ROOT_LB
                        print "minimax regert heuristic deterministic ", minmax.UB_HEUR


                        minmax.solve_deterministic_opt_stack(36000, 0.01, False)

                        # if (minmax.controesempio ==True and ( ( minmax.UB_HEUR - minmax.UB ) / minmax.UB ) > 0.01 ):
                        text_file.write(str(reward_type) + ';' + str(prob_low) + ';' + str(seed) + ';')
                        text_file.write(str(minmax.UB)  # optimal deterministic policy value
                                        + ';' + str(minmax.UB_HEUR)  # heuristic deterministic policy value
                                        + ';' + str(minmax.ROOT_LB)  # optimal stochastic policy value
                                        + ';' + str(minmax.BB_tot_nodes)  # total number of BB nodes
                                        + ';' + str(minmax.BB_nodes_pruned)  # total nodes pruned for BB
                                        + ';' + str(minmax.MASTER_tot_cuts)  # total cuts on BB nodes
                                        + ';' + str(
                            minmax.ROOT_tot_cuts)  # total number of constraints <g,r> added to solve master program
                                        + ';' + str(minmax.TIME_master)  # total computing time for master
                                        + ";" + str(minmax.TIME_slave)  # total computing time for slave
                                        + ';' + str(minmax.TIME_root)  # total computing time for benders decomposition
                                        + ';' + str(
                            minmax.TIME_limit_reached)  # flag equal to true if time limit is reached
                                        + ';' + str(
                            minmax.controesempio)  # is the optimal stochastic solution different from the heuristic?
                                        + '; FALSE'   #
                                        )
                        text_file.write(minmax.output)
                        text_file.write('\n')
                        text_file.flush()
                        sys.stdout.flush()

    text_file.close()
    pass


if __name__ == "__main__":
    # execute only if run as a script
    main()
