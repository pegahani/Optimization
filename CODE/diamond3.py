import sys
from classic_mdp import create_diamond_MDP
from minmax_regret import minmax_regret



def main():
    # h = Display_mdp( _file= '../MDPs/ex2.txt')#, _nstate= 4, _naction= 5)
    # #h.load(1)
    # h.reload(1)
    # h.save_mdp()

    text_file = open("../DATA/results/diamond_3.txt", "w")
    reward_type = 0
    for prob_low in range (5,50,5):
                    seed = 0
                    print reward_type, prob_low, seed
                    mdp = create_diamond_MDP(2, prob_low / 100.0, 1.0 - prob_low / 100.0, reward_type)
                    # load_mdp(3, 2, 0.9, 1,-1.0, 1.0)
                    # mdp = reload_mdp(3, 2, 0.9, 1,-1.0, 1.0)

                    minmax = minmax_regret(mdp)
                    minmax.solve_deterministic_opt_stack(36000, 0.01, False)

                    print "minimax regert deterministic optimal   ", minmax.UB
                    print "minimax regert stochastic optimal      ", minmax.ROOT_LB
                    print "minimax regert heuristic deterministic ", minmax.UB_HEUR

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


    text_file.close()
    pass


if __name__ == "__main__":
    # execute only if run as a script
    main()
