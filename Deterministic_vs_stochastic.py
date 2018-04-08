from minmax_regret import reload_mdp, load_mdp, minmax_regret
import sys

def main():
    text_file = open("./results/deterministic_vs_stochastic_controesempio.txt", "w")
    #text_file = open("./results/deterministic_vs_stochastic_fango_07_04_2018.txt", "w")
    #text_file = open("./results/deterministic_vs_stochastic.txt", "w")
    #text_file = open("./results/deterministic_vs_stochastic_clear.txt", "w")
    _gamma = 0.9
    for _action in range(20,55,5): #11
        #for _state in range(3,20): #51
        for _state in range(3,5): #51
            for _seed in range(1,11):
#                print "**************************"
#                print "**************************"
                print "_state _action, _gamma, _seed : " , _state, " , " , _action, " , " , _gamma, " , " , _seed
#                print "**************************"
#                print "**************************"
                load_mdp(_state, _action, _gamma, _seed, _reward_lb= -10, _reward_up= 10 )
                _mdp = reload_mdp(_state, _action, _gamma, _seed, _reward_lb= -10, _reward_up= 10)
                #_mdp.modify_mdp()
                minmax = minmax_regret(_mdp, [-1, 1])
                
                for cut_every_node in {True, False}:
                    minmax.solve_deterministic_opt_stack(3600,0.01,cut_every_node)
                
                    #if (minmax.controesempio ==True and ( ( minmax.UB_HEUR - minmax.UB ) / minmax.UB ) > 0.01 ):    
                    text_file.write(str(_state) + ';' + str(_action) + ';' + str(_seed) + ';')
                    text_file.write(str(minmax.UB) #optimal deterministic policy value
                         + ';' + str(minmax.UB_HEUR)  #heuristic deterministic policy value
                         + ';' + str(minmax.ROOT_LB)  #optimal stochastic policy value
                         + ';' + str(minmax.BB_tot_nodes) #total number of BB nodes
                         + ';' + str(minmax.BB_nodes_pruned) # total nodes pruned for BB
                         + ';' + str(minmax.MASTER_tot_cuts) # total cuts on BB nodes
                         + ';' + str(minmax.ROOT_tot_cuts) #total number of constraints <g,r> added to solve master program
                         + ';' + str(minmax.TIME_master) # total computing time for master
                         + ";" + str(minmax.TIME_slave) # total computing time for slave
                         + ';' + str(minmax.TIME_root) # total computing time for benders decomposition
                         + ';' + str(minmax.TIME_limit_reached) # flag equal to true if time limit is reached
                         + ';' + str(minmax.controesempio) # is the optimal stochastic solution different from the heuristic?
                         + ';' + str(cut_every_node) # 
                         )
                    text_file.write(minmax.output)
                    text_file.write('\n')
                    text_file.flush()
                    sys.stdout.flush()

    _mdp.display_mdp()

    text_file.close()
    pass


if __name__ == '__main__':


    main()
    
#    _state, _action, _gamma, _seed = 10, 2, 0.9, 2
#    #load_mdp(_state, _action, _gamma, _seed)
#    _mdp = reload_mdp(_state, _action, _gamma, _seed)
#    minmax = minmax_regret(_mdp, [-1, 1])
#    text_file.write(str(_state) + ';' + str(_action) + ';' + str(_seed) + ';')
#    minmax.solve_deterministic_opt(0.04,0.01)
#    text_file.write(str(minmax.UB)  # optimal stochastic policy value
#                    + ';' + str(minmax.ROOT_LB)  # optimal deterministic policy value
#                    + ';' + str(minmax.BB_tot_nodes)  # total number of BB nodes
#                    + ';' + str(minmax.BB_nodes_pruned)  # total nodes pruned for BB
#                    + ';' + str(minmax.MASTER_tot_cuts)  # total cuts on BB nodes
#                    + ';' + str(minmax.ROOT_tot_cuts)  # total number of constraints <g,r> added to solve master program
#                    + ';' + str(minmax.TIME_master)  # total computing time for master
#                    + ";" + str(minmax.TIME_slave)  # total computing time for slave
#                    + ';' + str(minmax.TIME_root)  # total computing time for benders decomposition
#                    + ';' + str(minmax.TIME_limit_reached) # flag equal to true if time limit is reached
#                    )
#    text_file.write(minmax.output)
#    text_file.write('\n')
#
#    text_file.close()

