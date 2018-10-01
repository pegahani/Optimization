from minmax_regret import reload_mdp, load_mdp, minmax_regret
import sys

def main(min_states, max_states, step_states, min_actions, max_actions,max_seeds):
    text_file = open("../DATA/results/deterministic_vs_stochastic_"
    + str(min_states)+"_"+ str(max_states)+"_"+ str(step_states)
    + str(min_actions)+"_"+str(max_actions)+"_"+str(max_seeds)+".txt", "w")
    #text_file = open("../DATA/results/deterministic_vs_stochastic_fango_07_04_2018.txt", "w")
    #text_file = open("../DATA/results/deterministic_vs_stochastic.txt", "w")
    #text_file = open("../DATA/results/deterministic_vs_stochastic_clear.txt", "w")
    _gamma = 0.9
    for _action in range(min_actions,max_actions+1): #11
        #for _state in range(3,20): #51
        for _state in range(min_states,max_states+step_states,step_states): #51
            for _seed in range(1,max_seeds+1):
#                print "**************************"
#                print "**************************"
                print "_state _action, _gamma, _seed : " , _state, " , " , _action, " , " , _gamma, " , " , _seed
#                print "**************************"
#                print "**************************"
                load_mdp(_state, _action, _gamma, _seed, _reward_lb= -10, _reward_up= 10, reward_on_state= False )
                _mdp = reload_mdp(_state, _action, _gamma, _seed, _reward_lb= -10, _reward_up= 10)
                #_mdp.modify_mdp()
                minmax = minmax_regret(_mdp)
                
                for cut_every_node in {True, False}:
                    minmax.solve_deterministic_opt_stack(36000,0.01,cut_every_node)
                
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

    # a = int(sys.argv[1])
    # b = int(sys.argv[2])
    # c = int(sys.argv[3])
    # d = int(sys.argv[4])
    # e = int(sys.argv[5])
    # f = int(sys.argv[6])
    #
    # main(a,b,c,d,e,f)

    main(5, 5, 1, 3, 3, 1)




