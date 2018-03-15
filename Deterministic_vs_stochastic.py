from minmax_regret import reload_mdp, load_mdp, minmax_regret



def main():
    text_file = open("./results/deterministic_vs_stochastic.txt", "w")
    _gamma = 0.9
    for _state in range(4,5): #51
        for _action in range(2,3): #11
            for _seed in range(1,2):
                #load_mdp(_state, _action, _gamma, _seed)
                _mdp = reload_mdp(_state, _action, _gamma, _seed)
                minmax = minmax_regret(_mdp, [-1, 1])
                text_file.write(str(_state)+ ';' + str(_action) + ';' + str(_seed) + ';')
                minmax.solve_deterministic_opt(60,0.01)
                text_file.write( str(minmax.UB) #optimal stochastic policy value
                     + ';' + str(minmax.ROOT_LB)  #optimal deterministic policy value
                     + ';' + str(minmax.BB_tot_nodes) #total number of BB nodes
                     + ';' + str(minmax.BB_nodes_pruned) # total nodes pruned for BB
                     + ';' + str(minmax.MASTER_tot_cuts) # total cuts on BB nodes
                     + ';' + str(minmax.ROOT_tot_cuts) #total number of constraints <g,r> added to solve master program
                     + ';' + str(minmax.TIME_master) # total computing time for master
                     + ";" + str(minmax.TIME_slave) # total computing time for slave
                     + ';' + str(minmax.TIME_root) # total computing time for benders decomposition
                     + ';' + str(minmax.TIME_limit_reached) # flag equal to true if time limit is reached
                     )
                text_file.write(minmax.output)
                text_file.write('\n')

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
