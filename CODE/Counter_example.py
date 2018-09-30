from classic_mdp import mdp_counter_example
from minmax_regret import minmax_regret



def main():
    mdp = mdp_counter_example(0.49, 0.51, 100.0, 1.0, -0.001)
    minmax = minmax_regret(mdp)

    minmax.solve_deterministic_opt_stack(36000, 0.01, False)
    print "minimax regert deterministic optimal   ", minmax.UB
    print "minimax regert stochastic optimal      ", minmax.ROOT_LB
    print "minimax regert heuristic deterministic ", minmax.UB_HEUR
    pass


if __name__ == "__main__":
    # execute only if run as a script
    main()