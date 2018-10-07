from classic_mdp import mdp_counter_example, trident_mdp
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
    mdp = trident_mdp(n_states = 5, _gamma = 0.9,_reward_lb = -10.0, _reward_up = 10.0, probability = 0.3, next_states = 3)
    mdp.display_mdp()

    minmax = minmax_regret(mdp)
    minmax.solve_deterministic_opt_stack(36000, 0.01)
