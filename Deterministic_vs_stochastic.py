from minmax_regret import reload_mdp, load_mdp, minmax_regret


def main(_state, _action):
    load_mdp(4, 4, 0.9, './Models/test' + str(_state) + str(_action))
    _mdp = reload_mdp('./Models/test' + str(_state) + str(_action))
    minmax = minmax_regret(_mdp, [-1, 1])
    stochastic_result = minmax.solve_stochastic_opt(0.01)
    determistic_result = minmax.solve_deterministic_opt(0.01)

    pass



if __name__ == '__main__':

    #for _nstate in range(5,51):
    #    for _naction in range(2,11):
    #        main(_nstate, _naction)

    main(4,4)
