import pylab as plt
import sys
import json

import pickle
import time

from optimization_approach import handle_mdps, my_mdp
from optimization_approach.classic_mdp import general_random_mdp


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

    _mdp = general_random_mdp(3, 2, 0.9)
    pass


if __name__ == "__main__":
    # execute only if run as a script
    main()
