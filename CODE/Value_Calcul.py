import numpy as np

ftype = np.float32

class Value_iteration:

    def __init__(self, _mdp, reward_bounds):

        self.mdp = _mdp
        self.reward_bounds = [reward_bounds] * (self.mdp.nstates * self.mdp.nactions)

        print self.reward_bounds

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

            if delta < epsilon * (1 - gamma) / gamma:
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

            if delta < epsilon * (1 - gamma) / gamma:
                return Q.tolist()

