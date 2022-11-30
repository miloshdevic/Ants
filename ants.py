# Chaima Boussorra 20159909
# Milosh Devic 20158232
import numpy as np
import random as rand


class Colony:
    class Ant:
        def __init__(self, colony):
            self.colony = colony
            self.pos = rand.randrange(self.colony.n)

            self.mem = np.zeros(self.colony.n)
            self.mem[self.pos] = 1

            self.path = [self.pos]
            self.cost = 0

        def reset(self, colony):
            self.__init__(colony)

        def __str__(self):
            pathPrint = "["
            for i in range(len(self.path)-1):
                pathPrint += str(self.path[i]) + ", "

            pathPrint += str(self.path[-1]) + "]"
            return pathPrint

        # déterminer fourmi pour qui ça a couté le moins cher
        def __lt__(self, other):
            return self.cost <= other.cost

        # Returns city to be travelled to from current position
        def policy(self):
            next_city = 0  # prochaine destination
            if rand.random() < self.colony.q_0:
                # Deterministic decision
                arg_max = 0
                for s in range(self.colony.n):
                    if self.mem[s] == 0:
                        formuleDecision = self.colony.tau[self.path[-1]][s] * \
                                  ((self.colony.eta(self.path[-1], s))**self.colony.beta)
                        if formuleDecision > arg_max:
                            arg_max = formuleDecision
                            next_city = s
                return next_city
            else:
                # Stochastic decision
                proba_de_prendre = []  # probabilités de d'être choisi (chaque indexe représente un sommet)
                denominateur = 0
                for u in range(self.colony.n):
                    if self.mem[u] == 0:
                        denominateur += self.colony.tau[self.path[-1]][u] * \
                                        ((self.colony.eta(self.path[-1], u))**self.colony.beta)

                for s in range(self.colony.n):
                    if self.mem[s] == 0:
                        nominateur = self.colony.tau[self.path[-1]][s] * \
                              (self.colony.eta(self.path[-1], s))**self.colony.beta
                        formuleDecision = nominateur/denominateur
                        proba_de_prendre.append(formuleDecision)
                    else:
                        proba_de_prendre.append(0)
                return np.random.choice(np.arange(self.colony.n), p=proba_de_prendre)

        # Updates the local pheromones and position of ant
        # while keeping track of total cost and path
        def move(self):
            destination = self.policy()  # prochain noeud à visiter

            # local updating
            # la matrice est symétrique donc il faut update aussi les coordonnées inversées
            self.colony.tau[self.path[-1]][destination] = \
                (1 - self.colony.alpha) * self.colony.tau[self.path[-1]][destination] + \
                self.colony.alpha * self.colony.tau_0
            self.colony.tau[destination][self.path[-1]] = \
                (1 - self.colony.alpha) * self.colony.tau[destination][self.path[-1]] + \
                self.colony.alpha * self.colony.tau_0

            # Change position
            if len(self.path) != 0:
                self.cost += self.colony.adjMat[self.path[-1]][destination]
            self.mem[destination] = 1
            self.path.append(destination)
            if len(self.path) == self.colony.n:
                self.colony.tau[self.path[-1]][self.path[0]] = \
                    (1 - self.colony.alpha) * self.colony.tau[self.path[-1]][self.path[0]] + \
                    self.colony.alpha * self.colony.tau_0
                self.colony.tau[self.path[0]][self.path[-1]] = \
                    (1 - self.colony.alpha) * self.colony.tau[self.path[-1]][self.path[-1]] + \
                    self.colony.alpha * self.colony.tau_0
                self.cost += self.colony.adjMat[self.path[-1]][self.path[0]]

        # Updates the pheromone levels of ALL edges that form
        # the minimum cost loop at each iteration
        def globalUpdate(self):
            for s in range(len(self.path)):
                # la matrice est symétrique donc il faut update aussi les coordonnées inversées
                # on utilise le modulo pour avoir le coût du dernier au premier aussi
                self.colony.tau[self.path[s]][self.path[(s + 1) % self.colony.n]] = \
                    (1 - self.colony.alpha) * \
                    self.colony.tau[self.path[s]][self.path[(s + 1) % self.colony.n]] + \
                    (self.colony.alpha * (1 / self.cost))
                self.colony.tau[self.path[(s + 1) % self.colony.n]][self.path[s]] = \
                    (1 - self.colony.alpha) * \
                    self.colony.tau[self.path[(s + 1) % self.colony.n]][self.path[s]] + \
                    (self.colony.alpha * (1 / self.cost))

            print(self, ", costs:", self.cost)

    def __init__(self, adjMat, m=10, beta=2, alpha=0.1, q_0=0.9):
        # Parameters:
        # m => Number of ants
        # beta => Importance of heuristic function vs pheromone trail
        # alpha => Updating propensity
        # q_0 => Probability of making a non-stochastic decision
        # tau_0 => Initial pheromone level

        self.adjMat = adjMat
        self.n = len(adjMat)

        self.tau_0 = 1 / (self.n * self.nearestNearbourHeuristic())
        self.tau = [[self.tau_0 for _ in range(self.n)] for _ in range(self.n)]
        self.ants = [self.Ant(self) for _ in range(m)]

        self.beta = beta
        self.alpha = 0.1
        self.q_0 = q_0

    def __str__(self):
        return "Nearest Nearbour Heuristic cost:" + str(self.nearestNearbourHeuristic())

    # Returns the cost of the solution produced by
    # the nearest neighbour heuristix
    def nearestNearbourHeuristic(self):
        costs = np.zeros(self.n)

        for i in range(self.n):
            somme_cout = 0
            noeud_courant = i
            forbidden = [i]  # pour ne pas retourner sur un noeud deja visite
            for _ in range(self.n):
                next_node = 0  # garder compte de l'indexe du noeud avec la distance minimale
                dist_min = float("inf")

                # chercher la plus petite distance avec un noeud non parcourue
                for k in range(self.n):
                    if (self.adjMat[noeud_courant][k] != 0) and (self.adjMat[noeud_courant][k] < dist_min) and \
                            (k not in forbidden):
                        dist_min = self.adjMat[noeud_courant][k]
                        next_node = k
                #ajouter la distance du dernier au premier pour avoir un cycle
                if dist_min == float("inf"):
                    dist_min = self.adjMat[noeud_courant][i]
                somme_cout += dist_min
                forbidden.append(next_node)
                noeud_courant = next_node  # passer au noeud avec la distance la plus petite

            costs[i] = somme_cout
        return min(costs)

    # Heuristic function
    # Returns inverse of smallest distance between r and u
    def eta(self, r, u):
        return 1 / self.adjMat[r][u]

    def optimize(self, num_iter):
        for _ in range(num_iter):
            for _ in range(self.n - 1):
                for ant in self.ants:
                    ant.move()

            min(self.ants).globalUpdate()

            for ant in self.ants:
                ant.reset(self)


if __name__ == "__main__":
    rand.seed(420)

    # file = open('d198')
    file = open('dantzig.csv')

    adjMat = np.loadtxt(file, delimiter=",")

    ant_colony = Colony(adjMat)

    ant_colony.optimize(1000)
