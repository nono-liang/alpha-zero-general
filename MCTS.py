import math
import numpy as np
EPS = 1e-8

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs


    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
            print('Es', self.Es[s])
            """
            begin recursive search, next_s
             [[-1  1  1  1  1  1]
             [-1 -1 -1 -1  1  1]
             [-1 -1 -1  1 -1  1]
             [-1 -1 -1 -1  1 -1]
             [-1 -1 -1 -1 -1 -1]
             [-1 -1 -1 -1 -1  1]]
            Es -1
            """
        if self.Es[s]!=0:
            # terminal node
            print('terminal, v', -self.Es[s])
            """
            terminal, v 1
            """
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            print('leaf, Ps\n', self.Ps[s], '\nv', v)
            """
            leaf, Ps
             [0.02686859 0.02613893 0.02653145 0.02728407 0.02770755 0.02686941
             0.02598217 0.02750654 0.02616615 0.02631035 0.02812076 0.02694663
             0.02664369 0.02628738 0.02577865 0.02826667 0.02805581 0.02641091
             0.02782914 0.02741454 0.0271013  0.0277985  0.02596407 0.02609453
             0.02803502 0.02790424 0.02751453 0.02790191 0.02647677 0.02724783
             0.02596787 0.02729226 0.02709833 0.02613078 0.02644561 0.02812006
             0.02778683] 
            v [-0.00393087]
            """
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s]*valids      # masking invalid moves
            print('leaf, Ps\n', self.Ps[s], '\nv', v)
            """
            leaf, Ps
             [0.         0.         0.         0.         0.         0.
             0.         0.         0.02616615 0.         0.         0.
             0.         0.02628738 0.         0.         0.         0.
             0.         0.         0.         0.         0.02596407 0.
             0.         0.         0.         0.02790191 0.         0.
             0.         0.         0.         0.         0.         0.
             0.        ] 
            v [-0.00393087]
            """
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    u = self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        print('begin recursive search, next_s\n', next_s)
        """
        begin recursive search, next_s
         [[ 0  0  0  0  0  0]
         [ 0  0  0  0  0  0]
         [ 0  0  1 -1  0  0]
         [ 0  0 -1 -1  0  0]
         [ 0  0  0 -1  0  0]
         [ 0  0  0  0  0  0]]
        """
        v = self.search(next_s)
        print('end recursive search, next_s\n', next_s, '\nv', v)
        """
        end recursive search, next_s
         [[ 0  0  0  0  0  0]
         [ 0  0  0  0  0  0]
         [ 0  0  1 -1  0  0]
         [ 0  0 -1 -1  0  0]
         [ 0  0  0 -1  0  0]
         [ 0  0  0  0  0  0]] 
        v [0.0039525]
        """

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1
            print('update Qsa', self.Qsa[(s,a)], '\nNsa', self.Nsa[(s,a)])

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1
            print('init Qsa', self.Qsa[(s, a)], '\nNsa', self.Nsa[(s, a)])
            """
            init Qsa [0.0039525] 
            Nsa 1
            """

        self.Ns[s] += 1
        return -v
