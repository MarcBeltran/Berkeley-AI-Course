# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # Todos los estados del grid.
        states = self.mdp.getStates()
        # Para cada iteracion:
        for i in range(iterations):
            # Creamos una serie de valores vacios
            valuesCopy = self.values.copy()
            # Para cada estado:
            for state in states:
                maxVal = None
                # Calculamos los QValues del estado con respecto a todas
                # las acciones legales desde dicho estado. Vamos guardando
                # y sobreescribiendo si procede el QValue maximo.
                for action in self.mdp.getPossibleActions(state):
                    val = self.computeQValueFromValues(state,action)
                    if maxVal<val or maxVal == None:
                        maxVal = val
                # Si no se ha asignado ningun valor valido, asignamos 0.
                if maxVal == None:
                    valuesCopy[state] = 0
                # Guardamos el maximo valor a la posicio del correspondiente
                # estado.
                else:
                    valuesCopy[state] = maxVal
            self.values = valuesCopy


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Inicializamos el valor.
        val = 0
        # Cogemos las probabilidades de ir de nuestro estado al siguiente por
        # medio de la accion.
        transitionSP = self.mdp.getTransitionStatesAndProbs(state,action)
        # Para cada siguiente estado y su probabilidad, sumamos al valor
        # siguiendo la formula vista en clase.
        for nextState, prob in transitionSP:
            val += prob * (self.mdp.getReward(state, action, nextState) + (self.discount * self.values[nextState]))
        return val

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Seleccionamos todas las acciones possibles.
        actions = self.mdp.getPossibleActions(state)
        # Si no hay ninguna accion possible retornamos None.
        if not actions:
            return None
        # maxVal guardara el QValue mayor de entre todas las
        # acciones posibles.
        # maxAction guardara la accion a la que corresponde este
        # maximo valor.
        maxVal = None
        maxAction = None
        for action in actions:
            aux = self.computeQValueFromValues(state, action)
            if aux>maxVal or maxVal == None:
                maxVal = aux
                maxAction = action
        return maxAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
