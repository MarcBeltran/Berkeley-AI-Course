# -*- coding: utf-8 -*-
# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ## Queremos que el pacman nunca se pare, por tanto si la acion es la
        ## de pararse (Stop) le damos el minimo de puntos posibles.
        if action == "Stop":
            return float("-inf")
        ## Guardamos en un array las distancias a los fantasmas.
        ghostDist = []
        for ghost in newGhostStates:
            ghostDist.append(manhattanDistance(newPos,ghost.getPosition()))
        ## Si realizar la accion causa que el fantasma muera le damos el minimo
        ## de puntos possibles para evitar que pase.
        if min(ghostDist)<=1:
            return float("-inf")
        ## Guardamos en un array las distancias a la comida.
        foodDist = []
        for food in newFood.asList():
            foodDist.append(manhattanDistance(newPos,food))
        ## La estrategia para la funcion consiste en atribuir mas puntos
        ## cuanta menos comida haya y cuanto mas cerca este la siguiente comida
        ## mas cercana. Asi, conseguimos que el pacman busque la comida.
        if not len(foodDist)==0:
            return 1000-min(foodDist)-100*newFood.count(True)
        ## Si no quedan comidas entonces retornamos el maximo valor posible.
        return float("inf")

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    """
        Implementamos el minimax siguiendo el pseudocodigo siguiente:
        function minimax(nodo, profundidad, deboMaximizar):
            if profundidad = 0 or nodo es terminal
                return evaluación heurística del nodo
            if deboMaximizar
                α := -∞
                for each siguiente accion posible
                    α := max(α, minimax(hijo, profundidad - 1, False))
                return α
            else
                β := +∞
                for each siguiente accion posible
                    β := min(β, minimax(hijo, profundidad - 1, True))
                return β
        Cambiando:
        - En lugar de deboMaximizar, pasamos el numero del agente y la cantidad
        de agentes. El numero de agente funciona como el deboMaximizar, ya que
        el 0(pacman) correspondra al True y el resto de numeros (fantasmas)
        correspondran al False. La cantidad de agentes la usamos para encontrar
        el siguiente agente a expandir.
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        ## Acciones legales del agente 0 (pacman).
        legalActions = gameState.getLegalActions(0)
        numAgents = gameState.getNumAgents()
        bestVal = float("-inf")
        ## Profundidad real, self.depth no tiene en cuenta la cantidad de agentes.
        realDepth = numAgents*self.depth
        for action in legalActions:
            val = self.minimax(gameState.generateSuccessor(0,action), realDepth-1, 1, numAgents)
            if val > bestVal:
                bestVal = val
                bestAction = action
        return bestAction

    def minimax(self, state, depth, agent, numAgents):
        ## Si el nodo es terminal, retornamos la funcion de evaluacion.
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        legalActions = state.getLegalActions(agent)
        ## Si el agente es el PACMAN, usamos el max.
        if agent == 0:
            alpha = float("-inf")
            for action in legalActions:
				## (agent+1)%numAgents retornara el siguiente agent excepto cuandos sea el ultimo, entonces volvera al pacman.
                alpha = max(alpha, self.minimax(state.generateSuccessor(agent,action), depth-1, (agent+1)%numAgents, numAgents))
            return alpha
        ## Si el agente es un fantasma, usamos el min.
        else:
            beta = float("inf")
            for action in legalActions:
				## (agent+1)%numAgents retornara el siguiente agent excepto cuandos sea el ultimo, entonces volvera al pacman.
                beta = min(beta, self.minimax(state.generateSuccessor(agent,action), depth-1, (agent+1)%numAgents, numAgents))
            return beta
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        legalActions = gameState.getLegalActions(0)
        numAgents = gameState.getNumAgents()
        bestVal = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        realDepth = numAgents*self.depth
        for action in legalActions:
            val = self.minimaxPoda(gameState.generateSuccessor(0,action), realDepth-1, 1, numAgents, alpha, beta)
            if val > bestVal:
                bestVal = val
                bestAction = action
	    # Poda de sucesores del nodo inicial.
	    # Si el valor es mayor que beta, podamos.
            if val > beta:
                return bestAction
            alpha = max(alpha, val)
        return bestAction

    ## La estuctura es similar a la del minimax. Como modificacion, ponemos
    ## condiciones a partir de las cuales realizar la poda.
    def minimaxPoda(self, state, depth, agent, numAgents, alpha, beta):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        legalActions = state.getLegalActions(agent)
        if agent == 0:
            val = float("-inf")
            for action in legalActions:
                val = max(val, self.minimaxPoda(state.generateSuccessor(agent,action), depth-1, (agent+1)%numAgents, numAgents, alpha, beta))
                ## Poda del max.
                if beta < val:
                    return val
                alpha = max(alpha, val)
            return val
        else:
            val = float("inf")
            for action in legalActions:
                val = min(val, self.minimaxPoda(state.generateSuccessor(agent,action), depth-1, (agent+1)%numAgents, numAgents, alpha, beta))
                ## Poda del min.
                if alpha > val:
                    return val
                beta = min(beta, val)
            return val
    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    ## La estructura es igual que la del minimax. La diferencia esta comentada.

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        legalActions = gameState.getLegalActions(0)
        numAgents = gameState.getNumAgents()
        bestVal = float("-inf")
        realDepth = numAgents*self.depth
        for action in legalActions:
            val = self.expectimax(gameState.generateSuccessor(0,action), realDepth-1, 1, numAgents)
            if val > bestVal:
                bestVal = val
                bestAction = action
        return bestAction

    def expectimax(self, state, depth, agent, numAgents):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        legalActions = state.getLegalActions(agent)
        if agent == 0:
            alpha = float("-inf")
            for action in legalActions:
                alpha = max(alpha, self.expectimax(state.generateSuccessor(agent,action), depth-1, (agent+1)%numAgents, numAgents))
            return alpha
        else:
            beta = 0
            numActions = 0
	    ## En lugar de calcular el min como anteriormente, calculamos la media de los resultados de los movimientos sucesores.
            for action in legalActions:
                numActions += 1
                beta = beta + self.expectimax(state.generateSuccessor(agent,action), depth-1, (agent+1)%numAgents, numAgents)
            return float(beta/numActions)
        
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    ## Cargamos variables que usaremos para realizar la funcion.
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newCapsules = currentGameState.getCapsules()

    ## Creamos una lista con las distancias a los fantasmas.
    ghostDist = []
    for ghost in newGhostStates:
        ghostDist.append(manhattanDistance(newPos,ghost.getPosition()))
    ## Creamos una lista con las distancias a las comidas.
    foodDist = []
    for food in newFood.asList():
        foodDist.append(manhattanDistance(newPos,food))
        
    t = 0 # Almacenara el valor de la funcion. Se ira modificando.
    ## Si el fantasma esta al lado y no es comible, retornamos un valor
    ## minimo para evitar que vaya.
    if min(ghostDist)<=1 and min(newScaredTimes)<=2:
        return float("-inf")
    ## Sumamos el Score ponderado.
    t += currentGameState.getScore()*100
    ## Si los fantasmas estan asustados, recomensamos estar cerca de el mas cercano.
    if min(newScaredTimes) > 0:
        t -= min(ghostDist)*10000
    ## Si no estan asustados, penalizamos estar cerca de el mas cercano.
    else:
        t += min(ghostDist)*10
    ## Recomensamos que quede menos comida.
    t -= newFood.count(True)*50
    ## Recompensamos estar cerca de la comida mas cercana.
    if len(foodDist)!=0:
        t -= min(foodDist)*100
    ## No quedan comidas.
    else:
        return float("inf")
    ## Recomensamos que queden menos capsulas. Ponderado muy alto para que vaya
    ## a buscarlas ya que da mas puntos comer fantasmas.
    t -= len(newCapsules)*100000
    return t

class BoundedIntelligenceMaxAgent(MultiAgentSearchAgent):
    ## La estructura global es similar al minmax.
    def getAction(self, gameState):
        legalActions = gameState.getLegalActions(0)
        numAgents = gameState.getNumAgents()
        bestVal = float("-inf")
        realDepth = numAgents*self.depth
        for action in legalActions:
            val = self.expectimax(gameState.generateSuccessor(0,action), realDepth-1, 1, numAgents)
            if val > bestVal:
                bestVal = val
                bestAction = action
        return bestAction

    def expectimax(self, state, depth, agent, numAgents):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        legalActions = state.getLegalActions(agent)
        if agent == 0:
            alpha = float("-inf")
            for action in legalActions:
                alpha = max(alpha, self.expectimax(state.generateSuccessor(agent,action), depth-1, (agent+1)%numAgents, numAgents))
            return alpha
        ## Modificamos la parte de los fantasmas.
        else:
            beta = 0
            numActions = 0
            ## Creamos un array donde guardaremos los valores de las acciones legales.
            l = []
            for action in legalActions:
                numActions += 1
                ## Guardamos en la lista el valor de cada accion.
                l.append(self.expectimax(state.generateSuccessor(agent,action), depth-1, (agent+1)%numAgents, numAgents))
            ## Ordenamos la lista de menor a mayor.
            l.sort()
            ## La variable suma almacenara el total de las sumas ponderadas.
            suma = 0
            ## Para cada accion legal, calculamos su ponderacion y la multiplicamos por el valor.
            for i in range(len(l)):
                suma += 2*l[i]*(len(l)-i)/(len(l)*(len(l)+1))
            return suma
        

# Abbreviation
better = betterEvaluationFunction

