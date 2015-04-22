# searchAgents.py
# ---------------
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


"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from game import Directions
from game import Agent
from game import Actions
import util
import time
import search

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError, fn + ' is not a search function in search.py.'
        func = getattr(search, fn)
        if 'heuristic' not in func.func_code.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print 'Warning: no food in corner ' + str(corner)
        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded
        ##
        ## Creamos 4 varialbes booleanas que nos indiquen si las esquinas
        ## han sido visitadas o no.
        self.topL,self.topR,self.botL,self.botR = False,False,False,False
        ## Consideramos el caso de que la posicion inicial sea uno de estas
        ## esquinas que queremos encontrar. En caso de serlo marcamos el
        ## booleano correspondiente como True.
        if self.startingPosition == self.corners[0]: # 1,1
            self.botL = True
        elif self.startingPosition == self.corners[1]: # 1,top
            self.topL = True
        elif self.startingPosition == self.corners[2]: # right,1
            self.botL = True
        elif self.startingPosition == self.corners[3]: # right,top
            self.topR = True

    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        ## Devuelve la posicion inicial y los booleanos que indican si se han
        ## visitado las esquinas o no. Al ser la posicion inicial lo mas probable
        ## es que los 4 booleanos sean False salvo que la posicion incial sea una
        ## de las esquinas.
        return (self.startingPosition, self.topL, self.topR, self.botL, self.botR)

    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """
        ## Devolvemos si el estado actual es el estado objectivo.
        ## Sera el estado objectivo cuando los 4 booleanos que indican si
        ## hemos visitado cada una de las esquinas sean todos True.
        position,topL,topR,botL,botR = state
        if topL and topR and botL and botR:
            return True
        return False

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """
        ## Creamos la lista de sucesores del nodo dado.
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            ## Guardamos los maximos valores que se pueden alcanzar horizontal y
            ## verticalmente para un uso mas comodo.
            top,right = self.walls.height-2, self.walls.width-2
            ## Guardamos el estado actual.
            position, topL, topR, botL, botR = state
            ## Separamos la posicion en componentes para comodidad.
            x,y = position
            ## Calculamos el vector direccion.
            dx,dy = Actions.directionToVector(action)
            ## Hallamos la siguiente posicion sumando el vector a el nodo actual.
            nextx,nexty = int(x+dx),int(y+dy)
            ## Comprobamos si la nueva posicion es una pared (invalida) o no (valida).
            hitsWall = self.walls[nextx][nexty]
            ## Si no es una pared la guardamos y, en caso de que sea alguna de las
            ## esquinas, activamos el booleano correspondiente para indicar que hemos
            ## llegado a tal esquina.
            if not hitsWall:
                nextposition = (nextx,nexty)
                if nextposition == (1,1):
                    botL = True
                elif nextposition == (1,top):
                    topL = True
                elif nextposition == (right,1):
                    botR = True
                elif nextposition == (right,top):
                    topR = True
                ## Creamos un nuevo estado.
                nextState = (nextposition, topL, topR, botL, botR)
                ## Guardamos el estado como sucesor del nodo padre.
                ## Usamos un valor de 1 para todos ya que no estamos usando
                ## ninguna heuristica en concreto.
                successors.append((nextState,action,1))
        self._expanded += 1 # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible (as well as consistent).
    """
    ## Usaremos la heuristica Manhattan.
    ## Guardamos los valores de top y right del problema por comodidad.
    top,right = problem.corners[3][1],problem.corners[3][0]
    walls = problem.walls
    ## Cargamos el estado.
    position, topL, topR, botL, botR = state
    ## Guardamos en una lista los booleanos que indican las esquinas que hemos
    ## visitado
    cornersVisited = [topL,topR,botL,botR]
    ## Guardamos en una lista las coordenadas de las esquinas.
    cornersCoord = [(1,top),(right,top),(1,1),(right,1)]
    ## Inicializamos el valor de la heuristica y del total.
    heuristicVal = 0
    total = 0
    ## Mientras queden esquinas por visitar, es decir, que alguno de 
    ## los elementos de la matrix sea False.
    while cornersVisited!=[True,True,True,True]:
        ## Separamos la posicion en componentes por comodidad.
        x,y = position
        ## Creamos una fita superior. Esta es la distancia maxima que puede
        ## asolir la heuristica, representa la aplicacion de la funcion heuristica
        ## a los dos estados de esquinas opuestas.
        heuristicVal = top + right
        ## Para cada uno de las 4 esquinas, si no han sido visitadas, calculamos
        ## el valor de la heuristica y guardamos la menor de todas. Este sera la esquina
        ## a la que iremos.
        for i in range(len(cornersVisited)):
            if not cornersVisited[i]:
                ## La heuristica corresponde a la Manhattan. Es la suma del valor absoluto
                ## de las diferencias por componentes de las coordenadas de los dos estados.
                dx,dy = abs(x - cornersCoord[i][0]),abs(cornersCoord[i][1]-y)
                if dx+dy < heuristicVal:
                    ## Si la nueva es menor que la que tenemos, la sobreescribimos.
                    heuristicVal = dx+dy
                    ## Guardamos el indicador para saber que esquina es a la que corresponde
                    ## el minimo valor de la heuristica.
                    closestCorner = i
        ## Cargamos la variable posicion con las coordenadas del estado que ha dado el
        ## menor numero a la hora de aplicar la heuristica. A partir de este estado
        ## repetiremos el proceso hasta que las cuatro esquinas hayan sido visitadas.
        position = cornersCoord[closestCorner]
        ## Actualizamos el total sumando las heuristicas que vamos calculando por el camino.
        total += heuristicVal
        ## Indicamos que la esquina que sera nuestra siguiente posicion ha sido visitada.
        cornersVisited[closestCorner] = True
    ## Devolvemos el coste total de encontrar todas las esquinas.
    return total
    return 0 # Default to trivial solution

class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    ## La heuristica que utilizaremos es la siguiente:
    ## Del nodo actual, buscaremos el objetivo mas cercano tomando como heuristica
    ## la Manhattan. Usaremos esta heuristica ya que es la mas simple y optima para
    ## trabajar con elementos en forma de cuadricula.
    ## A partir del objetivo mas cercano, buscaremos, entre el resto de objetivos, el
    ## que mas lejos este, usando nuevamente la misma heuristica.
    ## Devolveremos como resultado la suma de los dos valores que hemos calculado.
    ## Esta heuristica expande 7798 nodos en el caso de el autograder.
    
    ## Cargamos la posicion actual i los objectivos (foods) que faltan por conseguir.
    position,foods = state
    ## Semparamos la posicion en componentes por comodidad.
    x,y = position[0],position[1]
    ## Guardamos los maximos valores de top y right por comodidad.
    top,right = problem.walls.height-2, problem.walls.width-2
    ## Convertimos foods en una lista.
    foods = foods.asList()
    ## Guardamos una fita superior para guardar la minima distancia.
    mindist = top + right
    ## Para todos los objetivos, calcularemos la minima distancia de la posicion
    ## actual a un objetivo.
    if len(foods) > 0:
        for food in foods:
            ## Utilizaremos la distancia Manhattan.
            ## Corresponde a la suma de los valores absolutos de las diferencias
            ## por componenetes de las coordenadas de ambos nodos.
            dist = abs(x - food[0]) + abs(y - food[1])
            ## Si la distancia es menor a la minima anterior la guardamos.
            if dist < mindist:
                ## Al acabar el for, mindist contendra la minima distancia de la
                ## posicion actual a un objetivo, y closestFood guardara las coordenadas
                ## de dicho objetivo.
                mindist = dist
                closestFood = food
    ## En caso de no haber objetivos retornamos la minima distancia como 0.
    else:
        mindist = 0
    ## Ahora vamos a calcular la maxima distancia que puede haber entre el objetivo
    ## que hemos guardado anteriormente com cualquiera del resto de los objetivos.
    ## En maxdist guardaremos la maxima distancia. La inicializamos a 0 para usarla
    ## como fita inferior.
    maxdist = 0
    ## Para cada objectivo en la lista de objetivos calculamos la distancia Manhattan
    ## entre el objetivo mas cercano a la posicion actual y el resto de objectivos.
    for food0 in foods:
        dist = abs(food0[0] - food[0]) + abs(food0[1] - food[1])
        ## Si la distancia es mayor que la maxima la guardamos.
        if dist > maxdist:
            ## Al acabar el for, maxdist contendra la maxima distancia entre el
            ## objetivo mas cercano a la posicion actual y el objetivo que mas lejos
            ## se encuentre de el.
            maxdist = dist
    ## Devolvemos la suma de la distancia de ir de la posicion actual a el objetivo
    ## mas cercano, mas la maxima distancia que hay entre el objectivo mas cercano
    ## y cualquier otro objetivo.
    return mindist + maxdist

class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception, 'findPathToClosestDot returned an illegal move: %s!\n%s' % t
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print 'Path found with cost %d.' % len(self.actions)

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
