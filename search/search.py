# -*- coding: cp1252 -*-
# search.py
# ---------
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
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    '''
    DFS. Search the deepest nodes in the search tree first.
    '''
    ## Vamos a explorar los nodos en profuncidad primero por tanto
    ## utilizaremos una pila para guardar los nodos a explorar.
    ## Ademas, guardaremos el camino realizado para llegar a cada
    ## nodo para despues poder devolverlo una vez hayamos llegado
    ## al destino. Para evitar visitar el mismo nodo varias veces,
    ## guardaremos los nodos visitados en una lista.
    stack = util.Stack()
    ## Cada elemento de la pila sera una tupla (nodo, camino).
    stack.push((problem.getStartState(),[]))
    ## En el set visited guardaremos los nodos que hayamos visitado
    ## en forma de tupla.
    visited = set()
    ## Mientras la pila no este vacia, haremos pop para sacar el
    ## elemento mas reciente y evaluarlo.
    while not stack.isEmpty():
        node, path = stack.pop()
        ## Una vez hecho el pop, podemos meterlo en la lista de visitados.
        visited.add(node)
        ## Si el nodo es el destino ya hemos terminado. Devolvemos el
        ## camino que hemos seguido hasta llegar a dicho nodo.
        if(problem.isGoalState(node)):
            return path
        ## Para cada nodo hijo, si no ha sido ya visitado, lo metemos
        ## en la pila junto a el camino que hemos hecho hasta llegar
        ## a dicho nodo.
        for coord, direction, steps in problem.getSuccessors(node):
            if not coord in visited:
                stack.push((coord,path+[direction]))
    ## No deberia llegar nunca aqui.
    return []  
 
def dfsLimit(problem,limit=70):
    '''
        NOTA: En comentarios entre 3 ' los comentarios relacionados
        con la modificacion del DFS. Los comentarios en ## son
        equivalentes a los comentarios del DFS normal.
    '''
    ## Vamos a explorar los nodos en profuncidad primero por tanto
    ## utilizaremos una pila para guardar los nodos a explorar.
    ## Ademas, guardaremos el camino realizado para llegar a cada
    ## nodo para despues poder devolverlo una vez hayamos llegado
    ## al destino. Para evitar visitar el mismo nodo varias veces,
    ## guardaremos los nodos visitados en una lista.
    stack = util.Stack()
    ## Cada elemento de la pila sera una tupla (nodo, camino).
    stack.push((problem.getStartState(),[]))
    ## En el set visited guardaremos los nodos que hayamos visitado
    ## en forma de tupla.
    visited = set()
    ## Mientras la pila no este vacia, haremos pop para sacar el
    ## elemento mas reciente y evaluarlo.
    while not (stack.isEmpty()):
        node, path = stack.pop()
        '''
            Para evitar que supere el limite de profundidad, guardamos la
            longitud del camino, es decir, la profuncidad del nodo.
        '''
        i = len(path)
        ## Una vez hecho el pop, podemos meterlo en la lista de visitados.
        visited.add(node)
        ## Si el nodo es el destino ya hemos terminado. Devolvemos el
        ## camino que hemos seguido hasta llegar a dicho nodo.
        if(problem.isGoalState(node)):
            return path
        ## Para cada nodo hijo, si no ha sido ya visitado, lo metemos
        ## en la pila junto a el camino que hemos hecho hasta llegar
        ## a dicho nodo.
        for coord, direction, steps in problem.getSuccessors(node):
            '''
                Si la profundidad es superior a el limite establecido, no
                metemos el nodo en la pila. Asi evitamos que se expandan
                nodos con profundidad mayor que la establecida en el limite.
            '''
            if not coord in visited and i<=limit:
                stack.push((coord,path+[direction]))
    '''
        Retornamos una lista vacia cuando, por culpa del valor del limite,
        el DFS no es capaz de encontrar una solcuion.
    '''
    return []  
 

def breadthFirstSearch(problem):
    '''
    BFS. Search the shallowest nodes in the search tree first.
    '''
    ## Vamos a explorar los nodos en anchura primero por tanto
    ## utilizaremos una cola para guardar los nodos a explorar.
    ## Ademas, guardaremos el camino realizado para llegar a cada
    ## nodo para despues poder devolverlo una vez hayamos llegado
    ## al destino. Para evitar visitar el mismo nodo varias veces,
    ## guardaremos los nodos visitados en una lista.
    queue = util.Queue()
    ## Cada elemento de la pila sera una tupla (nodo, camino).
    queue.push((problem.getStartState(),[]))
    ## En el set visited guardaremos los nodos que hayamos visitado
    ## en forma de tupla.
    visited = set()
    visited.add(problem.getStartState())
    ## Mientras la cola no este vacia, haremos pop para sacar el
    ## primer elemento y evaluarlo.
    while not queue.isEmpty():
        node,path = queue.pop()
        ## Si el nodo es el destino ya hemos terminado. Devolvemos el
        ## camino que hemos seguido hasta llegar a dicho nodo.
        if(problem.isGoalState(node)):
            return path
        ## Para cada nodo hijo, si no ha sido ya visitado, lo metemos
        ## en la cola junto a el camino que hemos hecho hasta llegar
        ## a dicho nodo.
        for coord, direction, steps in problem.getSuccessors(node):
            if not coord in visited:
                queue.push((coord,path+[direction]))
                visited.add(coord)
    ## No deberia de llegar nunca aqui.
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    ## Para el Astar utilizaremos una cola de prioridades. El nodo que se expandira
    ## al hacer pop sera el que tenga el menor valor asociado.
    ## La heuristica que utilizaremos para determinar estos valores viene dada por
    ## la llamada a la funcion.
    ## Para evitar visitar el mismo nodo varias veces, iremos guardando los nodos
    ## que visitamos en el array visited.
    queue = util.PriorityQueue()
    ## Nodo inicial.
    startNode = problem.getStartState()
    ## Los elementos de la cola son de la forma ((nodo,camino),valor)) donde nodo
    ## es un nodo del laberinto, camino es el camino realizado para llegar a dicho
    ## nodo y valor es el valor de la heuristica utilizada aplicada a el nodo en
    ## el problema.
    queue.push((startNode,[]),heuristic(startNode, problem))
    ## En el set visited guardaremos los nodos que hayamos visitado
    ## en forma de tupla.
    visited = set()
    ## Mientras que la cola no este vacia, expandimos el nodo de menor valor heuristico.
    while not queue.isEmpty():
        node, path = queue.pop()
        ## Si el nodo expandido es el objectivo devolvemos el camino realizado para llegar.
        if problem.isGoalState(node):
            return path
        ## Si el nodo no ha sido visitado, lo marcamos como visitado.
        if not node in visited:
            visited.add(node)
            ## Para cada hijo del nodo, calculamos el valor heuristico y lo pusheamos a la cola.
            for coord, direction, value in problem.getSuccessors(node):
                if not coord in visited:
                    ## El valor heuristico viene dado por el coste de realizar las acciones
                    ## desde el principio hasta llegar al nodo actual, mas el valor de la
                    ## funcion heuristica aplicada a este nodo.
                    score = problem.getCostOfActions(path+[direction])+heuristic(coord, problem)
                    queue.push((coord,path+[direction]),score)
    ## No deberia llegar nunca aqui.
    return []

            
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
