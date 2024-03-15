# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        print(successorGameState)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Computing a score for a given state and action pair
        # Pacman's position, remaining food, ghost states, and scared times for ghosts
        # position of the ghost with index 1 (the first ghost) in the successor game state
        first_ghost = successorGameState.getGhostPosition(1)
        # Computing the Manhattan distance between Pacman's new position and the ghost's position
        dist_to_ghost = util.manhattanDistance(first_ghost, newPos)
        # getting the current score from the successor game state
        current_score = successorGameState.getScore()
        # the list of remaining food dots in the successor game state
        remaining_food = newFood.asList()
        # the list of healths in the current game state
        power = currentGameState.getCapsules()
        # adding score based on proxy to food
        shortestDistance  = 1000
        for dots in remaining_food:
            distance = util.manhattanDistance(dots, newPos)
            if distance < shortestDistance:
                shortestDistance = distance
        current_score += max(dist_to_ghost, 3)
        if len(remaining_food) < len(currentGameState.getFood().asList()):
            current_score += 50
        current_score += 50 / shortestDistance
        # score based on capturing powers
        if newPos in power:
            current_score += 100
        # stopping pacman from moving costs points
        if action == Directions.STOP:
            current_score -= 5
        return current_score
        

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, 0, self.depth)[1]
    
    def min_val(self, gameState, agentIndex, depth):
        future_moves = []
        for move in gameState.getLegalActions(agentIndex):
            future_moves.append((self.minimax(gameState.generateSuccessor(agentIndex, move), agentIndex + 1, depth)[0], move))    
        return min(future_moves)
    

    def max_val(self, gameState, agentIndex, depth):
        future_moves = []
        for move in gameState.getLegalActions(agentIndex):
            future_moves.append((self.minimax(gameState.generateSuccessor(agentIndex, move), agentIndex + 1, depth)[0], move))   
        return max(future_moves)
        
    # reccursively going down the tree to find the max/min value
    def minimax(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return ( self.evaluationFunction(gameState), "Stop")
        
        number_of_agents = gameState.getNumAgents()
        agentIndex = agentIndex % number_of_agents
 
        if agentIndex == number_of_agents - 1:
            depth -= 1
        #pac-man
        if agentIndex == 0:
            return self.max_val(gameState, agentIndex, depth)
        else:
            #ghosts
            return self.min_val(gameState, agentIndex, depth)




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, 0, self.depth)[1]
        util.raiseNotDefined()

        
    def min_val(self, gameState, agentIndex, depth, a, b):
        future_moves = []
        for move in gameState.getLegalActions(agentIndex):
            node = self.minimax(gameState.generateSuccessor(agentIndex, move), agentIndex + 1, depth, a, b)[0]
            future_moves.append((node, move))
            # checking for min value
            if node < a :
                return (node, move)
            # updating beta value
            b = min(b, node)
        return min(future_moves)

    def max_value(self, gameState, agentIndex, depth, a, b):
        future_moves = []
        for move in gameState.getLegalActions(agentIndex):
            node = self.minimax(gameState.generateSuccessor(agentIndex, move), agentIndex + 1, depth, a, b)[0]
            future_moves.append((node, move))
            if node > b:
                return (node, move)
            a = max(a, node)
        return max(future_moves)
 
    def minimax(self, gameState, agentIndex, depth, a = -10000000, b = 10000000):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return ( self.evaluationFunction(gameState), "Stop")
        
        agentsNum = gameState.getNumAgents()
        agentIndex %=  agentsNum
        if agentIndex == agentsNum - 1:
            depth -= 1

        if agentIndex == 0:
            return self.max_value(gameState, agentIndex, depth, a, b)
        else:
            return self.min_val(gameState, agentIndex, depth, a, b)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.Expectimax(gameState, 0, self.depth)[1]
    
    def min_val(self, gameState, agentIndex, depth):
        future_moves = []
        factor = 0
        for move in gameState.getLegalActions(agentIndex):
            vertex = self.Expectimax(gameState.generateSuccessor(agentIndex, move), agentIndex + 1, depth)[0]
            factor += vertex
            future_moves.append((vertex, move))
        
        return (factor / len(future_moves), )

    def max_val(self, gameState, agentIndex, depth):
        future_moves = []
        for move in gameState.getLegalActions(agentIndex):
            future_moves.append((self.Expectimax(gameState.generateSuccessor(agentIndex, move), agentIndex + 1, depth)[0], move))   
        return max(future_moves)

    
    def Expectimax(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return ( self.evaluationFunction(gameState), "Stop")
        
        number_of_agents = gameState.getNumAgents()
        agentIndex %=  number_of_agents
        if agentIndex == number_of_agents - 1:
            depth -= 1

        if agentIndex == 0:
            return self.max_val(gameState, agentIndex, depth)
        else:
            return self.min_val(gameState, agentIndex, depth)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Ghost Scared vs. Not Scared:
    If ghost_track > 0 (ghosts are scared), calculates a score based on the distance from the ghost.
    If not scared, calculates a negative score based on the distance.
    Food Proximity:
    Loops through all remaining food pellets and calculates the minimum distance to any food from Pac-Man's current position.
    Computes a food score based on the remaining food and its proximity to Pac-Man.
    Capsule Proximity:
    Loops through all power powers and calculates the minimum distance to any capsule from Pac-Man's current position.
    Computes a score based on the remaining powers and their proximity to Pac-Man.
    """
    "*** YOUR CODE HERE ***"
    power = currentGameState.getCapsules()
    # Tracks the remaining time during which ghosts are scared due to Pac-Man consuming a power.
    ghost_track = currentGameState.getGhostStates()[0].scaredTimer
    curr_spot = currentGameState.getPacmanPosition()
    food_available = currentGameState.getFood().asList()
    ghost = currentGameState.getGhostPosition(1)
    
    dist_to_ghost = manhattanDistance(ghost, curr_spot)
  
    
    
    if ghost_track > 0:
        score = max(75 - dist_to_ghost, 55) 
    else:
        score = -max(75 - dist_to_ghost, 55)

    food_track = 80
    for food in food_available:
        food_track = min(manhattanDistance(curr_spot, food), food_track)
    food_points = 500 - len(food_available) * 10 -  food_track

    distance = 80
    for i in power:
        distance = min(manhattanDistance(curr_spot, i), distance)
    power_points =  100 - len(power) * 80 - distance
    score = currentGameState.getScore() + food_points + score + power_points

    return score

# Abbreviation
better = betterEvaluationFunction
