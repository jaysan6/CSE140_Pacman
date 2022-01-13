import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core.distance import manhattan
from pacai.core.distance import euclidean
from pacai.core.directions import Directions


class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)
        self.prev_action = []   # list of actions the reflex agent takes

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.
        self.prev_action.append(legalMoves[chosenIndex])   # appends chosen action
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPosition = successorGameState.getPacmanPosition()
        nextFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        foodDistances = list()
        score = 0

        if len(self.prev_action) > 3:    # Checks for repeated actions to minimize repetition
            if action == self.prev_action[len(self.prev_action) - 3]:
                score -= 100000
        if len(nextFood) != 0:
            for food in range(len(nextFood)):
                dist = euclidean(nextFood[food], newPosition)
                foodDistances.append(-1 * (dist + food**2))   # negative values are preferred
            score = min(foodDistances)
        for ghosts in range(len(newGhostStates)):
            ghost_pos = successorGameState.getGhostPosition(ghosts + 1)
            if manhattan(newPosition, ghost_pos) <= 2:   # if a ghost is close enough to kill pacman
                score -= 10000
        return score

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """
    def __init__(self, index, **kwargs):
        super().__init__(index)
        for key in kwargs:    # correctly updates depth argument
            if key == 'depth':
                self._treeDepth = int(kwargs[key])

    def getAction(self, state):
        def minimax(state, depth, agent):
            if state.isLose() or state.isWin() or depth == 0:  # terminal state
                return self.getEvaluationFunction()(state)
            if agent == 0:   # pacman max agent
                max_val = float('-inf')
                actions = state.getLegalActions()
                actions.remove(Directions.STOP)
                for a in actions:
                    successor = state.generateSuccessor(agent, a)
                    for ghosts in range(state.getNumAgents() - 1):  # run on each ghost on map
                        ev = minimax(successor, depth - 1, ghosts + 1)
                        max_val = max(max_val, ev)
                return max_val
            else:  # min agent or ghost
                min_val = float('inf')
                actions = state.getLegalActions(agent)
                for a in actions:
                    successor = state.generateSuccessor(agent, a)
                    ev = minimax(successor, depth, 0)
                    min_val = min(min_val, ev)
                return min_val
        actions = state.getLegalActions()
        actions.remove(Directions.STOP)
        depth = self.getTreeDepth()
        best_op = None
        best_val = float('-inf')
        for action in actions:
            successor = state.generateSuccessor(0, action)
            old_val = best_val
            best_val = max(best_val, minimax(successor, depth, 0))
            if old_val != best_val:  # choose action with best minimax value
                best_op = action
        return best_op

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """
    def __init__(self, index, **kwargs):
        super().__init__(index)
        for key in kwargs:
            if key == 'depth':
                self._treeDepth = int(kwargs[key])

    def getAction(self, state):
        def max_val(state, alpha, beta, depth):  # pac man alpha beta maximizer
            if state.isLose() or state.isWin() or depth == 0:   # terminal state
                return self.getEvaluationFunction()(state)
            v = float('-inf')
            actions = state.getLegalActions()
            actions.remove(Directions.STOP)
            for a in actions:
                successor = state.generateSuccessor(0, a)
                for ghosts in range(state.getNumAgents() - 1):
                    ev = min_val(successor, alpha, beta, depth - 1, ghosts + 1)
                    v = max(v, ev)
                    if v >= beta:  # prune if not necessary to explore
                        return v
                    alpha = max(alpha, v)
            return v

        def min_val(state, alpha, beta, depth, agent):  # ghost alpha beta minimizer
            if state.isLose() or state.isWin() or depth == 0:  # terminal state
                return self.getEvaluationFunction()(state)
            v = float('inf')
            actions = state.getLegalActions(agent)
            for a in actions:
                successor = state.generateSuccessor(agent, a)
                ev = max_val(successor, alpha, beta, depth)
                v = min(v, ev)
                if v <= alpha:  # prune if not necessary to explore
                    return v
                beta = min(beta, v)
            return v
        actions = state.getLegalActions()
        actions.remove(Directions.STOP)
        depth = self.getTreeDepth()
        best_op = None
        best_op = None
        best_val = float('-inf')

        for action in actions:
            successor = state.generateSuccessor(0, action)
            old_val = best_val
            best_val = max(best_val, max_val(successor, float('-inf'), float('inf'), depth))
            if old_val != best_val:  # choose action with best value
                best_op = action
        return best_op


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)  # Q4 constructor
        for key in kwargs:
            if key == 'depth':
                self._treeDepth = int(kwargs[key])  # initialize necessary arguments
            if key == 'evalFn':
                super().__init__(index, kwargs[key])  # Q5 constructor
                return

    def getAction(self, state):
        def expectimax(state, depth, agent):
            if state.isLose() or state.isWin() or depth == 0:  # terminal state
                return self.getEvaluationFunction()(state)
            if agent == 0:  # pac man expectimax agent
                actions = state.getLegalActions()
                actions.remove(Directions.STOP)
                v = float('-inf')
                for a in actions:
                    s = state.generateSuccessor(0, a)
                    for g in range(state.getNumAgents() - 1):
                        ev = expectimax(s, depth - 1, g + 1)
                        v = max(v, ev)
                return v
            else:   # chance nodes
                actions = state.getLegalActions(agent)
                v = 0
                p = 1.0 / len(actions)   # uniform probability distribuition
                for a in actions:
                    s = state.generateSuccessor(agent, a)
                    v += p * expectimax(s, depth, 0)
                return v

        agents = state.getNumAgents()
        actions = state.getLegalActions()
        actions.remove(Directions.STOP)
        depth = self.getTreeDepth()
        best_op = None
        best_val = float('-inf')

        for action in actions:
            successor = state.generateSuccessor(0, action)
            old = best_val
            best_val = max(best_val, expectimax(successor, depth, agents - 1))
            if best_val != old:  # pick action with the best value
                best_op = action
        return best_op

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: For my evaluation function, I took the manhattan distances to the food and
    outputted a score to the nearest one, similar to my reflex agent. The score was also
    multiplied by an extra factor so that pacman would me more incentivized to prioritize
    food and winning. Then, I calculated the manhattan distances to the nearest capsules,
    which can be useful in scaring the ghosts. I put them as lower priority but if pacman
    reaches a capsule, then he should prioritize food over all else. Finally, I checked
    the ghosts states and increased the score if a capsule is reached, or if the ghosts
    are scared, and constantly penalized pacman as long as it lives due to the fact that
    ghosts exists. If a ghost is not scared but they are too close, then there should be
    a large punishment so that pacman evades the ghost.
    """
    legalMoves = currentGameState.getLegalActions()

    if len(legalMoves) == 0:
        return 0

    newPosition = currentGameState.getPacmanPosition()
    nextFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    foodDistances = list()
    powerMultiplier = 40.0   # multiplier for capsule distance
    powerUp = currentGameState.getCapsules()
    score = 25 * currentGameState.getScore()
    if len(nextFood) != 0:
        for food in range(len(nextFood)):
            dist = 1.0 / manhattan(nextFood[food], newPosition)
            foodDistances.append(dist)
        score += max(foodDistances) + 100 * len(nextFood)
    powerUpDistances = list()  # list of manhattan distances to capsules
    for powers in powerUp:
        dist = euclidean(newPosition, powers)
        powerUpDistances.append(powerMultiplier / dist)
    if len(powerUpDistances) != 0:
        score += max(powerUpDistances)
    for ghosts in newGhostStates:
        ghost_pos = ghosts.getPosition()
        dist_ghost = manhattan(newPosition, ghost_pos)
        if ghosts.getScaredTimer() > 0:    # kill ghost if they are scared
            score += 10000
        else:
            score -= 10000
            if dist_ghost <= 3:  # stay away from ghost if close enough
                score -= 10000
    return score * 5

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)
