"""
This file contains incomplete versions of some agents that can be selected to control Pacman.
You will complete their implementations.

Good luck and happy searching!
"""

import logging

from pacai.core.actions import Actions
from pacai.core.search import heuristic
from pacai.core.search.position import PositionSearchProblem
from pacai.core.search.problem import SearchProblem
from pacai.agents.base import BaseAgent
from pacai.agents.search.base import SearchAgent
from pacai.core.directions import Directions
from pacai.core.search.search import uniformCostSearch

from pacai.core.distance import manhattan
from pacai.core.distance import maze
from pacai.core.distance import euclidean

class CornersProblem(SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function.
    See the `pacai.core.search.position.PositionSearchProblem` class for an example of
    a working SearchProblem.

    Additional methods to implement:

    `pacai.core.search.problem.SearchProblem.startingState`:
    Returns the start state (in your search space,
    NOT a `pacai.core.gamestate.AbstractGameState`).

    `pacai.core.search.problem.SearchProblem.isGoal`:
    Returns whether this search state is a goal state of the problem.

    `pacai.core.search.problem.SearchProblem.successorStates`:
    Returns successor states, the actions they require, and a cost of 1.
    The following code snippet may prove useful:
    ```
        successors = []

        for action in Directions.CARDINAL:
            x, y = currentPosition
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]

            if (not hitsWall):
                # Construct the successor.

        return successors
    ```
    """
    def __init__(self, startingGameState):
        super().__init__()

        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top = self.walls.getHeight() - 2
        right = self.walls.getWidth() - 2

        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                logging.warning('Warning: no food in corner ' + str(corner))

    def startingState(self):
        initial = list()   # list of corners at start
        if self.startingPosition in self.corners:
            initial.append(self.startingPosition)   # if the agent starts on a corner
        return (self.startingPosition, initial)

    def isGoal(self, state):
        loc, visit = state
        if(len(visit) == len(self.corners)):
            return True
        else:
            return False

    def successorStates(self, state):
        """
        Returns successor states, the actions they require, and a constant cost of 1.
        """
        successors = []
        location, corners_visited = state
        cost = 1
        self._numExpanded += 1
        for action in Directions.CARDINAL:
            x, y = location
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]
            if (not hitsWall):
                next_location = (nextx, nexty)
                next_corners_visited = list(corners_visited)
                if next_location in self.corners:    # updates list if next position is a corner
                    if next_location not in next_corners_visited:
                        next_corners_visited.append(next_location)
                successors.append(((next_location, next_corners_visited), action, cost))
        return successors

    def actionsCost(self, actions):
        """
        Returns the cost of a particular sequence of actions.
        If those actions include an illegal move, return 999999.
        This is implemented for you.
        """

        if (actions is None):
            return 999999

        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999

        return len(actions)

def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem;
    i.e. it should be admissible.
    (You need not worry about consistency for this heuristic to receive full credit.)
        walls = problem.walls  # These are the walls of the maze, as a Grid.

    """
    corners = problem.corners
    location, visited = state
    distances = list()    # manhattan distances between position
    separation_distances = list()   # distances between the corners
    numCornersVisited = 1
    for corner in corners:
        if corner not in visited:
            dist = manhattan(location, corner)
            distances.append(dist)
        for others in corners:
            if others != corner:
                separation_distances.append(euclidean(corner, others))
    if len(separation_distances) != 0:
        e_dist = max(separation_distances)
    if len(distances) == 0:
        return heuristic.null(state, problem)   # if goal state is reached
    if len(visited) != 0:
        numCornersVisited = len(visited)
    return max(distances) + e_dist / numCornersVisited   # heuristic equation

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.
    First, try to come up with an admissible heuristic;
    almost all admissible heuristics will be consistent as well.

    If using A* ever finds a solution that is worse than what uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!
    On the other hand, inadmissible or inconsistent heuristics may find optimal solutions,
    so be careful.

    The state is a tuple (pacmanPosition, foodGrid) where foodGrid is a
    `pacai.core.grid.Grid` of either True or False.
    You can call `foodGrid.asList()` to get a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the problem.
    For example, `problem.walls` gives you a Grid of where the walls are.

    If you want to *store* information to be reused in other calls to the heuristic,
    there is a dictionary called problem.heuristicInfo that you can use.
    For example, if you only want to count the walls once and store that value, try:
    ```
    problem.heuristicInfo['wallCount'] = problem.walls.count()
    ```
    Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount'].
    """
    start = problem.startingGameState
    position, foodGrid = state
    foodList = foodGrid.asList()
    distances = list()
    separations = list()   # manhattan separations between nodes
    euclid_seperations = list()  # euclidean separations between nodes
    add_dist = 0   # shortest manhattan distance
    edist = 0   # shortest euclidean distance

    if(len(foodList) == 0):
        return heuristic.null(state, problem)   # returns 0 if goal is reached
    for i in range(len(foodList)):
        dist1 = maze(position, foodList[i], start)
        for j in range(len(foodList)):
            if foodList[i] != foodList[j]:
                dist = manhattan(position, foodList[j])
                if dist1 < dist:    # replaces if the manhattan is a lower bound
                    dist1 = dist
                separations.append(manhattan(foodList[i], foodList[j]))
                euclid_seperations.append(euclidean(foodList[i], foodList[j]))
        distances.append(dist1)
    if(len(separations) != 0):
        add_dist = min(separations)
    if(len(euclid_seperations) != 0):
        edist = min(euclid_seperations)
    return distances[int(len(foodList) / 2)] + add_dist - edist   # heuristic equation

class ClosestDotSearchAgent(SearchAgent):
    """
    Search for all food using a sequence of searches.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def registerInitialState(self, state):
        self._actions = []
        self._actionIndex = 0

        currentState = state

        while (currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState)  # The missing piece
            self._actions += nextPathSegment

            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' %
                            (str(action), str(currentState)))

                currentState = currentState.generateSuccessor(0, action)

        logging.info('Path found with cost %d.' % len(self._actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from gameState.
        """
        problem = AnyFoodSearchProblem(gameState)
        return uniformCostSearch(problem)

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem,
    but has a different goal test, which you need to fill in below.
    The state space and successor function do not need to be changed.

    The class definition above, `AnyFoodSearchProblem(PositionSearchProblem)`,
    inherits the methods of `pacai.core.search.position.PositionSearchProblem`.

    You can use this search problem to help you fill in
    the `ClosestDotSearchAgent.findPathToClosestDot` method.

    Additional methods to implement:

    `pacai.core.search.position.PositionSearchProblem.isGoal`:
    The state is Pacman's position.
    Fill this in with a goal test that will complete the problem definition.
    """
    def __init__(self, gameState, start = None):
        super().__init__(gameState, goal = None, start = start)

        # Store the food for later reference.
        self.food = gameState.getFood()

    def isGoal(self, state):
        x_loc, y_loc = state   # returns T or F if the location is a goal
        return self.food[x_loc][y_loc]

class ApproximateSearchAgent(BaseAgent):
    """
    Implement your contest entry here.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Get a `pacai.bin.pacman.PacmanGameState`
    and return a `pacai.core.directions.Directions`.

    `pacai.agents.base.BaseAgent.registerInitialState`:
    This method is called before any moves are made.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)
