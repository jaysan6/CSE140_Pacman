"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue
from pacai.core.directions import Directions

def stringToDirection(direction):
    if direction == "North":
        return Directions.NORTH
    elif direction == "South":
        return Directions.SOUTH
    elif direction == "East":
        return Directions.EAST
    elif direction == "West":
        return Directions.WEST
    else:
        return Directions.STOP


def depthFirstSearch(problem):

    visited = [problem.startingState()]
    path = list()

    states = Stack()
    visits = Stack()   # holds the path
    states.push(problem.startingState())
    visits.push([])

    while not states.isEmpty():
        state = states.pop()
        direction = visits.pop()
        if problem.isGoal(state):
            path = direction
            break
        for element in problem.successorStates(state):
            if element[0] not in visited:   # pushes onto stack if its a new node
                visited.append(element[0])
                states.push(element[0])
                visits.push(direction + [element[1]])
    return path

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """
    visited = [problem.startingState()]
    path = list()

    states = Queue()
    visits = Queue()   # holds the path
    states.push(problem.startingState())
    visits.push([])

    while not states.isEmpty():
        state = states.pop()
        nextpath = visits.pop()
        if problem.isGoal(state):
            path = nextpath
            break

        for element in problem.successorStates(state):
            if element[0] not in visited:   # pushes onto queue if its a new node
                visited.append(element[0])
                states.push(element[0])
                visits.push(nextpath + [element[1]])
    return path


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """
    visited = [problem.startingState()]
    path = list()

    states = PriorityQueue()
    states.push((problem.startingState(), []), 0)   # empty lists with cost = 0

    while not states.isEmpty():
        state, direction = states.pop()

        if problem.isGoal(state):
            path = direction
            break

        for element in problem.successorStates(state):
            if not element[0] in visited:
                visited.append(element[0])
                newpath = direction + [element[1]]   # adds the direction to the successor path
                cost = problem.actionsCost(newpath)
                states.push((element[0], newpath), cost)
    return path

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    visited = [problem.startingState()]
    path = list()

    states = PriorityQueue()
    states.push((problem.startingState(), []), 0)   # empty lists with cost = 0

    while not states.isEmpty():
        state, direction = states.pop()

        if problem.isGoal(state):
            path = direction
            break

        for element in problem.successorStates(state):
            if not element[0] in visited:
                visited.append(element[0])
                newpath = direction + [element[1]]   # adds the direction to the successor path
                combined = problem.actionsCost(newpath) + heuristic(element[0], problem)
                states.push((element[0], newpath), combined)
    return path
