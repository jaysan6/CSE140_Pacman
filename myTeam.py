import random
import math
from pacai.agents.capture.offense import OffensiveReflexAgent
from pacai.agents.capture.defense import DefensiveReflexAgent
from pacai.util import counter
# from pacai.core import distance
# from pacai.core.distance import manhattan
from pacai.core.directions import Directions

"""
PROJECT 4: PACMAN CTF myTeam.py
Includes the agents we used for the tournament
@ authors: Arka Pal, Sanjay Shrikanth, Dhatchi Govindarajan
"""

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.agents.capture.dummy.DummyAgent',
        second = 'pacai.agents.capture.dummy.DummyAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = OffensiveAgent
    secondAgent = DefenseAgent

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]

class OffensiveAgent(OffensiveReflexAgent):
    """
    A OffensiveReflex Agent that overrides most of the parent class'
    methods and adds more functionalities and computations that make the
    agent aggressive and able to react to the opponent's actions. Implements
    a head-first strategy and goes for as much food as it can get.
    """

    def __init__(self, index, timeForComputing = 0.1):
        super().__init__(index)

    def registerInitialState(self, gameState):
        self.nextEntrance = None    # The next entrance to go through when agent dies
        return super().registerInitialState(gameState)

    def getEntrances(self, gameState):
        """
        This function goes through the game map layout and calculates
        the locations of the entrances, or the tunnels that allow agents
        to go from the home side to the opponent side
        """
        width = gameState.data.layout.width
        legalPositions = [x for x in gameState.getWalls().asList(False)]
        left_half = [p for p in legalPositions if p[0] == width / 2 - 1]
        right_half = [p for p in legalPositions if p[0] == width / 2]
        redEntrances = list()
        blueEntrances = list()
        for x in left_half:
            for y in right_half:
                if x[0] + 1 == y[0] and x[1] == y[1]:  # tunnels that pass through the middle
                    redEntrances.append(x)
                    blueEntrances.append(y)
        if self.blue:   # return the relevant set of entrances to consider
            return blueEntrances
        else:
            return redEntrances

    def getAction(self, gameState):
        return super().getAction(gameState)     # gets optimal action

    def chooseAction(self, gameState):
        """
        Evaluates the game state of all possible actions and chooses the action
        that has the lowest score, or dot product of the features and weights
        """
        actions = gameState.getLegalActions(self.index)    # get actions except STOP
        values = [self.evaluate(gameState, a) for a in actions]
        minValue = min(values)   # optimal value
        bestActions = [a for a, v in zip(actions, values) if v == minValue]
        return random.choice(bestActions)   # break ties if more than one optimal action

    def getFeatures(self, gameState, action):
        """
        Evaluates the current game state and the action considered and calculates
        the values for the features. Our features are dependent on distances to
        the nearest enemies, food, capsules, and the entrances to effectively
        implement our strategies.
        """
        features = counter.Counter()
        successor = gameState.generateSuccessor(self.index, action)
        nextPos = successor.getAgentPosition(self.index)

        if action != Directions.STOP:    # the offense agent should not stop
            features['stop'] = 0
        else:
            features['stop'] = 1     # a truth value to activate the corresponding weight

        # check if opponents are nearby
        enemies = self.getOpponents(gameState)
        # if there are enemies
        if len(enemies) > 0:
            closest = math.inf
            closest_opponent = None
            distToEnemy = math.inf
            for op in enemies:
                dist = self.getMazeDistance(gameState.getAgentPosition(op), nextPos)
                if dist < closest:
                    closest = dist
                    distToEnemy = dist
                    closest_opponent = op
            # check if opponent's defense is scared
            if gameState.getAgentState(closest_opponent).getScaredTimer() == 0:
                # since lower scores are preferred, negate it to run away from the closest enemy
                distToEnemy = -1 * distToEnemy
        else:
            distToEnemy = 0

        # The necessary data of the game state
        foodList = self.getFood(gameState).asList()
        capsuleList = self.getCapsules(gameState)
        distToFood = [self.getMazeDistance(nextPos, food) for food in foodList]
        distToCapsules = [self.getMazeDistance(cap, nextPos) for cap in capsuleList]

        # evaluates the game states of the opponents
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]

        if len(distToFood) > 0:
            foodDist = min(distToFood)
        else:
            foodDist = 0

        if len(distToCapsules) > 0:
            capDist = min(distToCapsules)
        else:
            capDist = 0

        features['attack'] = distToEnemy
        features['eat'] = 1.125 * foodDist   # extra weights adjusted using trial and error
        features['powerUp'] = 1.05 * capDist
        return features

    def evaluate(self, gameState, action):
        """
        Multiplies the features and weights together
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getWeights(self, gameState, action):
        """
        Designates the necessary weights to the features. Low scores are preferred
        so higher weights are given to less preferred actions
        """
        successor = self.getSuccessor(gameState, action)
        agentPos = successor.getAgentState(self.index).getPosition()
        OPS = self.getOpponents(gameState)
        if len(OPS) > 0:
            # find the distance to the closest opponent (in the successor state)
            minOp = min([self.getMazeDistance(agentPos, gameState.getAgentPosition(op))
            for op in OPS])
            if minOp > 3:
                score = 0.9   # lower the score multiplier if we are safe
            else:
                score = 140   # do NOT take this action if an enemy is too close
        else:
            score = 0.4

        return {
            'attack': score,
            'eat': 1,
            'powerUp': 1,
            'stop': 40,
            'runToNextEntrance': -20
        }

class DefenseAgent(DefensiveReflexAgent):
    """
    This class represents our Defensive Agent. It implements a tracking strategy
    that guards high concentrations of food and reacts when any opponent tries
    to cross into our territory. The defensive agent chases any opponent that
    invades.
    """
    def __init__(self, index, **kwargs):
        super().__init__(index)

    def isAgentAttacking(self, state, agentIndex):
        """
        This function checks if an agent from the enemy team has reached
        our side. Also checks based on if we are on team red or blue and
        returns True or False given the agentIndex.
        """
        isBlueSide = state.isOnBlueSide(state.getAgentPosition(agentIndex))
        isBlueTeam = state.isOnBlueTeam(agentIndex)
        isRedSide = state.isOnRedSide(state.getAgentPosition(agentIndex))
        isRedTeam = state.isOnRedTeam(agentIndex)
        return (isBlueSide and isBlueTeam) or (isRedSide and isRedTeam)

    def getAction(self, gameState):
        return super().getAction(gameState)

    def chooseAction(self, gameState):
        """
        Similar to the offense agent, evaluates the score of all possible
        actions and picks the one with the smallest value
        """
        actions = gameState.getLegalActions(self.index)
        values = [self.getScore(gameState, a) for a in actions]
        minVal = min(values)
        # action containing lowest score is optimal
        bestAction = [a for a, v in zip(actions, values) if v == minVal]
        return random.choice(bestAction)

    def getScore(self, gameState, action):
        """
        Evaluates the game states and computes the features of the agent
        based on factors like distance to enemies. Only considers invaders
        as they are the only agents that the defense should worry about.
        Our features include the distance to the most vulnerable food
        and the distance to the closest enemy.
        """
        features = counter.Counter()
        if action == Directions.STOP:
            #  We want our defense to be dynamic, so do not choose STOP
            return math.inf

        successor = gameState.generateSuccessor(self.index, action)
        nextPos = successor.getAgentPosition(self.index)

        if not self.isAgentAttacking(successor, self.index):
            return math.inf    # Don't do anything if there are no invaders

        invaders = self.getOpponents(successor)  # holds all threats
        enemies = [invader for invader in invaders     # all threats that are NOT attacking
                  if not self.isAgentAttacking(gameState, invader)
                   ]

        # calculates the distance to the cloest threat that is not on our side
        closestEnemy = None
        closestOpDist = math.inf
        for invader in invaders:
            if not self.isAgentAttacking(gameState, invader):
                dist = self.getMazeDistance(nextPos, gameState.getAgentPosition(invader))
                if dist < closestOpDist:
                    closestOpDist = dist
                    closestEnemy = invader
        if not enemies:   # if all agents are attacking
            enemies = invaders
        # if there are no opponents that are attacking, find the closest enemy
        if closestEnemy is None:
            distances = []
            for ee in enemies:
                distances.append(self.getMazeDistance(gameState.getAgentPosition(ee), nextPos))
            closestDist = min(distances)
            closestEnemy = [a for a, d in zip(enemies, distances) if d == closestDist][0]
        d_list = []
        for e in enemies:
            d_list.append(self.getMazeDistance(gameState.getAgentPosition(e), nextPos))
        min_distance = min(d_list)   # closest distance to an enemy on our side

        if gameState.getAgentState(self.index).getScaredTimer != 0:   # kill any agent that is scare
            min_distance *= -1   # motivate pacman to eat monster when scared

        if len(self.getFoodYouAreDefending(gameState).asList()) == 0:
            final_dist = 0
        else:     # if there is food left to defend
            food = self.getFoodYouAreDefending(gameState).asList()
            foodDistances = []
            for f in food:   # find the minimum distance to the cloest food to the cloest enemy
                food_dist = self.getMazeDistance(gameState.getAgentPosition(closestEnemy), f)
                foodDistances.append(food_dist)
            # based on the closest enemy, predict which food it will eat
            closestFoodToEnemy = min(foodDistances)
            closestFood = [fo for fo, ce in zip(food, foodDistances) if ce == closestFoodToEnemy][0]
            # the distance to the vulnerable food
            final_dist = self.getMazeDistance(nextPos, closestFood)

        features['distToClosestEnemy'] = 1.75 * final_dist   # assign the features
        features['distToVulnerableFood'] = min_distance

        return self.evaluate(features, self.getWeights(gameState, action))   # calculate the scores

    def getWeights(self, gameState, action):
        return {
            'distToClosestEnemy': 1,
            'distToVulnerableFood': 0.3,     # vulnerable food matters more
        }

    def evaluate(self, features, weights):
        return features * weights    # returns dot product of weight and features
