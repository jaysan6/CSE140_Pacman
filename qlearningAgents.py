from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection
from pacai.util import counter
from pacai.util.probability import flipCoin
import random


class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:
    Get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION: I created a counter Dict to hold the q-values. The dict is indexed by tuple
    keys (state, action) and hold the respective q-values of the mdp. For update, I followed
    the equation from the slides to get the learned q-value based on samples. For getAction,
    I used the exploration/exploitation concept through the implementation of flipCoin to decide
    which action to take. getValue and getPolicy are similar implementations to value iteration,
    where the max q-val or the action with the max-qval is chosen as the Value and Policy for the
    state.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

        self.qVals = counter.Counter()  # counter dict indexed by (state, action) pairs

    def update(self, state, action, nextState, reward):
        qval = self.getQValue(state, action)

        next_legal_actions = self.getLegalActions(nextState)
        qvals = list()
        for action2 in next_legal_actions:
            qvals.append(self.getQValue(nextState, action2))
        if len(next_legal_actions) != 0:   # if successor states exist
            best_qval_sample = max(qvals) * self.getDiscountRate()
        else:
            best_qval_sample = 0
        sample = reward + best_qval_sample   # sample part of the q learning

        self.qVals[(state, action)] = ((1 - self.getAlpha()) * qval) + (self.getAlpha() * sample)

    def getAction(self, state):
        legal_actions = self.getLegalActions(state)
        if len(legal_actions) == 0:
            return None
        if flipCoin(self.getEpsilon()):   # exploration/exploitation
            return random.choice(legal_actions)
        else:
            return self.getPolicy(state)

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """
        if (state, action) not in self.qVals:
            return 0.0
        else:
            return self.qVals[(state, action)]

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """
        legal_actions = self.getLegalActions(state)
        if len(legal_actions) == 0:    # terminal state
            return 0.0
        qvals = list()
        for action in legal_actions:
            qvals.append(self.getQValue(state, action))
        return max(qvals)

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """
        legal_actions = self.getLegalActions(state)
        if len(legal_actions) == 0:   # terminal state
            return None
        bestAction = None
        bestqval = float('-inf')
        for action in legal_actions:
            q = self.getQValue(state, action)
            if q > bestqval:    # update action with the best q-val
                bestAction = action
                bestqval = q
        return bestAction

class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action

class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION: Similar to q-learning, I created a counter dict of weights
    to hold the weights at each respective feature. Instead of being indexed
    by (state, action) pairs, I indexed them based on the feature. For the update
    function, I followed the same equation as the instructions page and updated
    each weight of the respective features. get qValue is simply a translation
    of the equation given.
    """

    def __init__(self, index,
            extractor = 'pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)

        # You might want to initialize weights here.
        self.weights = counter.Counter()

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            # You might want to print your weights here for debugging.
            # *** Your Code Here ***
            # for weight in self.weights:
            # w_i = self.weights[weight]
            # print(weight, w_i)
            pass

    def update(self, state, action, nextState, reward):
        qval = self.getQValue(state, action)
        legal_actions = self.getLegalActions(nextState)
        if len(legal_actions) == 0:   # if there is no legal action, value must be 0
            value = 0
        else:
            value = self.getDiscountRate() * self.getValue(nextState)

        correction = reward + value - qval
        feats = self.featExtractor.getFeatures(self.featExtractor, state, action)
        for feature in feats:
            update = self.getAlpha() * correction * feats[feature]
            self.weights[feature] += update

    def getQValue(self, state, action):
        qval = 0.0
        feats = self.featExtractor.getFeatures(self.featExtractor, state, action)
        for feature in feats:
            qval += self.weights[feature] * feats[feature]  # sum of the weights * feature vector
        return qval
