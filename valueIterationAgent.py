from pacai.agents.learning.value import ValueEstimationAgent
from pacai.util import counter

class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = counter.Counter()   # Holds the values of the mdp
        self.policy = counter.Counter()   # Holds the policy of the mdp

        for num_iter in range(iters):
            computed_values = counter.Counter()   # counter to calculate the mdp state q-vals
            states = self.mdp.getStates()
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                maxvals = list()
                for action in actions:
                    maxvals.append(self.getQValue(state, action))
                if len(maxvals) != 0:
                    computed_values[state] = max(maxvals)
            self.values = computed_values

    def getQValue(self, state, action):
        q = 0
        for x in self.mdp.getTransitionStatesAndProbs(state, action):
            next_state, probability = x
            reward = self.mdp.getReward(state, action, next_state)   # q-val equation
            discounted_v = self.discountRate * self.getValue(next_state)
            q += probability * (reward + discounted_v)
        return q

    def getPolicy(self, state):
        maxval = float('-inf')
        bestAction = None
        for action in self.mdp.getPossibleActions(state):
            val = self.getQValue(state, action)
            if val > maxval:   # only updates if the q-val for action is better than previous
                bestAction = action
                maxval = val
        return bestAction

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        return self.getPolicy(state)
