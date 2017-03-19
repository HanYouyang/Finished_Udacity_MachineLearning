import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class QLearningAgent(Agent):

    def __init__(self, env):
        super(QLearningAgent, self).__init__(env)
        self.color = 'red'
        self.planner = RoutePlanner(self.env, self)
        self.deadline = self.env.get_deadline(self)
        self.next_waypoint = None
        self.moves = 0

        self.qDict = dict()
        self.alpha = 0.6  # The fold of reward of adding value.
        self.epsilon = 0.01  # Chance of randomly move.
        self.gamma = 0.2  # Determine the possible sensitivity of reward of each action.

        self.state = None
        self.new_state = None

        self.reward = None
        self.cum_reward = 0

        self.possible_actions = Environment.valid_actions
        self.action = None

    def reset(self, destination = None):
        self.planner.route_to(destination)
        self.next_waypoint = None
        self.moves = 0

        self.state = None
        self.new_state = None

        self.reward = 0
        self.cum_reward = 0

        self.action = None

    def getQvalue(self, state, action):
        key = (state, action)
        return self.qDict.get(key, 10.0)

    def getMaxQ(self, state):
        q = [self.getQvalue(state, a) for a in self.possible_actions]
        return max(q)

    def get_action(self, state):
        """
        epsilon-greedy approach to choose action given the state
        """
        if random.random() < self.epsilon:
            action = random.choice(self.possible_actions)
        else:
            q = [self.getQvalue(state, a) for a in self.possible_actions]
            if q.count(max(q)) > 1:
                best_actions = [i for i in range(len(self.possible_actions)) if q[i] == max(q)]
                index = random.choice(best_actions)

            else:
                index = q.index(max(q))
            action = self.possible_actions[index]

        return action

    def qlearning(self, state, action, nextState, reward):
        """
        use Qlearning algorithm to update q values
        """
        key = (state, action)
        if (key not in self.qDict):
            # initialize the q values
            self.qDict[key] = 10.0
        else:
            self.qDict[key] = self.qDict[key] + self.alpha * (reward + self.gamma*self.getMaxQ(nextState) - self.qDict[key])

    def update(self, t):
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        self.new_state = inputs
        self.new_state['next_waypoint'] = self.next_waypoint
        self.new_state = tuple(sorted(self.new_state.items()))

        action = self.get_action(self.new_state)
        new_reward = self.env.act(self, action)

        if self.reward != None:
            self.qlearning(self.state, self.action, self.new_state, self.reward)
        self.action = action
        self.state = self.new_state
        self.reward = new_reward
        self.cum_reward = self.cum_reward + new_reward
        self.moves = self.moves + 1
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, new_reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()
    a = e.create_agent(QLearningAgent)  # Create Qlearning agent
    e.set_primary_agent(a, enforce_deadline=True)  # Set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.001)  # Reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # Press Esc or close pygame window to quit

if __name__ == '__main__':
    run()
