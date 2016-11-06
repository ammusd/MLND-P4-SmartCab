import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import defaultdict
from operator import itemgetter

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.gamma = 0.5
        self.epsilon = 0.05
        self.alpha_power = 0.8
        self.pos_rewards = 0
        self.neg_rewards = 0

        def build_starting_q_values():
            start_q_values = {}
            for action in self.env.valid_actions:
                start_q_values[action] = 0
            return start_q_values
        self.q_values = defaultdict(build_starting_q_values)    

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.pos_rewards = 0
        self.neg_rewards = 0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.get_state_tuple(self.next_waypoint, inputs)
        
        # TODO: Select action according to your policy
        action = self.get_nxt_action()
        # action = random.choice([None, 'forward','left','right'])

        # Execute action and get reward
        reward = self.env.act(self, action)
        if reward < 0:
            self.neg_rewards += 1

        # TODO: Learn policy based on state, action, reward
        self.update_Q_values(t, action, reward)
        self.pos_rewards += reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
		
        if self.planner.next_waypoint() == None:
            print "Destination reached in {} timesteps. Positive reward: {}, Negative reward: {}".format(t+1, self.pos_rewards, self.neg_rewards)
        elif deadline == 0:
            print "Failed to arrive at desination in {} timesteps. Positive reward: {}, Negative reward: {}".format(t+1, self.pos_rewards, self.neg_rewards)

    def get_state_tuple(self, waypoint, inputs):
        return (waypoint, inputs['light'], inputs['oncoming'], inputs['left'])

    def get_nxt_action(self):
        best_action = self.get_best_action(self.state)[0]

        ## Introducing random actions with probability 'epsilon' to take the agent away from possible local minima
        if random.random() < self.epsilon:
            random_actions = [action for action in self.env.valid_actions if action != best_action]
            return random.choice(random_actions)

        return best_action

    def get_best_action(self, state):
        all_actions = self.q_values[state]
        return max(all_actions.iteritems(), key=itemgetter(1))

    def update_Q_values(self, t, action, reward):
        alpha = 1.0/(t + 1) ** self.alpha_power
        
        nxt_waypoint = self.planner.next_waypoint()
        nxt_inputs = self.env.sense(self)
        nxt_state = self.get_state_tuple(nxt_waypoint, nxt_inputs)
        nxt_best_action = self.get_best_action(nxt_state)
        nxt_q_value = reward + self.gamma * nxt_best_action[1]

        present_q_value = self.q_values[self.state][action]        
        self.q_values[self.state][action] = (1 - alpha) * present_q_value + alpha * nxt_q_value        

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track #####added code
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.00001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

if __name__ == '__main__':
    run()
