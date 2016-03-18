import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, actions=(None, 'forward', 'left', 'right'),
                 exploration_rate=0.3, learning_rate=0.9, future_reward_discount=0.25/8, DEFAULT_Q=1.0,
                 #exploration_rate=0.3, learning_rate=0.5, future_reward_discount=0.1, DEFAULT_Q=1.0, # Success: 92.0 Mistakes: 2.44
                 #exploration_rate=0.3, learning_rate=0.9, future_reward_discount=0.25/8, DEFAULT_Q=1.0, # Success: 100.0 Mistakes: 2.68
                 #exploration_rate=0.1, learning_rate=0.4*2, future_reward_discount=0.25/4, DEFAULT_Q=1.0, # Success: 100.0 Mistakes: 2.76
                 injected_q=None,
                 mode="LEARN"):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here

        #Here we will store what we have learned
        self.q = {} if not injected_q else injected_q
        self.DEFAULT_Q = DEFAULT_Q

        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.future_reward_discount = future_reward_discount
        self.actions = actions
        self.mode = mode

        self.arrivals_on_time = 0
        self.mistakes_this_run = 0
        self.run_score = 0
        self.mistakes_per_run = []
        self.run_scores = []

    def reset(self, destination=None):
        self.planner.route_to(destination)

        self.mistakes_per_run.append(self.mistakes_this_run)
        self.run_scores.append(self.run_score)

        self.mistakes_this_run = 0
        self.run_score = 0

    def statistics(self):
        no_runs = float(len(self.mistakes_per_run))
        av_mistakes = reduce(lambda x, y: x + y, self.mistakes_per_run) / no_runs
        success_rate = self.arrivals_on_time/no_runs*100
        av_run_scores = reduce(lambda x, y: x + y, self.run_scores) / no_runs
        params = "Params: lr={} fr={} er={}".format(self.learning_rate, self.future_reward_discount, self.exploration_rate)
        stats = "{}|{}|{}|{}|{}".format(self.mode[:4], params, success_rate, av_mistakes, av_run_scores)
        return stats

    def learn_q(self, state, action, reward, value):
        old_v = self.q.get((state, action), None)
        if old_v:
            self.q[(state, action)] = old_v + self.learning_rate * (value - old_v)
        else:
            self.q[(state, action)] = reward

    def learn(self, state1, action1, reward, state2):
        """
        We cannot just overwrite the old Q value
        :param state1:
        :param action1:
        :param reward:
        :param state2:
        :return:
        """
        # print("looking for: {}".format([(state2, a) for a in self.actions]))
        max_q_new = max([self.q.get((state2, a), self.DEFAULT_Q) for a in self.actions])
        # print("Max found: {}".format(max_q_new))
        self.learn_q(state1, action1, reward, reward + self.future_reward_discount*max_q_new)
        #self.learn_q(state1, action1, reward, reward + max_q_new)

    def choose_a(self, state):

        if random.random() < self.exploration_rate and self.mode == "LEARN":
            action = random.choice(self.actions)
        else:
            q = [self.q.get((state, a), self.DEFAULT_Q) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.actions[i]
        return action

    def current_state(self):
        """
        A function to gather all the values we ween for the sate and turn them into something hashable.
        We need the state to be hashable as it will be a key on our dict
        :return:
        """
        next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)  # input sample: {'light': 'red', 'oncoming': None, 'right': None, 'left': None}
        deadline = self.env.get_deadline(self)
        # radar = "{}-{}-{}".format(inputs["left"], inputs["oncoming"], inputs["right"])
        hashable_state = (
            ("wp", next_waypoint),
            ("light", inputs["light"]),
            # ("red-light", inputs["light"] == "red"),
            # ("radar", radar),
            ("oncoming", inputs["oncoming"]),
            ("right", inputs["right"]),
            ("left", inputs["left"]),
            ("deadline", deadline),
        )
        # hashable_state = tuple(sorted(state_dict.items()))

        return hashable_state

    def update(self, t):
        # Gather inputs

        inputs = self.env.sense(self)
        next_waypoint = self.planner.next_waypoint()
        self.next_waypoint = next_waypoint
        # input sample: {'light': 'red', 'oncoming': None, 'right': None, 'left': None}
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        old_state = self.state
        self.state = self.current_state()
        
        # TODO: Select action according to your policy
        #action = random.choice((None, 'forward', 'left', 'right'))
        action = self.choose_a(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        self.state = self.current_state()
        new_state = self.state
        """
        # Testing reward modification
        if deadline == 1 and reward < 10:
            # We did not make it, very bad reward
            reward = -5
        """

        if reward >= 10:
            self.arrivals_on_time += 1

        if reward < 0:
            self.mistakes_this_run += 1

        self.run_score += reward

        if self.mode == "LEARN":
            self.learn(state1= old_state, action1=action, reward=reward, state2=new_state)

        #print("State1: {}, Action:{}, State2:{}, Reward:{}".format(old_state, action, new_state, reward))


def run():
    """Run the agent for a finite number of trials."""

    # Parameter optimization
    """
    params_grid = {
        'exploration_rate': [.0, .1, .2, .3, .4, .5, .6, .7, .8, .9],
        'learning_rate': [.1, .2, .3, .4, .5, .6, .7, .8, .9],
        'future_reward_discount': [.0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
    }

    import numpy
    params_grid = {
        'exploration_rate': numpy.arange(0, 1, 0.05),
        'learning_rate': numpy.arange(0, 1, 0.05),
        'future_reward_discount': numpy.arange(0, 1, 0.05)
    }

    tests = []
    for value in params_grid["exploration_rate"]:
        for value2 in params_grid["learning_rate"]:
            for value3 in params_grid["future_reward_discount"]:
                tests.append({"exploration_rate": value, "learning_rate": value2, "future_reward_discount": value3})

    import cStringIO
    out = cStringIO.StringIO()
    tests_len = len(tests)
    for i, test in enumerate(tests):
        print("Trail {} of {}".format(i, tests_len))
        # Set up environment and agent
        e = Environment()  # create environment (also adds some dummy traffic)
        a = e.create_agent(LearningAgent)  # create agent
        a.learning_rate = test["learning_rate"]
        a.future_reward_discount = test["future_reward_discount"]
        a.exploration_rate = test["exploration_rate"]
        e.set_primary_agent(a, enforce_deadline=True)  # set agent to track
        # Now simulate it
        sim = Simulator(e, update_delay=0)  # reduce update_delay to speed up simulation
        sim.run(n_trials=100, render=False)  # press Esc or close pygame window to quit
        #print a.q
        out.write(a.statistics()+"\n")


        test_e = Environment()  # create environment (also adds some dummy traffic)
        # Create a new agent based on the policy we determined with Q-Learning
        trained_agent = test_e.create_agent(LearningAgent, injected_q=a.q, mode="TEST")  # create agent
        # Only set values for easier printing, this agent will not learn
        trained_agent.learning_rate = test["learning_rate"]
        trained_agent.future_reward_discount = test["future_reward_discount"]
        trained_agent.exploration_rate = test["exploration_rate"]
        test_e.set_primary_agent(trained_agent, enforce_deadline=True)  # set agent to track

        # Now simulate it
        sim = Simulator(test_e, update_delay=0)  # reduce update_delay to speed up simulation
        sim.run(n_trials=50, render=False)  # press Esc or close pygame window to quit

        out.write(trained_agent.statistics()+"\n")

    with open ('results.txt', 'w') as fd:
        fd.write(out.getvalue())
    out.close()
    """

    print("FINAL RESULT:")
    # Final optimized result
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    a.learning_rate = 0.35
    a.future_reward_discount = 0.3
    a.exploration_rate = 0.7
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track
    # Now simulate it
    sim = Simulator(e, update_delay=0)  # reduce update_delay to speed up simulation
    try:
        sim.run(n_trials=100, render=False)  # press Esc or close pygame window to quit
    except:
        sim.run(n_trials=100)
    #print a.q
    print a.statistics()

    test_e = Environment()  # create environment (also adds some dummy traffic)
    # Create a new agent based on the policy we determined with Q-Learning
    trained_agent = test_e.create_agent(LearningAgent, injected_q=a.q, mode="TEST")  # create agent
    # Only set values for easier printing, this agent will not learn
    trained_agent.learning_rate = a.learning_rate
    trained_agent.future_reward_discount = a.future_reward_discount
    trained_agent.exploration_rate = a.exploration_rate
    test_e.set_primary_agent(trained_agent, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(test_e, update_delay=0)  # reduce update_delay to speed up simulation
    try:
        sim.run(n_trials=25, render=True)  # press Esc or close pygame window to quit
    except:
        sim.run(n_trials=25)
    print trained_agent.statistics()


if __name__ == '__main__':
    run()
