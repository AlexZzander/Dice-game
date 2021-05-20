import cProfile
import time
from abc import ABC, abstractmethod
import numpy as np
from dice_game import DiceGame
import matplotlib.pyplot as plt


class DiceGameAgent(ABC):
    def __init__(self, game):
        self.game = game

    @abstractmethod
    def play(self, state):
        pass


class AlwaysHoldAgent(DiceGameAgent):
    def play(self, state):
        return 0, 1, 2


class PerfectionistAgent(DiceGameAgent):
    def play(self, state):
        if state == (1, 1, 1) or state == (1, 1, 6):
            return 0, 1, 2
        else:
            return ()

class MyAgent(DiceGameAgent):
    def __init__(self, game, gamma=0.99):
        self.discount_factor = gamma
        # this calls the superclass constructor (does self.game = game)
        super().__init__(game)
        # a dictionary with best policy: state -> action
        # initialized once and used through the games
        self.policy = self.get_optimal_policy()

    def get_optimal_policy(self):
        """
        Calculates optimal policy by value iteration method
        :return:
              policy:
                a list containing optimal policy
                key - state
                value - action
        """
        # Value iteration
        utilities = {key: 0 for key in self.game.states}
        theta = 0.01
        while True:
            delta = 0
            for state in self.game.states:
                temp = utilities[state]
                max_utility = 0
                for action in self.game.actions:
                    st, game_over, reward, probabilities = self.game.get_next_states(action, state)
                    if game_over:
                        max_utility = max(reward, max_utility)
                        continue
                    utility = 0
                    for st_new, prob in zip(st, probabilities):
                        utility += (prob * (reward + self.discount_factor*utilities[st_new]))
                    max_utility = max(utility, max_utility)
                utilities[state] = max_utility
                delta = max(delta, abs(temp - utilities[state]))
            # termination condition
            if delta < theta:
                break

        # Finding optimal policy
        max_values = {}
        policy = {}
        for state in self.game.states:
            for a in self.game.actions:
                sum_utility = 0
                new_states, _, reward, prob = self.game.get_next_states(a, state)
                for p, st_new in zip(prob, new_states):
                    if st_new is None:
                        sum_utility += reward
                    else:
                        sum_utility += (p * (reward + self.discount_factor*utilities[st_new]))
                max_values[a] = sum_utility
            # finding an action(key) with maximum utility and assign it to a value
            # This function max(max_values, key=max_values.get) works
            # as an argmax function
            policy[state] = max(max_values, key=max_values.get)
        return policy

    def play(self, state):
        """
        given a state, return the chosen action for this state
        by finding best policies for every action
        """
        return self.policy[state]


def play_game_with_agent(agent, game, verbose=False):
    state = game.reset()

    if (verbose): print(f"Testing agent: \n\t{type(agent).__name__}")
    if (verbose): print(f"Starting dice: \n\t{state}\n")

    game_over = False
    actions = 0
    while not game_over:
        action = agent.play(state)
        actions += 1

        if (verbose): print(f"Action {actions}: \t{action}")
        _, state, game_over = game.roll(action)
        if (verbose and not game_over): print(f"Dice: \t\t{state}")

    if (verbose): print(f"\nFinal dice: {state}, score: {game.score}")

    return game.score, actions


def run_tests(n=1, gamma=0.9):
    import time

    total_score = 0
    total_time = 0

    np.random.seed()

    # print("Testing basic rules.")
    # print()

    game = DiceGame()

    start_time = time.process_time()
    test_agent = MyAgent(game, gamma)
    total_time += time.process_time() - start_time

    total_actions = 0
    for i in range(n):
        start_time = time.process_time()
        score, actions = play_game_with_agent(test_agent, game)
        total_time += time.process_time() - start_time
        total_actions += actions
        # print(f"Game {i} score: {score}")
        total_score += score

    print(f"Average amount of actions:{total_actions / n}")
    print(f"Average score: {total_score / n}")
    print(f"Total time: {total_time:.4f} seconds")
    return total_score / n


def run_extended_tests():
    total_score = 0
    total_time = 0
    n = 10

    print("Testing extended rules – two three-sided dice.")
    print()

    game = DiceGame(dice=2, sides=3)

    start_time = time.process_time()
    test_agent = MyAgent(game)
    total_time += time.process_time() - start_time

    for i in range(n):
        start_time = time.process_time()
        score = play_game_with_agent(test_agent, game)
        total_time += time.process_time() - start_time

        print(f"Game {i} score: {score}")
        total_score += score

    print()
    print(f"Average score: {total_score / n}")
    print(f"Average time: {total_time / n:.5f} seconds")


if __name__ == "__main__":
    # scores = []
    # gammas = []
    # for gamma in np.arange(0.6, 0.999, 0.03):
    #     score = run_tests(100, gamma)
    #     scores.append(score)
    #     gammas.append(gamma)
    #     print(f"gamma: {gamma}")
    # fig, ax = plt.subplots()  # Create a figure containing a single axes.
    # ax.plot(gammas, scores)
    # fig.show()# Add a legend.

    run_tests(1000, 0.99)
    #  cProfile.run('run_tests(1000)')
    # run_extended_tests()
    # cProfile.run('run_games()')
