""" Module defines the GridWorld class, where the agent can move in either
4 or 8 directions and wind can be enabled.
"""
import numpy as np
import random


class GridWorld:
    """Grid World simulation."""

    def __init__(self, initial_state=(0, 3), goal_state=(7, 3), stochastic_wind=False):
        self.stochastic_wind = stochastic_wind
        self.wind = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
        self._initial_state = initial_state
        self.state = np.array(initial_state)

        # information about the board
        self.width = len(self.wind)
        self.height = 7
        self.goal_state = goal_state
        self.num_actions = 4
        self.moves = np.array([[x, y] for x, y in zip([1, 0, -1, 0],
                                                      [0, 1, 0, -1])])
        if stochastic_wind:
            self._num_actions = 8
            self.moves = np.array([[x, y] for x, y in zip([1, 1, 0, -1, -1, -1, 0, 1],
                                                          [0, 1, 1, 1, 0, -1, -1, -1])])  # kings moves

    def step(self, move):
        """Move the agent in Grid World.

        Attributes:
            move(int): index of move to take.

        Returns tuple: (state, reward, done)
        """
        wind = self._apply_wind()  # get wind from original state
        if self._check_move(self.state + self.moves[move]):  # would move leave the grid
            self.state += self.moves[move]
        if self._check_move(self.state + wind):  # would wind push player off the grid
            self.state += wind
        if self._game_over():
            return self.state, 1, True  # Reward of 1
        return self.state, -1, False  # Game not over: reward -1

    def reset(self):
        """Resets state to original position and returns state"""
        self.state = np.array(self._initial_state)
        return self.state

    def _check_move(self, state):
        x, y = state
        if x < 0 or x > self.width-1 or y < 0 or y > self.height-1:
            return False
        return True

    def _apply_wind(self):
        wind = self.wind[self.state[0]]
        if self.stochastic_wind:
            wind = random.choice((wind - 1, wind, wind + 1))
        return np.array([0, wind])

    def _game_over(self):
        x, y = self.state
        return x == self.goal_state[0] and y == self.goal_state[1]
