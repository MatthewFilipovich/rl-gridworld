""" Module defines the GridWorld class, where the agent can move in either
4 or 8 directions and wind can be enabled.
"""
import numpy as np
import random


class GridWorld:
    """Grid World simulation."""

    def __init__(self, stochastic_wind=False):
        self.stochastic_wind = stochastic_wind
        self.wind = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
        self.state = np.array([0, 3])

    def step(self, move):
        """Move the agent in Grid World.

        Attributes:
            move(ndarray): direction agent should move in (x, y).

        Returns tuple: (state, reward, done)
        """
        wind = self._apply_wind()
        if self._check_move(self.state + move):
            self.state += move 
        self.state += wind
        if self.state[1] < 0:  # Wind moved agent below grid
            self.state[1] = 0
        if self.state[1] > 6:  # Wind moved agent above grid
            self.state[1] = 6
        if self._game_over():
            return (self.state, 1, True)  # Reward of 1
        return (self.state, -1, False)  # Game not over: reward -1

    def reset(self):
        """Resets state to original position and returns state"""
        self.state = np.array([0, 3])
        return self.state

    def _check_move(self, state):
        x, y = state
        if x < 0 or x > 9 or y < 0 or y > 6:
            return False
        return True

    def _apply_wind(self):
        wind = self.wind[self.state[0]]
        if self.stochastic_wind:
            wind = random.choice((wind - 1, wind, wind + 1))
        return np.array([0, wind])

    def _game_over(self):
        x, y = self.state
        if x == 7 and y == 3:
            return True
        return False
