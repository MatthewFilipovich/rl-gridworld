import unittest
import numpy as np
from gridworld import GridWorld


class TestStringMethods(unittest.TestCase):

    def test_move_up(self):
        env = GridWorld()
        state = env.reset()
        state, _, _ = env.step(np.array([0, 1]))
        self.assertTrue(np.array_equal(state,np.array([0, 4])))

    def test_move_left(self):
        env = GridWorld()
        state = env.reset()
        state, _, _ = env.step(np.array([-1, 0]))
        self.assertTrue(np.array_equal(state,np.array([0, 3]))) 

    def test_norm_wind(self):
        env = GridWorld()
        state = env.reset()
        for _ in range(4):
            state, _, _ = env.step(np.array([1, 0]))
        self.assertTrue(np.array_equal(state,np.array([4, 4])))           
        for _ in range(2):
            state, _, _ = env.step(np.array([1, 0]))
        self.assertTrue(np.array_equal(state,np.array([6, 6])))       
        state, _, _ = env.step(np.array([0, -1]))
        self.assertTrue(np.array_equal(state,np.array([6, 6])))          
        for _ in range(5):
            state, _, _ = env.step(np.array([1, 0]))
        self.assertTrue(np.array_equal(state,np.array([9, 6])))      
        for _ in range(4):        
            state, _, _ = env.step(np.array([0, -1]))
        for _ in range(2):
            state, _, done = env.step(np.array([-1, 0]))
        self.assertTrue(done)

    

if __name__ == '__main__':
    unittest.main()