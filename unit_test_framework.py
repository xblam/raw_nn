import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from game import SnakeGameAI
from agent import Agent, train



class test_snake_ai(unittest.TestCase):
    def test_snake_game(self):
        self.assertEqual(1,1)
        

if __name__ == '__main__':
    unittest.main()