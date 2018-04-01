import unittest
from envs.tictactoe_env import TicTacToeEnv
import numpy as np

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.env = TicTacToeEnv()

    def test_reset(self):
        state = self.env.reset()
        self.assertTrue(np.array_equal(np.array([0,  0,  0,  0,  0,  0,  0,  0, 0]), state))

    def test_simple_game(self):
        e_reward, e_done = 0.0, False

        self.env.reset()
        state, reward, done = self.env.step(0)

        self.assertTrue(np.array_equal(1, state[0]))
        self.assertTrue(reward == self.env.default)
        self.assertEqual(e_done, done)

    def test_illegal_move(self):
        e_reward, e_done = 0.0, True

        self.env.reset()
        state, reward, done = self.env.step(8)

        opponent = np.argmin(state)

        state, reward, done = self.env.step(opponent)

        self.assertTrue(np.array_equal(1, state[8]))
        self.assertTrue(reward == self.env.illegal)
        self.assertEqual(e_done, done)

    def test_draw_or_win_or_loose(self):
        e_reward, e_done = 0.0, True

        self.env.reset()
        next_move = 8
        while(True):
          state, reward, done = self.env.step(next_move)
          if not done:
            next_move = np.where(state == 0)[0][0]
          else:
              break

        self.assertTrue(np.array_equal(1, state[8]))
        self.assertTrue(reward in [self.env.draw,self.env.win, self.env.lose])
        self.assertEqual(e_done, done)


if __name__ == '__main__':
    unittest.main()