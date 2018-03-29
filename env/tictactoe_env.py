import numpy as np
import itertools
import random

class TicTacToeEnv:

    def __init__(self, illegal = -10, win= 100, draw= 5, lose = -5 ,default = 0):
        self.size = 3
        self.board = np.zeros((self.size, self.size))
        self.illegal = illegal
        self.win = win
        self.draw = draw
        self.lose = lose
        self.default = default
        self.x = 1
        self.opponent = -1

    def reset(self):
       self.board = np.zeros_like(self.board)
       return self.board.flatten()

    def step(self, action):
        move = (int(action / 3), int(action % 3))
        state, reward, done = self._step(move, self.x)
        if(done):
           return (self.board.flatten(), reward, done)
        move = self._random_player()
        return self._step(move, self.opponent)

    def _step(self, move, current_player):
        if not self._is_available(move):
            return (self.board.flatten(), self.illegal, True)

        self._make_move(move, current_player)

        if(self._is_winner(current_player)):
            if(current_player == self.opponent):
              return (self.board.flatten(), self.lose, True)
            else:
              return (self.board.flatten(), self.win, True)

        all_moves = self._all_available()
        all_moves = list(all_moves)
        if len(all_moves) == 0:
            return (self.board.flatten(), self.draw, True)

        return (self.board.flatten(), self.default, False)

    def _random_player(self):
        all_moves = self._all_available()
        all_moves = list(all_moves)
        return random.choice(all_moves)


    def _is_winner(self, player):

        for i in range(self.size):
          row = self.board[i, :]
          if np.all(row == player):
              return True

        for i in range(self.size):
          col = self.board[:, i]
          if np.all(col == player):
              return True

        diagonal = np.diagonal(self.board)
        diagonalFliped = np.diagonal(np.fliplr(self.board))

        if np.all(diagonal == player) or np.all(diagonalFliped == player):
            return True

        return False

    def _all_available(self):
       for x, y in itertools.product(list(range(3)), list(range(3))):
          if self.board[x][y] == 0:
            yield (x, y)

    def _make_move(self, action, player):
        self.board[action] = player

    def _is_available(self, move):
        row, col = move
        return (self.board[row][col] == 0)









