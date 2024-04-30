from collections import deque
import random
import numpy as np
from chess import Chess
class Memory(Chess):
    def __init__(self, max_memory):
        super().__init__()
        self.max_memory = max_memory
        self.buffer = deque(maxlen=max_memory)
    def preprocess_input(self,move):
        board=self.visualisation_board()
        board.make_move(move)
        if self.current_player == 'black':
            board.state[0].append(board)
            piece,start,end=move
            board.state[1].append(piece,start[0],start[1],end[0],end[1])
            board.state[0].pop(0)
            board.state[1].pop(0)
        else:
            #invert board so board is always from white's perspective
            for i in range(4):
                for j in range(8):
                    a=0
                    a=board.board[i][j]
                    board.board[i][j] = -board.board[7-i][7-j]
                    board.board[7-i][7-j] = -1*a
            board.state[0].append(board)            
            piece,start,end=move
            piece=-piece
            start=(7-start[0],7-start[1])
            end=(7-end[0],7-end[1])
            board.state[1].append([piece,start[0],start[1],end[0],end[1]])
            board.state[0].pop(0)
            board.state[1].pop(0)
        return board.state
    def push(self, state, reward, next_state):
        experience = [state, reward, next_state]
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        reward_batch = []
        next_state_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, reward, next_state = experience
            state_batch.append(state)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
        
        return state_batch, reward_batch, next_state_batch
    def __len__(self):
        return len(self.buffer)