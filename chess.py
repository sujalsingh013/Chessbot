import copy
import numpy as np
class Chess:

    def __init__(self):
        self.state = [[None],[None]]
        self.board = self.initialize_board()
        self.current_player = 'white'
        self.en_passant = None
        self.castling = [False, False, False, False]
        self.kingdidmove = [False, False]
        self.rookdidmove = [False, False, False, False]
        self.material=[[0,0],[0,0]]
        self.fiftymove=0
    def initialize_board(self):
        board = [
            [5, 2, 3, 9, 4, 3, 2, 5],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-5, -2, -3, -9, -4, -3, -2, -5]
        ]
        self.state[0].append(self.board)
        
        return board
    def copy(self):
        new_chess = Chess()
        new_chess.state = copy.deepcopy(self.state)
        new_chess.board = copy.deepcopy(self.board)
        new_chess.current_player = self.current_player
        new_chess.en_passant = self.en_passant
        new_chess.castling = self.castling.copy()
        new_chess.kingdidmove = self.kingdidmove.copy()
        new_chess.rookdidmove = self.rookdidmove.copy()
        new_chess.material = copy.deepcopy(self.material)
        new_chess.fiftymove=self.fiftymove
        return new_chess
    def is_inside_board(self,i,j):
        return i>=0 and i<8 and j>=0 and j<8
    def is_check(self,board,player):
        king = 4 if player == 'white' else -4
        king_position = None
        for i in range(8):
            for j in range(8):
                if board[i][j] == king:
                    king_position = i, j
                    break
            if king_position:
                break
        i,j=king_position    
        # checking for rooks and queens
        rook_directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for di, dj in rook_directions:
            ni, nj = i + di, j + dj
            while self.is_inside_board(ni, nj):
                if board[ni][nj] != 0:
                    if board[ni][nj] * king < 0:
                        piece = board[ni][nj]
                        if abs(piece) == 5 or abs(piece) == 9:
                            return True
                    break
                ni, nj = ni + di, nj + dj

        # checking for bishops and queens
        bishop_directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for di, dj in bishop_directions:
            ni, nj = i + di, j + dj
            while self.is_inside_board(ni, nj):
                if board[ni][nj] != 0:
                    if board[ni][nj] * king < 0:
                        piece = board[ni][nj]
                        if abs(piece) == 3 or abs(piece) == 9:
                            return True
                    break 
                ni, nj = ni + di, nj + dj
        # checking for knight
        for di, dj in [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                        (1, -2), (1, 2), (2, -1), (2, 1)]:
            ni, nj = i + di, j + dj
            if self.is_inside_board(ni, nj) and board[ni][nj] * board[i][j] < 0:
                piece = board[ni][nj]
                if abs(piece) == 2:
                    return True
        # checking for pawn
        pawn_directions = [(1, -1), (1, 1)] if player == 'white' else [(-1, -1), (-1, 1)]
        for di, dj in pawn_directions:
            ni, nj = i + di, j + dj
            if self.is_inside_board(ni, nj) and board[ni][nj] * king < 0:
                piece = board[ni][nj]
                if abs(piece) == 1:
                    return True
        #checking for king
        king_directions = [(1, 0), (-1, 0), (0, 1), (0, -1),(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for di, dj in king_directions:
            ni, nj = i + di, j + dj
            if self.is_inside_board(ni, nj) and board[ni][nj] * board[i][j] < 0:
                piece = board[ni][nj]
                if abs(piece) == 4:
                    return True
        return False
    def visualisation_board(self):
        temp_board = self.copy()
        return temp_board
    def illegal_move(self,start,end):
        temp_board = self.visualisation_board()
        temp_board.make_move((temp_board.board[start[0]][start[1]], start, end))
        temp_board.current_player='black' if temp_board.current_player=='white' else 'white'
        a = temp_board.is_check(temp_board.board,temp_board.current_player)
        return a
    def legal_moves_pawn(self, i, j):
        moves = []
        pawn = 1 if self.current_player == 'white' else -1
        directions = [[1,0,0],[1,1,1],[1,-1,1]]
        if (i == 1 and pawn == 1) or (i == 6 and pawn == -1) and self.board[i+directions[0][0]][j+directions[0][1]]==0:
            directions.append([2,0,0])
        for direction in directions:
            if direction[2]==1:
                if self.is_inside_board(i+pawn*direction[0],j+direction[1]) and self.board[i+pawn*direction[0]][j+direction[1]] * pawn < 0 and not self.illegal_move((i,j),(i + pawn*direction[0], j + direction[1])):
                    if not i+pawn*direction[0]==0 and not i+pawn*direction[0]==7:    
                        # Pawn capture
                        moves.append((pawn, (i, j), (i + pawn*direction[0], j + direction[1])))
                    else:
                        # Pawn capture promotion
                        for piece_type in [5, 9, 3, 2]:
                            moves.append((piece_type * pawn, (i, j), (i + pawn*direction[0], j + direction[1])))
            elif direction[2]==0:
                if self.is_inside_board(i+pawn*direction[0],j+direction[1]) and self.board[i+pawn*direction[0]][j+direction[1]]==0 and not self.illegal_move((i,j),(i + pawn*direction[0], j + direction[1])):
                    if not i+pawn*direction[0]==0 and not i+pawn*direction[0]==7:
                        # Pawn move
                        moves.append((pawn, (i, j), (i + pawn*direction[0], j + direction[1])))
                    else:
                        # Pawn promotion
                        for piece_type in [5, 9, 3, 2]:
                            moves.append((piece_type * pawn, (i, j), (i + pawn*direction[0], j + direction[1])))
        #en passant
        if self.en_passant is not None:
            en_passant_row, en_passant_col = self.en_passant
            if i == en_passant_row and abs(j - en_passant_col) == 1:
                if self.board[i][j]*self.board[en_passant_row][en_passant_col]<0 and not self.illegal_move((i,j),(i + pawn, en_passant_col)):
                    moves.append((pawn, (i, j), (i + pawn, en_passant_col)))
        return moves

    def legal_moves_rook(self, i, j):
        rook=5 if self.current_player=='white' else -5
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for di, dj in directions:
            ni, nj = i + di, j + dj
            while self.is_inside_board(ni, nj):
                if self.board[ni][nj] * self.board[i][j] <= 0:
                    if not self.illegal_move((i,j),(ni,nj)):
                        moves.append((rook,(i, j), (ni, nj)))
                    if self.board[ni][nj] != 0:
                        break
                else:
                    break
                ni, nj = ni + di, nj + dj
        return moves

    def legal_moves_knight(self, i, j):
        knight = 2 if self.current_player == 'white' else -2
        moves = []
        offsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                (1, -2), (1, 2), (2, -1), (2, 1)]
        for di, dj in offsets:
            ni, nj = i + di, j + dj
            if self.is_inside_board(ni, nj) and self.board[ni][nj] * self.board[i][j] <= 0 and not self.illegal_move((i,j),(ni,nj)):
                moves.append((knight, (i, j), (ni, nj)))  # Corrected this line
        return moves

    def legal_moves_bishop(self, i, j):
        bishop = 3 if self.current_player == 'white' else -3
        moves = []
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for di, dj in directions:
            ni, nj = i + di, j + dj
            while self.is_inside_board(ni, nj):
                if self.board[ni][nj] * self.board[i][j] <= 0:
                    if not self.illegal_move((i,j),(ni,nj)):
                        moves.append((bishop, (i, j), (ni, nj)))  # Corrected this line
                    if self.board[ni][nj] != 0:
                        break
                else:
                    break
                ni, nj = ni + di, nj + dj
        return moves

    def legal_moves_queen(self, i, j):
        queen = 9 if self.current_player == 'white' else -9
        moves = []
        moves.extend([(queen, (i, j), move[2]) for move in self.legal_moves_rook(i, j)])
        moves.extend([(queen, (i, j), move[2]) for move in self.legal_moves_bishop(i, j)])
        return moves

    def legal_moves_king(self, i, j):
        king = 4 if self.current_player == 'white' else -4  # Corrected this line
        moves = []
        offsets = [(1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for di, dj in offsets:
            ni, nj = i + di, j + dj
            if self.is_inside_board(ni, nj) and self.board[ni][nj] * self.board[i][j] <= 0 and not self.illegal_move((i,j),(ni,nj)):
                moves.append((king, (i, j), (ni, nj)))  # Corrected this line
        if not self.kingdidmove[0 if self.current_player == 'white' else 1]:
            #the function needs to check if the king is in check or passes through check
            if not self.rookdidmove[0 if self.current_player == 'white' else 2]:
                if self.board[i][1] == 0 and self.board[i][2] == 0 and self.board[i][3] == 0:
                    if not self.illegal_move((i,j),(i,2)) and not self.illegal_move((i,j),(i,3)):
                        moves.append((king, (i, j), (i, 2)))
            if not self.rookdidmove[1 if self.current_player == 'white' else 3]:
                if self.board[i][5] == 0 and self.board[i][6] == 0:
                    if not self.illegal_move((i,j),(i,5)) and not self.illegal_move((i,j),(i,6)):
                        moves.append((king, (i, j), (i, 6)))
                
        return moves
    def legal_moves(self):
        legal_moves = []
        flag=-1
        self.material=[[0,0],[0,0]]
        if self.current_player == 'white':
            flag=1
        for i in range(8):
            for j in range(8):
                piece = self.board[i][j]
                if piece<0:
                    self.material[1][0]+=piece
                    self.material[1][1]+=1
                if piece>0:
                    self.material[0][0]+=piece
                    self.material[0][1]+=1
                if piece == flag*1:  # Pawn
                    legal_moves.extend(self.legal_moves_pawn(i, j))
                elif piece == flag*5:  # Rook
                    legal_moves.extend(self.legal_moves_rook(i, j))
                elif piece == flag*2:  # Knight
                    legal_moves.extend(self.legal_moves_knight(i, j))
                elif piece == flag*3:  # Bishop
                    legal_moves.extend(self.legal_moves_bishop(i, j))
                elif piece == flag*9:  # Queen
                    legal_moves.extend(self.legal_moves_queen(i, j))
                elif piece == flag*4:  # King
                    legal_moves.extend(self.legal_moves_king(i, j))
        return legal_moves
    def make_move(self,move):
        piece, start, end =move
        i,j=start
        x,y=end

        self.board[x][y]=piece
        self.board[i][j]=0
        #castling priviledges
        if abs(piece)==4:
            self.kingdidmove[0 if piece>0 else 1]=True
        if abs(piece)==5:
            if i==0 and j==0:
                self.rookdidmove[0]=True
            if i==0 and j==7:
                self.rookdidmove[1]=True
            if i==7 and j==0:
                self.rookdidmove[2]=True
            if i==7 and j==7:
                self.rookdidmove[3]=True
        #en_passant adjustment
        if self.en_passant is not None  and self.en_passant[0]==i and self.en_passant[1]==y and abs(piece)==1 and abs(j-y)==1:
            self.board[self.en_passant[0]][self.en_passant[1]]=0
        #castling move adjustment
        if abs(piece)==4 and abs(j-y)==2:
            if y==2:
                self.board[i][0]=0
                self.board[i][3]=5 if piece>0 else -5
                self.rookdidmove[0 if piece>0 else 2]=True
                self.kingdidmove[0 if piece>0 else 1]=True
                self.castling[0 if piece>0 else 2]=True
            if y==6:
                self.board[i][7]=0
                self.board[i][5]=5 if piece>0 else -5
                self.rookdidmove[1 if piece>0 else 3]=True
                self.kingdidmove[0 if piece>0 else 1]=True
                self.castling[1 if piece>0 else 3]=True
        #en_passant flag
        if abs(piece)==1 and abs(x-i)==2:
            self.en_passant=(i,j)
        else:
            self.en_passant=None
        #50 move rule
        self.fifty_move_rule(move)
        self.current_player='black' if self.current_player=='white' else 'white'
    def game_end(self):
        #checkmate
        if self.legal_moves()==[] and self.is_check(self.board,self.current_player):
            return True
        #stalemate
        if self.legal_moves()==[] and not self.is_check(self.board,self.current_player):
            return True
        #draw by insufficient material
        if ((self.material[0][0]==7 and self.material[0][1]==2) or (self.material[0][0]==6 and self.material[0][1]==2) or (self.material[0][1]==1)) and ((self.material[1][0]==7 and self.material[1][1]==2) or (self.material[1][0]==6 and self.material[1][1]==2) or (self.material[1][1]==1)):
            return True
        #50 move rule
        if self.fiftymove==100:
            return True
        return False
    def fifty_move_rule(self,move):
        piece, start , end =move
        if abs(piece)==1:
            self.fiftymove=0
        elif self.board[end[0]][end[1]]!=0:
            self.fiftymove=0
        else:
            self.fiftymove+=1
    def print_board(self):
        for row in self.board:
            print(" ".join(f"{piece:3}" for piece in row))
    def reward(self,intermediate_state):
        if self.game_end():
            return 1000
        if intermediate_state.game_end():
            return -500
        return 0    
    def reset(self):
        self.__init__()
        return self.board
    def update_state(self,move):
        if self.current_player == 'black':
            self.state[0].append(self)
            piece,start,end=move
            self.state[1].append(piece,start[0],start[1],end[0],end[1])
            self.state[0].pop(0)
            self.state[1].pop(0)
        else:
            #invert board so board is always from white's perspective
            for i in range(4):
                for j in range(8):
                    a=0
                    a=self.board[i][j]
                    self.board[i][j] = -self.board[7-i][7-j]
                    self.board[7-i][7-j] = -1*a
            self.state[0].append(self)            
            piece,start,end=move
            piece=-piece
            start=(7-start[0],7-start[1])
            end=(7-end[0],7-end[1])
            self.state[1].append(piece,start[0],start[1],end[0],end[1])
            self.state[0].pop(0)
            self.state[1].pop(0)


