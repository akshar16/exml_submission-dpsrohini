import numpy as np

import random

boardtypes = [
    [
        "############################",
        "#............##............#",
        "#.####.#####.##.#####.####.#",
        "#o####.#####.##.#####.####o#",
        "#.####.#####.##.#####.####.#",
        "#..........................#",
        "#.####.##.########.##.####.#",
        "#.####.##.########.##.####.#",
        "#......##....##....##......#",
        "######.##### ## #####.######",
        "######.##### ## #####.######",
        "######.##          ##.######",
        "######.## ###  ### ##.######",
        "######.## #  b   # ##.######",
        " p     ##   c  d   ##       ",
        "######.## #   a  # ##.######",
        "######.## ###  ### ##.######",
        "######.##          ##.######",
        "######.## ######## ##.######",
        "######.## ######## ##.######",
        "#............##............#",
        "#.####.#####.##.#####.####.#",
        "#.####.#####.##.#####.####.#",
        "#o..##................##..o#",
        "###.##.##.########.##.##.###",
        "###.##.##.########.##.##.###",
        "#......##....##....##......#",
        "#.##########.##.##########.#",
        "#.##########.##.##########.#",
        "#..........................#",
        "############################"]
]


class Player:
    
    def __init__(self,player_pos):
        self.points = 0
        self.x = player_pos[0]
        self.y = player_pos[1]
        self.player_pos = [self.x, self.y]

    def __getitem__(self, index):
        return self.player_pos[index]
    
    def move(self,board,move):
       
        if move == "up":
            if board[self.x-1,self.y] in "abcd":
                   return 'death'
            
            if board[self.x-1,self.y] != '#':
                if board[self.x-1,self.y] == '.':
                    self.points += 1
                board[self.x,self.y] = ' '
                self.x -= 1
                board[self.x,self.y] = 'p'
                if board[self.x,self.y] == '.':
                    self.points += 1
               
        if move == "down":
            
            if board[self.x+1,self.y] in "abcd":
                    return 'death'
            elif board[self.x+1,self.y] != '#':
                if board[self.x+1,self.y] == '.':
                    self.points += 1
                board[self.x,self.y] = ' '
                self.x += 1
                board[self.x,self.y] = 'p'
                
                
        if move == "left":
            if board[self.x,self.y-1] in "abcd":
                   return 'death'
            elif board[self.x,self.y-1] != '#':
                if board[self.x,self.y-1] == '.':
                    self.points += 1
                board[self.x,self.y] = ' '
                self.y -= 1
                board[self.x,self.y] = 'p'
               
                
        if move == "right":
            if board[self.x,self.y+1] in "abcd":
                     return 'death'
            elif board[self.x,self.y+1] != '#':
                if board[self.x,self.y+1] == '.':
                    self.points += 1 
                board[self.x,self.y] = ' '
                self.y += 1
                board[self.x,self.y] = 'p'
                
                

class Ghost:
    
    def __init__(self,ghost_pos,id):
        self.x = ghost_pos[0]
        self.y = ghost_pos[1]
        self.id = id
        self.position = [self.x,self.y]
        self.last_block = ' '


   

    def move(self,board,move):
        
        if move == "up":
            if board[self.x-1,self.y] == 'p':
                return 'death'
            elif board[self.x-1,self.y] not in "abcd#":
                
                board[self.x,self.y] = self.last_block
                self.last_block = board[self.x-1,self.y]
                self.x -= 1
                board[self.x,self.y] = self.id
           
        if move == "down":
            if board[self.x+1,self.y] == 'p':
                return 'death'
            elif board[self.x+1,self.y] not in "abcd#":
                
              
                board[self.x,self.y] = self.last_block
                self.last_block = board[self.x+1,self.y]
                self.x += 1
                board[self.x,self.y] = self.id
                
            
        if move == "left":
            if board[self.x,self.y-1] == 'p':
               return 'death'
            elif board[self.x,self.y-1] not in "abcd#":
                board[self.x,self.y] = self.last_block
                self.last_block = board[self.x,self.y-1]
                self.y -= 1
                board[self.x,self.y] = self.id
           
        if move == "right":
            if board[self.x,self.y+1] == 'p':
                return 'death'
            elif board[self.x,self.y+1] not in "abcd#":
                
                board[self.x,self.y] = self.last_block
                self.last_block = board[self.x,self.y+1]
                self.y += 1
                board[self.x,self.y] = self.id
              
           

        
class Board:
    def __init__(self):
        global rows,cols
        rand = random.randint(0,boardtypes.__len__()-1)
        arr = boardtypes[rand]
        self.rand = rand
        np_board = np.array([list(row) for row in arr])
        self.board = np_board
        self.row, self.col = self.board.shape
        self.positions = self.get_positions()
        rows,cols = self.row,self.col

    def food_left(self):
        return np.count_nonzero(self.board == '.')
       
    def display(self):
      return self.board

    def __getitem__(self, index):
        return self.board[index]
    
    def __setitem__(self,index,value):
        
        self.board[index] = value

    def get_positions(self):
        
        for i in range(self.row):
            for j in range(self.col):
                if self.board[i,j] == 'p':
                    player_pos = [i,j]
                if self.board[i,j] == 'a':
                    ghost1_pos = [i,j]
                if self.board[i,j] == 'b':
                    ghost2_pos = [i,j]
                if self.board[i,j] == 'c':
                    ghost3_pos = [i,j]
                if self.board[i,j] == 'd':
                    ghost4_pos = [i,j] 

        return player_pos, ghost1_pos, ghost2_pos, ghost3_pos, ghost4_pos        
                   
    def get_board(self):
        return self.board.tolist() 
                

   