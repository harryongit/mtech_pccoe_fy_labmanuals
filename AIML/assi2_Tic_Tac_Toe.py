# Tic-Tac-Toe Game using Minimax Algorithm (AI vs Human)

import math

# Initialize board
board = [' ' for _ in range(9)]

def print_board():
    for i in range(0, 9, 3):
        print(board[i], "|", board[i+1], "|", board[i+2])
        if i < 6:
            print("--+---+--")

def check_winner(player):
    win_positions = [
        (0,1,2),(3,4,5),(6,7,8),
        (0,3,6),(1,4,7),(2,5,8),
        (0,4,8),(2,4,6)
    ]
    return any(board[a] == board[b] == board[c] == player for a,b,c in win_positions)

def is_draw():
    return ' ' not in board

def minimax(is_maximizing):
    if check_winner('O'):
        return 1
    if check_winner('X'):
        return -1
    if is_draw():
        return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'O'
                score = minimax(False)
                board[i] = ' '
                best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'X'
                score = minimax(True)
                board[i] = ' '
                best_score = min(score, best_score)
        return best_score

def ai_move():
    best_score = -math.inf
    move = 0
    for i in range(9):
        if board[i] == ' ':
            board[i] = 'O'
            score = minimax(False)
            board[i] = ' '
            if score > best_score:
                best_score = score
                move = i
    board[move] = 'O'

def human_move():
    while True:
        move = int(input("Enter your move (0-8): "))
        if board[move] == ' ':
            board[move] = 'X'
            break
        else:
            print("Invalid move, try again")

# Game Loop
print("Tic-Tac-Toe: You (X) vs AI (O)")
print_board()

while True:
    human_move()
    print_board()
    if check_winner('X'):
        print("You win!")
        break
    if is_draw():
        print("Draw!")
        break

    ai_move()
    print("\nAI Move:")
    print_board()
    if check_winner('O'):
        print("AI wins!")
        break
    if is_draw():
        print("Draw!")
        break
