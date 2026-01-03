# 8-Puzzle Problem using A* Search Algorithm

import heapq

GOAL_STATE = (
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 0)
)

MOVES = [(-1,0),(1,0),(0,-1),(0,1)]

def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

def manhattan_distance(state):
    distance = 0
    for i in range(3):
        for j in range(3):
            val = state[i][j]
            if val != 0:
                goal_x = (val - 1) // 3
                goal_y = (val - 1) % 3
                distance += abs(i - goal_x) + abs(j - goal_y)
    return distance

def get_neighbors(state):
    neighbors = []
    x, y = find_blank(state)
    for dx, dy in MOVES:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_state = [list(row) for row in state]
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
            neighbors.append(tuple(tuple(row) for row in new_state))
    return neighbors

def a_star(start):
    pq = []
    heapq.heappush(pq, (manhattan_distance(start), 0, start, [start]))
    visited = set()

    while pq:
        f, g, current, path = heapq.heappop(pq)

        if current == GOAL_STATE:
            return path

        if current in visited:
            continue

        visited.add(current)

        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                new_g = g + 1
                new_f = new_g + manhattan_distance(neighbor)
                heapq.heappush(pq, (new_f, new_g, neighbor, path + [neighbor]))

    return None

def print_state(state):
    for row in state:
        print(row)
    print()

# User Input
print("Enter the initial state (use 0 for blank):")
initial = []
for i in range(3):
    initial.append(tuple(map(int, input().split())))
initial_state = tuple(initial)

solution = a_star(initial_state)

if solution:
    print("\nSolution Steps:")
    for step in solution:
        print_state(step)
    print("Total moves:", len(solution) - 1)
else:
    print("No solution found")
