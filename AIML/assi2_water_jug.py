# Water Jug Problem using BFS (State Space Search)

from collections import deque

def water_jug_bfs(jug1_capacity, jug2_capacity, target):
    visited = set()
    queue = deque()

    # state: (jug1, jug2)
    queue.append((0, 0, []))

    while queue:
        jug1, jug2, path = queue.popleft()

        if (jug1, jug2) in visited:
            continue

        visited.add((jug1, jug2))
        path = path + [(jug1, jug2)]

        if jug1 == target or jug2 == target:
            return path

        # Possible actions
        states = [
            (jug1_capacity, jug2),          # Fill Jug1
            (jug1, jug2_capacity),          # Fill Jug2
            (0, jug2),                      # Empty Jug1
            (jug1, 0),                      # Empty Jug2
            (jug1 - min(jug1, jug2_capacity - jug2),
             jug2 + min(jug1, jug2_capacity - jug2)),  # Pour Jug1 -> Jug2
            (jug1 + min(jug2, jug1_capacity - jug1),
             jug2 - min(jug2, jug1_capacity - jug1))   # Pour Jug2 -> Jug1
        ]

        for state in states:
            if state not in visited:
                queue.append((state[0], state[1], path))

    return None

# User Input
jug1_capacity = int(input("Enter Jug1 Capacity: "))
jug2_capacity = int(input("Enter Jug2 Capacity: "))
target = int(input("Enter Target Amount: "))

solution = water_jug_bfs(jug1_capacity, jug2_capacity, target)

if solution:
    print("\nSteps to reach target:")
    for step in solution:
        print(step)
else:
    print("No solution found")
