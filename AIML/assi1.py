# Informed Search Algorithm (A*) – Real-Life Problem
# Problem: Shortest path between cities using A* Search

import heapq

# Graph: road distances between cities (real-life navigation problem)
graph = {
    'Mumbai': [('Pune', 150), ('Surat', 280)],
    'Pune': [('Mumbai', 150), ('Bangalore', 840), ('Hyderabad', 560)],
    'Surat': [('Mumbai', 280), ('Jaipur', 670)],
    'Jaipur': [('Surat', 670), ('Delhi', 280)],
    'Hyderabad': [('Pune', 560), ('Bangalore', 570), ('Chennai', 630)],
    'Bangalore': [('Pune', 840), ('Hyderabad', 570), ('Chennai', 350)],
    'Chennai': [('Bangalore', 350), ('Hyderabad', 630), ('Delhi', 2200)],
    'Delhi': [('Jaipur', 280), ('Chennai', 2200)]
}

# Heuristic: estimated distance to destination (Delhi)
heuristic = {
    'Mumbai': 1400,
    'Pune': 1200,
    'Surat': 1100,
    'Jaipur': 300,
    'Hyderabad': 900,
    'Bangalore': 1700,
    'Chennai': 2000,
    'Delhi': 0
}

def a_star_search(start, goal):
    pq = []
    heapq.heappush(pq, (0 + heuristic[start], 0, start, [start]))
    visited = set()

    while pq:
        f, g, current, path = heapq.heappop(pq)

        if current == goal:
            return path, g

        if current in visited:
            continue

        visited.add(current)

        for neighbor, cost in graph.get(current, []):
            if neighbor not in visited:
                new_g = g + cost
                new_f = new_g + heuristic[neighbor]
                heapq.heappush(pq, (new_f, new_g, neighbor, path + [neighbor]))

    return None, float('inf')

# User Input
start_city = input("Enter Start City: ")
goal_city = input("Enter Goal City: ")

path, total_distance = a_star_search(start_city, goal_city)

if path:
    print("\nShortest Path Found:")
    print(" -> ".join(path))
    print("Total Distance:", total_distance, "km")
else:
    print("No path found")
