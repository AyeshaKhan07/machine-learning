from collections import deque

graph = {'A': ['B', 'C'],
    'B': ['A', 'C', 'D', 'E'],
    'C': ['A', 'B', 'D'],
    'D': ['C', 'B', 'E'],
    'E': ['B', 'D']}

def bfs(start, goal, graph):
    queue = deque([start])
    visited = {start: None}

    while queue:
        cur_node = queue.popleft()
        if cur_node == goal:
            break

        next_nodes = graph[cur_node]
        for next_node in next_nodes:
            if next_node not in visited:
                queue.append(next_node)
                visited[next_node] = cur_node

        print("visited", visited)
    return visited

start = 'A'
goal = 'E'
visited = bfs(start, goal, graph)

cur_node = goal
print(f'\npath from {goal} to {start}: \n {goal} ', end='')
while cur_node != start:
    cur_node = visited[cur_node]
    print(f'---> {cur_node} ', end='')
print("\n")