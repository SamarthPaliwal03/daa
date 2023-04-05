# bfs
graph = {
  '5' : ['3','7'],
  '3' : ['2', '4'],
  '7' : ['8'],
  '2' : [],
  '4' : ['8'],
  '8' : []
}

def bfs(graph,node):
    visited =[]
    queue = []
    visited.append(node)
    queue.append(node)

    while queue:
        current = queue.pop(0)
        print(current,end=" ")

        for neighbour in graph[current]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)


bfs(graph,'5')

# dfs
graph = {
    'A' : ['B','C'],
    'B' : ['D', 'E'],
    'C' : ['F'],
    'D' : [],
    'E' : ['F'],
    'F' : []
}

visited =[]

def dfs(visited, graph,node):
    if node not in visited:
        visited.append(node)
        print(node,end=' ')
    
    for neighbour in graph[node]:
        dfs(visited,graph,neighbour)


dfs(visited,graph,'A')

# ucs

from queue import PriorityQueue

graph = {
    'A': [('B', 2), ('C', 4)],
    'B': [('D', 1), ('E', 4)],
    'C': [('D', 5), ('F', 1)],
    'D': [('F', 2)],
    'E': [('F', 3)],
    'F': []
}

def ucs(graph,start,goal):
    queue = PriorityQueue()
    queue.put((0,start,[]))
    visited =[]

    while queue:
        cost, node, path = queue.get()
        if node not in visited:
            print(node)
            visited.append(node)
            path = path + [node]

            if node==goal:
                return path, cost
            
            for child_node, child_cost in graph[node]:
                if child_node not in visited:
                    queue.put((cost+child_cost,child_node,path))

    return [],float("inf")

path,cost = ucs(graph,'A','F')
print(path)
print(cost)



# eight puzzle with heuristic
from queue import PriorityQueue
import numpy as np

goalState = [[1,2,3],
            [8,0,4],
            [7,6,5]]

def isGoal(state):
    return state==goalState

def heuristic(state):
    return np.sum(state!=goalState)

def findZero(state):
    for i in range(3):
        for j in range(3):
            if state[i][j]==0:
                return i,j

def swap(state,i1,j1,i2,j2):
    new_state = [row[:] for row in state]
    new_state[i1][j1] , new_state[i2][j2] = new_state[i2][j2], new_state[i1][j1]
    return new_state

def next_States(state):
    next_states = []
    zero_i,zero_j = findZero(state)
    if zero_i > 0:
        next_states.append(swap(state, zero_i, zero_j, zero_i-1, zero_j))
    if zero_i < 2:
        next_states.append(swap(state, zero_i, zero_j, zero_i+1, zero_j))
    if zero_j > 0:
        next_states.append(swap(state, zero_i, zero_j, zero_i, zero_j-1))
    if zero_j < 2:
        next_states.append(swap(state, zero_i, zero_j, zero_i, zero_j+1))
    return next_states

def bestFirstSearch(start_node):
    queue = PriorityQueue()
    visited = set()
    queue.put((heuristic(start_node),start_node,[]))

    while queue:
        h, currrent_node , path = queue.get()

        if isGoal(currrent_node):
            return path
        
        visited.add(tuple(map(tuple,currrent_node)))

        for next in next_States(currrent_node):
            if tuple(map(tuple,next)) not in visited:
                queue.put((heuristic(next),next,path+[next]))

    return None

start_node = [[0,2,3],
              [1,8,4],
              [7,6,5]]
path = bestFirstSearch(start_node)
print(path)



# eight puzzle with dfs
from queue import Queue
goalState = [[1,2,3],
            [8,0,4],
            [7,6,5]]

def isGoal(state):
    return state == goalState

def findZero(state):
    for i in range(3):
        for j in range(3):
            if state[i][j]==0:
                return i,j
    
def swap(state,i1,j1,i2,j2):
    new_state = [row[:] for row in state]
    new_state[i1][j1] , new_state[i2][j2] = new_state[i2][j2], new_state[i1][j1]
    return new_state

def get_next_move(state):
    zero_i , zero_j = findZero(state)
    next_state= []
    if zero_i<2:
        next_state.append(swap(state,zero_i,zero_j,zero_i+1,zero_j))
    if zero_i>0:
        next_state.append(swap(state,zero_i,zero_j,zero_i-1,zero_j))
    if zero_j<2:
        next_state.append(swap(state,zero_i,zero_j,zero_i,zero_j+1))
    if zero_j>0:
        next_state.append(swap(state,zero_i,zero_j,zero_i,zero_j-1))
    return next_state

def dfs(start_node):
    visited = set()
    stack = [(start_node,[])]
    while stack:
        current_state, path = stack.pop()
        visited.add(tuple(map(tuple,current_state)))
        if isGoal(current_state):
            return path
        for next_state in get_next_move(current_state):
            if tuple(map(tuple,next_state)) not in visited:
                print(tuple(map(tuple,next_state)))
                stack.append((next_state,path+[next_state]))
        
start_node = [[2,0,3],
              [1,8,4],
              [7,6,5]]
path = dfs(start_node)
print(path)
    

# eight puzzle using bfs
from queue import Queue

# goalState = [[1,2,3],
#             [4,5,6],
#             [7,8,0]]
# goalState = [[3,8,6],
#             [1,7,2],
#             [4,5,0]]
goalState = [[1,2,3],
            [8,0,4],
            [7,6,5]]

def isGoal(state):
    if state == goalState:
        return True
    return False

def findZero(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i,j


def swap(state,i1,j1,i2,j2):
    new_state = [row[:] for row in state]
    new_state[i1][j1], new_state[i2][j2] = new_state[i2][j2], new_state[i1][j1]
    return new_state

def get_next_state(state):
    next_states =[]
    zero_i,zero_j = findZero(state)
    if zero_i > 0:
        next_states.append(swap(state, zero_i, zero_j, zero_i-1, zero_j))
    if zero_i < 2:
        next_states.append(swap(state, zero_i, zero_j, zero_i+1, zero_j))
    if zero_j > 0:
        next_states.append(swap(state, zero_i, zero_j, zero_i, zero_j-1))
    if zero_j < 2:
        next_states.append(swap(state, zero_i, zero_j, zero_i, zero_j+1))
    
    return next_states

def bfs(initialState):
    visited = set()
    q = Queue()
    q.put((initial_state, []))
    visited.add(tuple(map(tuple, initial_state)))
    while q:
        current_state, path = q.get()
        if isGoal(current_state):
            return path
        
        for next_state in get_next_state(current_state):
            if tuple(map(tuple, next_state)) not in visited:
                print(tuple(map(tuple, next_state)))
                visited.add(tuple(map(tuple, next_state)))
                q.put((next_state,path+[next_state]))


initial_state = [[2,0,3],
                 [1,8,4],
                 [7,6,5]]
path = bfs(initial_state)
print("Path to goal state:", path)
        
        

#  water jug

from queue import Queue

def nextState(current,max_x,max_y):
    states = []
    # fill x
    states.append((max_x,current[1]))
    # fill y
    states.append((current[0],max_y))
    # Empty x
    states.append((0,current[1]))
    #Empty y
    states.append((current[0],max_y))
    # Transfer x to y
    pour_x_y = min(current[0],max_y-current[1])
    states.append((current[0]-pour_x_y,current[1]+pour_x_y))
    # Transfer y to x
    pour_y_x = min(current[1],max_x-current[0])
    states.append((current[0]+pour_y_x,current[1]-pour_y_x))

    if(states.index(current)):
        states.remove(current)
    
    return set(states)



def bfs(start,goal,max_x,max_y):
    q = Queue()
    q.put(start)
    visited = set()

    while q:
        current = q.get()
        if current==goal:
            return True
        visited.add(current)
        States = nextState(current,max_x,max_y)
        print("Current State ",current)
        print("Next States")
        for next in States:
            if next not in visited:
                q.put(next)
                print(next)
    return False


start_state = (0,0)
goal_state = (2,0)
max_x = 4
max_y = 3
is_reaxhable = bfs(start_state,goal_state,max_x,max_y)
print(is_reaxhable)

# block not correct

# Define the initial and goal states
initial_state = {
    "A": "-",
    "B": "A",
    "C": "B"
}
goal_state = {
    "A": "-",
    "B": "C",
    "C": "A"
}

# Define a stack to keep track of the nodes to explore
stack = [initial_state]

# Define a set to keep track of the visited nodes
visited = set()

# Define a dictionary to keep track of the parent nodes
parent = {str(initial_state): None}

# Define a function to generate the child nodes
# def generate_children(node):
#     children = []
#     for block, below in node.items():
#         if below == "-":  # move block onto table
#             child = node.copy()
#             child[block] = "-"
#             children.append(child)
#         else:
#             for other_block, other_below in node.items():
#                 if other_block != block and other_below == "-":  # move block onto other block
#                     child = node.copy()
#                     child[block] = other_below
#                     child[other_block] = block
#                     children.append(child)
#     return children

# # Define the DFS algorithm
# def dfs():
#     while stack:
#         node = stack.pop()
#         if node == goal_state:
#             # Construct the path from the initial state to the goal state
#             path = []
#             while node is not None:
#                 path.append(node)
#                 node = parent[str(node)]
#             path.reverse()
#             return path
#         if str(node) not in visited:
#             visited.add(str(node))
#             children = generate_children(node)
#             for child in reversed(children):
#                 if str(child) not in visited:
#                     stack.append(child)
#                     parent[str(child)] = node
#     return None  # goal state not found

# Call the DFS function and print the result


def position(node):
    table = set()
    top = set()
    key_list = list(node.keys())
    val_list = list(node.values())
    for block,below in node.items():
        if below=='-':
            table.add(block)
        else:
            if block in val_list:
                upper = val_list.index(block)
            if upper:
                continue
            else:
                top.add(key_list[upper])
    return table,top


def next(node):
    print('Next called')
    children = []
    table, top = position(node)
    print(table,top)
    for i in table:
        for j in top:
            copy = node.copy()
            copy[table[i]] = top[j]
            children.append(copy)
            copy2 = node.copy()
            copy2[top[j]] = table[i]
            children.append(copy2)
            
    for i in top:
        copy = node.copy
        copy[top] = '-'
        children.append(copy)

    return children
            
def dfs():
    while stack:
        node = stack.pop()
        if node == goal_state:
            path = []
            while node is not None:
                path.append(node)
                node = parent[str(node)]
            path.reverse()
            return path
        if str(node) not in visited:
            visited.add(str(node))
            for children in next(node):
                if str(children) not in visited:
                    print(children)
                    stack.append(children)
                    parent[str(children)] = node 
    return None
result = dfs()
if result is not None:
    print("The path from the initial state to the goal state is:")
    for node in result:
        print(node)
else:
    print("The goal state could not be reached.")