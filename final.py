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




#travelling salesmen 
from sys import maxsize
v=4

def travel(graph,s):
    vertex=[]
    for i in range(v):
        if i!=s:
            vertex.append(i)

    minpath=maxsize

    while True:
        present_cost=0
        k=s
        for i in range(len(vertex)):
            present_cost+=graph[k][vertex[i]]
            k=vertex[i]
            
        present_cost+=graph[k][s]
        minpath=min(minpath,present_cost)

    
        if not permutations(vertex):
            print("The path followed is:",vertex[::-1])
            break
    return minpath

def permutations(l):
        a=len(l)
        i=a-2
        
        while (i>=0 and l[i]>l[i+1]):
            i-=1
            
        if i==-1:
            return False
                   
        j=i+1
        while j<a and l[j]>l[i]:
            j+=1
            
        j-=1
        l[i],l[j]=l[j],l[i]
        left=i+1
        right=a-1
        
        while left<right:
            l[left],l[right]=l[right],l[left]
            left+= 1
            right-=1
        return True
        

graph=[[0,10,15,20],[10,0,35,25],[15,35,0,30],[20,25,30,0]]
s=int(input("Enter the starting node:"))
travel(graph,s)







#tower 

open = []
closed = []

start = [[2, 3, 4, 1], [], [], []]
final = [[1, 2, 3, 4], [], [], []]

table = {
    1: 0,
    2: [1],
    3: [1, 2],
    4: [1, 2, 3]
}

def heuristic(state):
    count = 0
    for i in range(0, 4):
        prev = 0
        for j in range(0, len(state[i])):
            if state[i][0:j] == table[state[i][j]]:
                count += state[i].index(state[i][j])
            else:
                count -= state[i].index(state[i][j])

    return count

def operate(currunt, s, d):
    l_currunt = []
    for tower in currunt:
        l_tower = []
        for item in tower:
            l_tower.append(item)
        l_currunt.append(l_tower)

    if len(l_currunt[s]) == 0:
        return l_currunt
    if len(l_currunt[s]) == 1 and len(l_currunt[d]) == 0:
        return l_currunt

    block = l_currunt[s][len(l_currunt[s]) - 1]
    l_currunt[s].pop()
    l_currunt[d].append(block)

    return l_currunt

def solve(currunt):
    open.append(currunt)
    while len(open) != 0:
        state = open[len(open) - 1]
        open.pop()

        print(state)

        if state.count([1, 2, 3, 4]):
            print('found')
            return

        for i in range(0, 4):
            for j in range(0, 4):
                if i == j:
                    continue
                n_state = operate(state, i, j)
                if state == n_state:
                    continue
                if heuristic(n_state) > heuristic(state):
                    open.append(n_state)
                    break

solve(start)


#a*
def aStarAlgo(start_node, stop_node):
         
        open_set = set(start_node) 
        closed_set = set()
        g = {} 
        parents = {}
        g[start_node] = 0
        parents[start_node] = start_node
         
         
        while len(open_set) > 0:
            n = None
            for v in open_set:
                if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):
                    n = v
             
                     
            if n == stop_node or Graph_nodes[n] == None:
                pass
            else:
                for (m, weight) in get_neighbors(n):
                    if m not in open_set and m not in closed_set:
                        open_set.add(m)
                        parents[m] = n
                        g[m] = g[n] + weight
                    else:
                        if g[m] > g[n] + weight:
                            g[m] = g[n] + weight
                            parents[m] = n
                            if m in closed_set:
                                closed_set.remove(m)
                                open_set.add(m)
 
            if n == None:
                print('Path does not exist!')
                return None
            if n == stop_node:
                path = []
 
                while parents[n] != n:
                    path.append(n)
                    n = parents[n]
 
                path.append(start_node)
 
                path.reverse()
 
                print('Path found: {}'.format(path))
                return path
 
            open_set.remove(n)
            closed_set.add(n)
 
        print('Path does not exist!')
        return None
def get_neighbors(v):
    if v in Graph_nodes:
        return Graph_nodes[v]
    else:
        return None
def heuristic(n):
        H_dist = {
            'A': 11,
            'B': 6,
            'C': 99,
            'D': 1,
            'E': 7,
            'G': 0,
             
        }
 
        return H_dist[n]
Graph_nodes = {
    'A': [('B', 2), ('E', 3)],
    'B': [('C', 1),('G', 9)],
    'C': None,
    'E': [('D', 6)],
    'D': [('G', 1)],
     
}
aStarAlgo('A', 'G')


#ao*

def Cost(H, condition, weight = 1):
    cost = {}
    if 'AND' in condition:
        AND_nodes = condition['AND']
        Path_A = ' AND '.join(AND_nodes)
        PathA = sum(H[node]+weight for node in AND_nodes)
        cost[Path_A] = PathA
 
    if 'OR' in condition:
        OR_nodes = condition['OR']
        Path_B =' OR '.join(OR_nodes)
        PathB = min(H[node]+weight for node in OR_nodes)
        cost[Path_B] = PathB
    return cost

def update_cost(H, Conditions, weight=1):
    Main_nodes = list(Conditions.keys())
    Main_nodes.reverse()
    least_cost= {}
    for key in Main_nodes:
        condition = Conditions[key]
        print(key,':', Conditions[key],'>>>', Cost(H, condition, weight))
        c = Cost(H, condition, weight)
        H[key] = min(c.values())
        least_cost[key] = Cost(H, condition, weight)           
    return least_cost

def shortest_path(Start,Updated_cost, H):
    Path = Start
    if Start in Updated_cost.keys():
        Min_cost = min(Updated_cost[Start].values())
        key = list(Updated_cost[Start].keys())
        values = list(Updated_cost[Start].values())
        Index = values.index(Min_cost)
        Next = key[Index].split()
        if len(Next) == 1:
 
            Start =Next[0]
            Path += ' = ' +shortest_path(Start, Updated_cost, H)
        else:
            Path +='=('+key[Index]+') '
 
            Start = Next[0]
            Path += '[' +shortest_path(Start, Updated_cost, H) + ' + '
 
            Start = Next[-1]
            Path +=  shortest_path(Start, Updated_cost, H) + ']'
 
    return Path
H1 = {'A': 1, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1, 'T': 3}
 
 
Conditions = {
 'A': {'OR': ['D'], 'AND': ['B', 'C']},
 'B': {'OR': ['G', 'H']},
 'C': {'OR': ['J']},
 'D': {'AND': ['E', 'F']},
 'G': {'OR': ['I']}
    
}

weight = 1
print('Updated Cost :')
Updated_cost = update_cost(H1, Conditions, weight=1)
print('*'*75)
print('Shortest Path :\n',shortest_path('A', Updated_cost,H1))


#DFS ID
# Python program to print DFS traversal from a given
# given graph
from collections import defaultdict

# This class represents a directed graph using adjacency
# list representation
class Graph:

	def __init__(self,vertices):

		# No. of vertices
		self.V = vertices

		# default dictionary to store graph
		self.graph = defaultdict(list)

	# function to add an edge to graph
	def addEdge(self,u,v):
		self.graph[u].append(v)

	# A function to perform a Depth-Limited search
	# from given source 'src'
	def DLS(self,src,target,maxDepth):

		if src == target : return True

		# If reached the maximum depth, stop recursing.
		if maxDepth <= 0 : return False

		# Recur for all the vertices adjacent to this vertex
		for i in self.graph[src]:
				if(self.DLS(i,target,maxDepth-1)):
					return True
		return False

	# IDDFS to search if target is reachable from v.
	# It uses recursive DLS()
	def IDDFS(self,src, target, maxDepth):

		# Repeatedly depth-limit search till the
		# maximum depth
		for i in range(maxDepth):
			if (self.DLS(src, target, i)):
				return True
		return False

# Create a graph given in the above diagram
g = Graph (7);
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 3)
g.addEdge(1, 4)
g.addEdge(2, 5)
g.addEdge(2, 6)

target = 6; maxDepth = 3; src = 0

if g.IDDFS(src, target, maxDepth) == True:
	print ("Target is reachable from source " +
		"within max depth")
else :
	print ("Target is NOT reachable from source " +
		"within max depth")
