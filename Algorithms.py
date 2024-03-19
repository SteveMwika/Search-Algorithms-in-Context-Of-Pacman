import util
from collections import deque
import heapq

'''
action_costs = {
    'up': 1,      # Assign a cost of 1 to the 'up' action
    'down': 1,    # Assign a cost of 1 to the 'down' action
    'left': 1,    # Assign a cost of 1 to the 'left' action
    'right': 1,   # Assign a cost of 1 to the 'right' action
    'jump': 2,    # Assign a cost of 2 to the 'jump' action (example of a more costly action)
    # Add more actions and their associated costs as needed
}

class Created_Problem:
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state
        self.goal_state = goal_state

    def getStartState(self):
        return self.initial_state

    def isGoalState(self, state):
        return state == self.goal_state


    def getSuccessors(self, state):
        x, y = state  # Assuming state is represented as (x, y)
        successors = []

        # Define possible moves and their associated costs
        possible_moves = [(1, 0, 'Right', 1), (-1, 0, 'Left', 1), (0, 1, 'Up', 1), (0, -1, 'Down', 1)]

        for dx, dy, action, cost in possible_moves:
            next_x, next_y = x + dx, y + dy

            # You can define custom bounds or constraints for your problem
            # For example, a 3x3 grid with custom bounds:
            if 0 <= next_x <= 50 and 0 <= next_y <= 50:
                next_state = (next_x, next_y)
                successors.append((next_state, action, cost))

        return successors
    
    
    def getCostOfActions(self, actions):
        total_cost = 0
        for action in actions:
            total_cost += action_costs.get(action, 0)
        return total_cost



# Example usage:

'''

class DFS(object):
    def depthFirstSearch(self, problem):
        """
        Search the deepest nodes in the search tree first
        [2nd Edition: p 75, 3rd Edition: p 87]

        Your search algorithm needs to return a list of actions that reaches
        the goal.  Make sure to implement a graph search algorithm
        [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

        To get started, you might want to try some of these simple commands to
        understand the search problem that is being passed in:

        print "Start:", problem.getStartState()
        print "Is the start a goal?", problem.isGoalState(problem.getStartState())
        print "Start's successors:", problem.getSuccessors(problem.getStartState())
        """
        "*** TTU CS 5368 Fall 2023 YOUR CODE HERE ***"

        stack = util.Stack() # Initialize a stack for DFS
        gone_To = set()  # Initialize a set to keep track of visited states

        # Push the initial state onto the stack
        initial_state = problem.getStartState()
        stack.push((initial_state, []))

        while not stack.isEmpty():
             # Pop the current state and its associated actions
            state, actions = stack.pop()
            
            # If the goal state is reached, return the list of actions
            if problem.isGoalState(state):
                return actions

            if state not in gone_To:
                # Mark the current state as visited
                gone_To.add(state)
                # Get the successors of the current state
                successors = problem.getSuccessors(state)
                for next_state, action, cost in successors:
                    if next_state not in gone_To:
                        # Create a new list of actions with the current action
                        new_actions = actions + [action]
                        # Push the next state and new actions onto the stack
                        stack.push((next_state, new_actions))
        return []
    
class BFS(object):
    def breadthFirstSearch(self, problem):
        "*** TTU CS 5368 Fall 2023 YOUR CODE HERE ***"
        # a stack is used as a queue to implement a breadth-first search.
        stack = util.Stack()  # Initialize a stack for BFS. stores states and actions
        gone_To = set()  # Create a set to keep track of gone_To states

        initial_state = problem.getStartState()
        stack.push((initial_state, []))  # Push the initial state onto the stack

        while not stack.isEmpty():
            # Pop the current state and actions
            state, actions = stack.pop()
            # If goal state reached return  list of actions
            if problem.isGoalState(state):
                return actions
            #mark as Gone TO
            if state not in gone_To:
                gone_To.add(state)
                # Get the successors of the current state
                successors = problem.getSuccessors(state)
                for next_state, action, cost in successors:
                    if next_state not in gone_To:
                        # Create a new list of actions with the current action
                        new_actions = actions + [action]
                        # Push the next state and new actions onto the stack
                        stack.push((next_state, new_actions))
        return []
            
        util.raiseNotDefined()

class UCS(object):
    def uniformCostSearch(self, problem):
        "*** TTU CS 5368 Fall 2023 YOUR CODE HERE ***"
        start_state = problem.getStartState()
        if problem.isGoalState(start_state):
            return []  # Already at the goal state

        gone_To = set() #keep ack of where we are.
        priority_queue = [] # priority queue stores states and  paths with their costs
        heapq.heappush(priority_queue, (0, (start_state, [])))  # Push the start state with a cost of 0 and an empty path onto the priority queue

        while priority_queue:
            cost, (current_state, path) = heapq.heappop(priority_queue)
            if current_state in gone_To:
                continue # Already been Skip.

            gone_To.add(current_state) 

            if problem.isGoalState(current_state):
                return path #  goal state reached, return the path 

            for successor, action, step_cost in problem.getSuccessors(current_state):  
                if successor not in gone_To:
                    # Create a new path by extending the current path with the current action
                    new_path = path + [action]
                    
                    # Calculate the new cost by adding the step cost
                    new_cost = cost + step_cost
                    
                    # Push the successor state with the new cost and path onto the priority queue
                    heapq.heappush(priority_queue, (new_cost, (successor, new_path)))

        return []  # If no solution found reurn Empy pah

        util.raiseNotDefined()
        
class aSearch (object):
    def nullHeuristic( state, problem=None):
        """
        A heuristic function estimates the cost from the current state to the nearest goal in the provided SearchProblem.  This heuristic is trivial.
        """
        return 0
    def aStarSearch(self,problem, heuristic=nullHeuristic):
        "Search the node that has the lowest combined cost and heuristic first."
        "*** TTU CS 5368 Fall 2023 YOUR CODE HERE ***"

        start_state = problem.getStartState()
        # priority queue to manage states and their priorities
        work_set = util.PriorityQueue() 
        # Push the initial state with an empty action list and priority 0
        work_set.push((start_state, []), 0)
        # set to keep track of visited states
        goneTo_set = set()

        while not work_set.isEmpty(): # While PrioriTy queue work_set is noT empTy: 
            # Get the state with the lowest combined cost and heuristic estimate
            current_state, actions = work_set.pop()

            if problem.isGoalState(current_state):
                return actions  # Return list of actions when goal is reached

            if current_state not in goneTo_set:
                # Mark state as visited
                goneTo_set.add(current_state)
                 # Get successor states and actions
                successors = problem.getSuccessors(current_state)

                for next_state, action, step_cost in successors: 
                    if next_state not in goneTo_set:
                        # Extend list of actions with current action
                        new_actions = actions + [action]
                        # Calculate path cost
                        g_cost = problem.getCostOfActions(new_actions)
                        # Estimate cost to goal state using the heuristic Null HueriTic 
                        h_cost = heuristic(next_state, problem)
                        # Calculate combined cost and heuristic estimate
                        f_cost = g_cost + h_cost
                        # Add the next state and actions to the open set with the new priority
                        work_set.push((next_state, new_actions), f_cost)

        return []  # Return an empty list if no solution is found
        
        util.raiseNotDefined()

if __name__ == '__main__':
    
    '''
    initial_state = (0, 0)
    goal_state = (5, 5)
    problem = Created_Problem(initial_state, goal_state)
    print(problem)
  

    
    print("\nDEPTH FIST SEARCH BELOW")
    dfs = DFS()
    dfs_solution = dfs.depthFirstSearch(problem)
    if dfs_solution:
        print("\nDFS Solution found:", dfs_solution)
    else:
        print("No solution found.")
    

    "**************************************************************************"

    print("\nBEADTH FIRST SEARCH BELOW")
    bfs = BFS()
    # Use BFS to search for a solution
    bfs_solution = bfs.breadthFirstSearch(problem)

    if bfs_solution:
        print("\nBFS Solution found:", bfs_solution)
    else:
        print("No solution found.")


    "**************************************************************************"

    print("\nUNIFORM SEARCH BELOW")
    ucs = UCS()

    # Use UCS to search for a solution
    ucs_solution = ucs.uniformCostSearch(problem)

    if ucs_solution:
        print("\nUCS Solution found:", ucs_solution)
    else:
        print("No solution found.")

    "**************************************************************************"

    print("\nA* SEARCH BELOW" )

    aStar  = aSearch()

    aStar_solution = aStar.aStarSearch(problem)

    if aStar_solution:
        print("\nA*_SEARCH Solution found:", aStar_solution)
    else:
        print("No solution found.")

'''
    