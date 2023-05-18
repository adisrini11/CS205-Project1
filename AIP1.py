# Importing Dependencies
import heapq
from itertools import chain
from copy import deepcopy
import time

class LegalMoves:

    def generate_legal_moves(self):

        # This function ensures that only legal moves are made on the board at any given state.

        blank_space_indices = self.initial_puzzle.get_blank_space_index()
        puzzle_array_sizes = self.initial_puzzle.get_puzzle_array_size()
        moves = []

        up = down = left = right = False

        if blank_space_indices[0] == 0:
            down = True
        elif blank_space_indices[0] == puzzle_array_sizes[0] - 1:
            up = True
        else:
            up = down = True

        if blank_space_indices[1] == 0:
            right = True
        elif blank_space_indices[1] == puzzle_array_sizes[1] - 1:
            left = True
        else:
            left = right = True

        if up:
            moves.append(self.move_up())
        if down:
            moves.append(self.move_down())
        if right:
            moves.append(self.move_right())
        if left:
            moves.append(self.move_left())

        return moves



    def move_up(self):
        
        # Moves the blank space up one row and returns a copy of the puzzle after moving the blank up.

        blank_space_indices = self.initial_puzzle.get_blank_space_index()
        puzzle_copy = deepcopy(self.initial_puzzle)

        if blank_space_indices[0] > 0:
            temp_value = puzzle_copy.get_index_value(blank_space_indices[0] - 1, blank_space_indices[1])
            puzzle_copy.set_puzzle(blank_space_indices[0] - 1, blank_space_indices[1], 0)
            puzzle_copy.set_puzzle(blank_space_indices[0], blank_space_indices[1], temp_value)

        return puzzle_copy




    def move_down(self):

        # Moves the blank space down one row and returns a copy of the puzzle after moving the blank down.

        blank_space_indices = self.initial_puzzle.get_blank_space_index()
        puzzle_copy = deepcopy(self.initial_puzzle)

        if blank_space_indices[0] + 1 < len(puzzle_copy.get_puzzle()):
            temp_value = puzzle_copy.get_index_value(blank_space_indices[0] + 1, blank_space_indices[1])
            puzzle_copy.set_puzzle(blank_space_indices[0] + 1, blank_space_indices[1], 0)
            puzzle_copy.set_puzzle(blank_space_indices[0], blank_space_indices[1], temp_value)

        return puzzle_copy


    def move_left(self):

        # Moves the blank space left one column and returns a copy of the puzzle after moving the blank left.

        blank_space_indices = self.initial_puzzle.get_blank_space_index()
        puzzle_copy = deepcopy(self.initial_puzzle)

        if blank_space_indices[1] > 0:
            temp_value = puzzle_copy.get_index_value(blank_space_indices[0], blank_space_indices[1] - 1)
            puzzle_copy.set_puzzle(blank_space_indices[0], blank_space_indices[1] - 1, 0)
            puzzle_copy.set_puzzle(blank_space_indices[0], blank_space_indices[1], temp_value)

        return puzzle_copy

    def move_right(self):

        #Moves the blank space right one column and returns a copy of the puzzle after moving the blank right.

        blank_space_indices = self.initial_puzzle.get_blank_space_index()
        puzzle_copy = deepcopy(self.initial_puzzle)

        if blank_space_indices[1] + 1 < len(puzzle_copy.get_puzzle()[0]):
            temp_value = puzzle_copy.get_index_value(blank_space_indices[0], blank_space_indices[1] + 1)
            puzzle_copy.set_puzzle(blank_space_indices[0], blank_space_indices[1] + 1, 0)
            puzzle_copy.set_puzzle(blank_space_indices[0], blank_space_indices[1], temp_value)

        return puzzle_copy


class Node(LegalMoves):

    # This class contains functions that help represent board states as nodes and update the heuristic for each board state.

    def __lt__(self, other):

        b = self.fn < other.fn
        return b

    def __init__(self, initial_puzzle, parent_puzzle, gn, heuristic):
        self.initial_puzzle = initial_puzzle
        self.parent_puzzle = parent_puzzle
        self.gn = gn + 1
        self.hn = None
        self.fn = None 
        self.heuristics = Heuristics()
        self.set_hn(heuristic)
        self.set_fn()

    def set_fn(self):
        
        self.fn = self.get_gn() + self.get_hn()

    def set_hn(self, heuristic):
        
        if heuristic == 'uniform':
            self.hn = 0
        elif heuristic == 'misplaced_tiles':
            self.hn = self.heuristics.misplaced_tile_heuristic(self.initial_puzzle)
        elif heuristic == 'manhattan':
            self.hn = self.heuristics.manhattan_distance_heuristic(self.initial_puzzle)

    def get_gn(self):
        return self.gn

    def get_hn(self):
        return self.hn
    
    def get_parent(self):
        
        return self.parent_puzzle

    def get_puzzle(self):
        
        return self.initial_puzzle.get_puzzle()

    def get_blank_space_index(self):
        
        return self.initial_puzzle.get_blank_space_index()

    def print_puzzle(self):
        
        self.initial_puzzle.print_puzzle()


class Heuristics:

    # This class contains the functions that determine the heuristic values for Misplaced Tile strategy and Manhattan Distance strategy 

    @staticmethod
    def misplaced_tile_heuristic(initial_puzzle):
        expected = 1
        misplaced_tiles = 0
        
        puzzle_array_sizes = initial_puzzle.get_puzzle_array_size()
        size1 = puzzle_array_sizes[0]
        size2 = puzzle_array_sizes[1]
        for i in range(size1):
            for j in range(size2):
                temp = puzzle_array_sizes[0] - 1
                if i == temp:
                    if j == puzzle_array_sizes[1] - 1:
                        if initial_puzzle.get_index_value(i, j) != 0:
                            misplaced_tiles += 1
                
                elif initial_puzzle.get_index_value(i, j) != expected:
                    misplaced_tiles += 1 

                expected += 1

        return misplaced_tiles

    @staticmethod
    
    def manhattan_distance_heuristic(initial_puzzle):
        distance = 0
        puzzle_array_sizes = initial_puzzle.get_puzzle_array_size()

        for i in range(puzzle_array_sizes[0]):
            for j in range(puzzle_array_sizes[1]):
                temp = initial_puzzle.get_index_value(i, j)

                if temp != 0:
                    row = (temp - 1) // puzzle_array_sizes[0]
                    column = (temp - 1) % puzzle_array_sizes[1]
                else:
                    row = puzzle_array_sizes[0] - 1
                    column = puzzle_array_sizes[1] - 1
                    # x = abs(row - i)
                    # y = abs(column - j)
                distance = distance + abs(row - i) + abs(column - j)

        return distance


class Puzzle:

    # This class contains functions to initialize the puzzle based on user input and we can change the puzzle size from here.

    def __init__(self):
        puzzle = None

    def get_puzzle(self):
        
        return self.puzzle

    def set_puzzle(self, puzzle_index1, puzzle_index2, new_puzzle_value):
        

        self.puzzle[puzzle_index1][puzzle_index2] = new_puzzle_value

    def print_puzzle(self):
        
        for row in self.puzzle:
            print(*row, sep=' ')

    def create_default_puzzle(self):
       
        self.puzzle = [[1, 2, 3],
                       [4, 5, 6],
                       [0, 7, 8]]
                       


    def create_custom_puzzle(self):
        custom_puzzle = []
        print('Enter your puzzle, use a zero to represent the blank')

        for i in range(3):
            user_selection = input('Enter the row, use space or tabs between numbers: ')
            row_list = [int(i) for i in user_selection.split()]
            custom_puzzle.append(row_list)

        self.puzzle = custom_puzzle
        print()
        self.print_puzzle()
        print()


    def get_index_value(self, puzzle_index1, puzzle_index2):
        
        return self.puzzle[puzzle_index1][puzzle_index2]



    def get_blank_space_index(self):
        for i, row in enumerate(self.puzzle):
            if 0 in row:
                return (i, row.index(0))
        raise ValueError("Blank space not found in puzzle.")


    def get_puzzle_array_size(self):

        return [len(self.get_puzzle()), len(self.get_puzzle()[0])]



class Solver:

    # This class contains functions to update the heuristic values and the general search algorithm as described in the slides.

    def __init__(self, algorithm, puzzle):
        if not isinstance(puzzle, Puzzle):
            raise TypeError('puzzle must be an instance of the Puzzle class.')
        self.puzzle = puzzle
        self.algorithm = algorithm
        self.heuristic = None
        self.goal_state = None
        self.set_heuristic()

    def set_heuristic(self):

        match self.algorithm:
            case '1':
                self.heuristic = 'uniform'
            case '2':
                self.heuristic = 'misplaced_tiles'
            case '3':
                self.heuristic = 'manhattan'



    def general_search(self):

        self.generate_goal_state()

        # Stores puzzles that have already been expanded
        visited = []
        queue = [(0, Node(self.puzzle, self.puzzle, -1, self.heuristic))]
        max_queue_size, nodes_explored = 0, 0

        while queue:
            max_queue_size = max(len(queue), max_queue_size)
            cost, current_node = heapq.heappop(queue)
            nodes_explored += 1

            if current_node.get_puzzle() not in visited:
                visited.append(current_node.get_puzzle())
                if current_node.get_puzzle() == self.goal_state:
                    print(f'\nThe best state to expand with a g(n) = {current_node.get_gn()} and h(n) = {current_node.get_hn()}')
                    current_node.print_puzzle()
                    print('\nGoal!!!')
                    print(f'\nTo solve this problem the search algorithm expanded a total of {nodes_explored} nodes.')
                    print(f'The maximum number of nodes in the queue at any one time was {len(queue)}')
                    print(f'The depth of the goal node was {current_node.get_gn()}')
                    return 1

                if nodes_explored == 1:
                    print('\nExpanding state')
                    current_node.print_puzzle()
                else:
                    print(f'\nThe best state to expand with a g(n) = {current_node.get_gn()} and h(n) = {current_node.get_hn()}')
                    current_node.print_puzzle()

                moves = current_node.generate_legal_moves()
                for puzzle in moves:
                    next_node = Node(puzzle, current_node, current_node.get_gn(), self.heuristic)
                    temp = (next_node.fn, next_node)
                    heapq.heappush(queue, temp)

        return 0


    
    
    def generate_goal_state(self):
        flat_list = list(chain.from_iterable(self.puzzle.get_puzzle()))
        flat_list = sorted(flat_list, key=lambda x: x if x > 0 else float('inf'))

        puzzle_size = len(self.puzzle.get_puzzle())
        list_size = len(self.puzzle.get_puzzle()[0])

        goal_state = []
        k = 0

        for i in range(puzzle_size):
            idx = flat_list[k:k+list_size]
            k += list_size
            goal_state.append(idx)

        self.goal_state = goal_state


    def get_goal_state(self):
        return self.goal_state



class PuzzleSolver:
    def __init__(self, algorithm_selection, puzzle):
        self.algorithm_selection = algorithm_selection
        self.puzzle = puzzle

    def solve(self):
        solver = Solver(self.algorithm_selection, self.puzzle)
        solver.general_search()


def main():
    puzzle = Puzzle()

    while True:
        user_selection = input('Enter "1" for the default puzzle or "2" for a custom puzzle: ')

        if user_selection == '1':
            print('Creating default puzzle\n')
            puzzle.create_default_puzzle()
            break
        elif user_selection == '2':
            print('Creating custom puzzle\n')
            puzzle.create_custom_puzzle()
            break
        else:
            print('Invalid input. Please choose "1" or "2".')

    while True:
        algorithm_selection = input('Enter your choice of algorithm:\n'
                                    '\t1. Uniform Cost Search\n'
                                    '\t2. A* with the Misplaced Tile Heuristic\n'
                                    '\t3. A* with the Manhattan Distance Heuristic\n')

        if algorithm_selection in {'1', '2', '3'}:
            algorithm_map = {
                '1': 'Uniform Cost Search',
                '2': 'A* with Misplaced Tile Heuristic',
                '3': 'A* with Manhattan Distance Heuristic'
            }
            print(algorithm_map[algorithm_selection])
            break
        else:
            print('Invalid input. Please choose a valid algorithm (1, 2, or 3).')
    t0 = time.time()
    puzzle_solver = PuzzleSolver(algorithm_selection, puzzle)
    puzzle_solver.solve()
    t1=time.time()
    exec_time = round((t1-t0),3)
    print("Executed in time:",exec_time)
if __name__ == '__main__':
    main()