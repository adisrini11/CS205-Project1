import AIP1
import time
import matplotlib.pyplot as plt

ucs = [] # list that stores the running time of the Uniform Cost Search Algorithm on puzzles of different depths
a1 = [] # list that stores the running time of the A* Algorithm using Misplaced Tile Heuristic on puzzles of different depths
a2 = [] # list that stores the running time of the A* Algorithm using Manhattan Distance Heuristic on puzzles of different depths

algos = ['1','2','3']

for algo_sel in algos:
    for i in range(8):
        puzzle = AIP1.Puzzle()
        puzzle.create_custom_puzzle()
        t0 = time.time()
        ps = AIP1.PuzzleSolver(algo_sel,puzzle)
        ps.solve()
        t1 = time.time()
        time_taken = t1 - t0
        if(algo_sel=='1'):
            ucs.append(round(time_taken,3))
        
        if(algo_sel=='2'):
            a1.append(round(time_taken,3))

        if(algo_sel=='3'):
            a2.append(round(time_taken,3))

# Code to plot the Time vs Depth Plots

depths = [0, 2, 4, 8, 12, 16, 20, 24]
plt.plot(depths, ucs, marker='o', label='Uniform Cost Search')
plt.plot(depths, a1, marker='o', label='Misplaced Tile Heuristic')
plt.plot(depths, a2, marker='o', label='Manhattan Distance Heuristic')

plt.xticks(depths)
plt.xlabel('Depth of Puzzle')
plt.ylabel('Time (seconds)')
plt.title('Time vs Depth Plots Comparing Running times of the 3 Algorithms')
plt.legend()

plt.show()