import numpy as np
import cv2

import matplotlib.pyplot as plt
from queue import PriorityQueue

def create_occupancy_grid_from_image(image_path, threshold=128):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresholded = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    #thresholded = cv2.bitwise_not(thresholded)

    occupancy_grid = thresholded // 255

    return occupancy_grid,image

# Example usage:
image_path = "grid4.png"  

start = (350, 50)
goal = (50,350)

# A* search function
def a_star_search(occupancy_grid, start, goal):
    def heuristic(cell, goal):
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {cell: float('inf') for cell in np.ndindex(occupancy_grid.shape)}
    g_score[start] = 0

    while not open_set.empty():
        _, current = open_set.get()

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path

        for small_step_x, small_step_y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            x, y = current[0] + small_step_x, current[1] + small_step_y

            if 0 <= x < occupancy_grid.shape[1] and 0 <= y < occupancy_grid.shape[0] and occupancy_grid[y, x] == 1:
                tentative_g_score = g_score[current] + 1

                if tentative_g_score < g_score[(x, y)]:
                    came_from[(x, y)] = current
                    g_score[(x, y)] = tentative_g_score
                    f_score = tentative_g_score + heuristic((x, y), goal)
                    open_set.put((f_score, (x, y)))

    return None

# A function to reconstruct the path from the goal to the start
def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.insert(0, current)
    return path

occupancy_grid,image = create_occupancy_grid_from_image(image_path)

path_found = a_star_search(occupancy_grid, start, goal)

if path_found:
    print("Path found:")
    for cell in path_found:
        print(cell)
else:
    print("No path found")
    plt.imshow(occupancy_grid, cmap='gray', origin='lower')
    plt.plot(start[0], start[1], 'go', label='Start')
    plt.plot(goal[0], goal[1], 'ro', label='Goal')
    plt.legend()
    plt.title("Occupancy Grid without Path")
    plt.grid(True)
    plt.show()



plt.imshow(image, cmap='gray', origin='lower')
plt.colorbar()
plt.plot(start[0], start[1], 'go', label='Start')
plt.plot(goal[0], goal[1], 'ro', label='Goal')
plt.legend()
plt.title("Original Image")
plt.grid(True)
plt.show()


plt.imshow(occupancy_grid, cmap='gray', origin='lower')
plt.colorbar()
plt.plot([cell[0] for cell in path_found], [cell[1] for cell in path_found], marker='o', markersize=4, color='blue', label='Path')
plt.plot(start[0], start[1], 'go', label='Start')
plt.plot(goal[0], goal[1], 'ro', label='Goal')
plt.legend()
plt.title("Occupancy Grid with Path")
plt.grid(True)
plt.show()
