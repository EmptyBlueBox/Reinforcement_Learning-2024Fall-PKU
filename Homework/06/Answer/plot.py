import numpy as np
import matplotlib.pyplot as plt

# Read data from files
blocking_dynaq = np.loadtxt('MazePolicyDynaQ-Blocking_Maze.txt')
blocking_dynaq_plus = np.loadtxt('MazePolicyDynaQ_plus-Blocking_Maze.txt')
shortcut_dynaq = np.loadtxt('MazePolicyDynaQ-Shortcut_Maze.txt')
shortcut_dynaq_plus = np.loadtxt('MazePolicyDynaQ_plus-Shortcut_Maze.txt')

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot Blocking Maze
ax1.plot(blocking_dynaq[:, 0], blocking_dynaq[:, 1], label='Dyna-Q', color='blue')
ax1.plot(blocking_dynaq_plus[:, 0], blocking_dynaq_plus[:, 1], label='Dyna-Q+', color='red')
ax1.axvline(x=1000, color='gray', linestyle='--', label='Environment Change')
ax1.set_title('Blocking Maze')
ax1.set_xlabel('Episodes')
ax1.set_ylabel('Cumulative Reward')
ax1.legend()
ax1.grid(True)

# Plot Shortcut Maze
ax2.plot(shortcut_dynaq[:, 0], shortcut_dynaq[:, 1], label='Dyna-Q', color='blue')
ax2.plot(shortcut_dynaq_plus[:, 0], shortcut_dynaq_plus[:, 1], label='Dyna-Q+', color='red')
ax2.axvline(x=3000, color='gray', linestyle='--', label='Environment Change')
ax2.set_title('Shortcut Maze')
ax2.set_xlabel('Episodes')
ax2.set_ylabel('Cumulative Reward')
ax2.legend()
ax2.grid(True)

# Adjust layout and save
plt.tight_layout()
plt.savefig('maze_experiments.png')
plt.close()
