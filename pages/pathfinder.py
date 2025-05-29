import streamlit as st
import queue
import matplotlib.pyplot as plt
import numpy as np # Often useful with matplotlib for array manipulation

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Maze Pathfinder", page_icon="🗺️")
st.title("🗺️ Maze Pathfinder")
st.markdown("""
This app visualizes a pathfinding algorithm (Breadth-First Search) solving a maze.
The path is shown from 'O' (Start) to 'X' (End).
""")

# --- Maze Definition ---
# The maze definition remains the same
maze = [
    ["#", "O", "#", "#", "#", "#", "#", "#", "#"],
    ["#", " ", " ", " ", " ", " ", " ", " ", "#"],
    ["#", " ", "#", "#", " ", "#", "#", " ", "#"],
    ["#", " ", "#", " ", " ", " ", "#", " ", "#"],
    ["#", " ", "#", " ", "#", " ", "#", " ", "#"],
    ["#", " ", "#", " ", "#", " ", "#", " ", "#"],
    ["#", " ", "#", " ", "#", " ", "#", "#", "#"],
    ["#", " ", " ", " ", " ", " ", " ", " ", "#"],
    ["#", "#", "#", "#", "#", "#", "#", "X", "#"]
]

# --- Core Pathfinder Logic (Adapted) ---

def find_start(maze, start_char):
    """Finds the starting position of the maze."""
    for r, row in enumerate(maze):
        for c, value in enumerate(row):
            if value == start_char:
                return r, c
    return None

def find_neighbors(maze, r, c):
    """Finds valid neighbors (up, down, left, right) for a given cell."""
    neighbors = []
    rows, cols = len(maze), len(maze[0])

    if r > 0:  # UP
        neighbors.append((r - 1, c))
    if r + 1 < rows:  # DOWN
        neighbors.append((r + 1, c))
    if c > 0:  # LEFT
        neighbors.append((r, c - 1))
    if c + 1 < cols:  # RIGHT
        neighbors.append((r, c + 1))

    return neighbors

def find_path_bfs(maze, start_char, end_char):
    """
    Finds the shortest path from start_char to end_char using Breadth-First Search (BFS).
    Returns the path as a list of (row, col) tuples.
    """
    start_pos = find_start(maze, start_char)
    if not start_pos:
        st.error(f"Error: Start character '{start_char}' not found in maze.")
        return None

    q = queue.Queue()
    q.put((start_pos, [start_pos])) # Store (current_position, path_to_current_position)

    visited = set()
    visited.add(start_pos)

    while not q.empty():
        current_pos, path = q.get()
        row, col = current_pos

        # If we reached the end, return the path
        if maze[row][col] == end_char:
            return path

        # Explore neighbors
        neighbors = find_neighbors(maze, row, col)
        for neighbor in neighbors:
            r_n, c_n = neighbor
            if neighbor not in visited and maze[r_n][c_n] != "#": # Not visited and not a wall
                visited.add(neighbor)
                new_path = path + [neighbor]
                q.put((neighbor, new_path))
    
    st.warning("No path found to the destination!")
    return None # No path found

# --- Matplotlib Visualization Function ---

def draw_maze_matplotlib(maze_grid, path=None):
    """
    Draws the maze using matplotlib.
    :param maze_grid: The maze as a list of lists of characters.
    :param path: Optional list of (row, col) tuples representing the path.
    :return: A matplotlib figure object.
    """
    rows = len(maze_grid)
    cols = len(maze_grid[0])

    # Create a numerical representation of the maze for plotting
    # 0 for pathable, 1 for walls, 2 for start, 3 for end, 4 for found path
    numeric_maze = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            if maze_grid[r][c] == '#':
                numeric_maze[r][c] = 1 # Wall
            elif maze_grid[r][c] == 'O':
                numeric_maze[r][c] = 2 # Start
            elif maze_grid[r][c] == 'X':
                numeric_maze[r][c] = 3 # End
            else:
                numeric_maze[r][c] = 0 # Empty path

    # If a path is provided, mark it
    if path:
        for r, c in path:
            if maze_grid[r][c] != 'O' and maze_grid[r][c] != 'X': # Don't overwrite start/end
                numeric_maze[r][c] = 4 # Path

    fig, ax = plt.subplots(figsize=(cols, rows)) # Make figure size proportional to maze
    
    # Define custom colormap:
    # 0: empty path (white/light gray)
    # 1: wall (black)
    # 2: start (green)
    # 3: end (blue)
    # 4: found path (orange/yellow)
    cmap = plt.cm.get_cmap('Pastel1', 5) # Use a pastel colormap with 5 distinct colors
    colors = ['#F0F0F0', '#333333', '#4CAF50', '#2196F3', '#FFC107'] # light_gray, black, green, blue, amber
    cmap = plt.matplotlib.colors.ListedColormap(colors)

    ax.imshow(numeric_maze, cmap=cmap, origin='upper', extent=[ -0.5, cols-0.5, rows-0.5, -0.5 ]) # extent for grid lines
    
    # Add grid lines
    ax.set_xticks(np.arange(cols+1)-0.5, minor=True)
    ax.set_yticks(np.arange(rows+1)-0.5, minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0) # Hide minor tick marks

    # Remove axis ticks and labels for cleaner display
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Annotate cells with their original characters
    for r in range(rows):
        for c in range(cols):
            ax.text(c, r, maze_grid[r][c], ha='center', va='center', color='darkslategray', fontsize=12)

    plt.tight_layout() # Adjust layout to prevent labels overlapping
    return fig

# --- Streamlit App Main Logic ---

def run_pathfinder_app():
    st.subheader("Current Maze:")
    
    # Display initial maze (optional)
    initial_fig = draw_maze_matplotlib(maze)
    st.pyplot(initial_fig)
    plt.close(initial_fig) # Close the figure to free memory

    if st.button("Solve Maze"):
        with st.spinner("Finding path..."):
            solved_path = find_path_bfs(maze, 'O', 'X')
        
        if solved_path:
            st.success("Path found!")
            solved_fig = draw_maze_matplotlib(maze, solved_path)
            st.pyplot(solved_fig)
            plt.close(solved_fig) # Close the figure to free memory
        else:
            st.error("Could not find a path in the maze.")

# Run the Streamlit app function
run_pathfinder_app()