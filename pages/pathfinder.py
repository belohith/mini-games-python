import streamlit as st
import queue
import matplotlib.pyplot as plt
import numpy as np
import time # NEW: Import time for delays in animation

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Maze Pathfinder", page_icon="🗺️")
st.title("🗺️ Maze Pathfinder")
st.markdown("""
This app visualizes a pathfinding algorithm (Breadth-First Search) solving a maze.
The path is shown from 'O' (Start) to 'X' (End).
Click 'Animate Solution' to see the BFS algorithm explore the maze step-by-step.
""")

# --- Maze Definition ---
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

# --- Core Pathfinder Logic (Adapted for Animation) ---

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

def find_path_bfs_animated(maze, start_char, end_char):
    """
    Finds the shortest path from start_char to end_char using BFS,
    and captures snapshots for animation.
    Returns a list of dictionaries, each containing {'visited': set, 'path': list}.
    """
    start_pos = find_start(maze, start_char)
    if not start_pos:
        st.error(f"Error: Start character '{start_char}' not found in maze.")
        return []

    q = queue.Queue()
    q.put((start_pos, [start_pos])) # Store (current_position, path_to_current_position)

    visited = set()
    visited.add(start_pos)

    animation_frames = []

    # Initial snapshot: just the start position, no path yet shown
    animation_frames.append({'visited': visited.copy(), 'path': []})

    while not q.empty():
        current_pos, path = q.get()
        row, col = current_pos

        # Add current state to frames (to show exploration)
        animation_frames.append({'visited': visited.copy(), 'path': path.copy()}) # path.copy() for current path
        
        # If we reached the end, add the final path and break
        if maze[row][col] == end_char:
            # Final snapshot: all visited cells, and the found path
            animation_frames.append({'visited': visited.copy(), 'path': path.copy()})
            return animation_frames # Return all collected frames

        # Explore neighbors
        neighbors = find_neighbors(maze, row, col)
        for neighbor in neighbors:
            r_n, c_n = neighbor
            if neighbor not in visited and maze[r_n][c_n] != "#": # Not visited and not a wall
                visited.add(neighbor)
                new_path = path + [neighbor]
                q.put((neighbor, new_path))
    
    st.warning("No path found to the destination!")
    return animation_frames # Return collected frames even if no path found

# --- Matplotlib Visualization Function ---

def draw_maze_matplotlib(maze_grid, visited_cells=None, current_path=None):
    """
    Draws the maze using matplotlib, highlighting visited cells and the current path.
    :param maze_grid: The maze as a list of lists of characters.
    :param visited_cells: Optional set of (row, col) tuples for visited cells (exploration).
    :param current_path: Optional list of (row, col) tuples representing the current path in BFS.
    :return: A matplotlib figure object.
    """
    rows = len(maze_grid)
    cols = len(maze_grid[0])

    # Create a numerical representation of the maze for plotting
    # Values will map to colors:
    # 0: empty path
    # 1: wall
    # 2: start 'O'
    # 3: end 'X'
    # 4: visited cells (explored)
    # 5: current path (leading to the cell being processed)
    
    numeric_maze_display = np.zeros((rows, cols))
    
    # First pass: map original maze elements
    for r in range(rows):
        for c in range(cols):
            if maze_grid[r][c] == '#':
                numeric_maze_display[r][c] = 1 # Wall
            elif maze_grid[r][c] == 'O':
                numeric_maze_display[r][c] = 2 # Start
            elif maze_grid[r][c] == 'X':
                numeric_maze_display[r][c] = 3 # End
            else:
                numeric_maze_display[r][c] = 0 # Empty

    # Second pass: Mark visited cells (exploration)
    if visited_cells:
        for r, c in visited_cells:
            # Only mark empty cells as visited; don't overwrite start/end/walls
            if numeric_maze_display[r][c] == 0:
                numeric_maze_display[r][c] = 4 # Visited (explored)

    # Third pass: Mark the current path (highest priority visualization)
    if current_path:
        for r, c in current_path:
            # Overwrite visited/empty cells for the current path
            if numeric_maze_display[r][c] in [0, 4]: 
                numeric_maze_display[r][c] = 5 # Current Path

    fig, ax = plt.subplots(figsize=(cols, rows))
    
    # Define custom colormap and normalization for 6 distinct values
    cmap_colors = ['#F0F0F0',    # 0: Empty (light gray)
                   '#333333',    # 1: Wall (dark gray/black)
                   '#4CAF50',    # 2: Start (green)
                   '#2196F3',    # 3: End (blue)
                   '#FFEB3B',    # 4: Visited (yellow - for exploration)
                   '#FF9800']    # 5: Current Path (orange - highlights the current path being evaluated)
    
    cmap = plt.matplotlib.colors.ListedColormap(cmap_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5] # Define boundaries for each value
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(numeric_maze_display, cmap=cmap, norm=norm, origin='upper', extent=[ -0.5, cols-0.5, rows-0.5, -0.5 ])
    
    # Add grid lines
    ax.set_xticks(np.arange(cols+1)-0.5, minor=True)
    ax.set_yticks(np.arange(rows+1)-0.5, minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)

    # Remove axis ticks and labels for cleaner display
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Annotate cells with their original characters (O, X, #)
    for r in range(rows):
        for c in range(cols):
            if maze_grid[r][c] in ['O', 'X', '#']: # Only annotate original fixed elements
                ax.text(c, r, maze_grid[r][c], ha='center', va='center', color='white' if maze_grid[r][c] == '#' else 'black', fontsize=12, weight='bold')

    plt.tight_layout()
    return fig

# --- Streamlit App Main Logic ---

def run_pathfinder_app():
    # Placeholder for the animated maze visualization
    animation_placeholder = st.empty()

    # Display initial maze when the page loads
    initial_fig = draw_maze_matplotlib(maze)
    animation_placeholder.pyplot(initial_fig)
    plt.close(initial_fig) # Close the figure to free memory

    st.markdown("---") # Separator

    # Slider for animation speed
    animation_speed = st.slider("Animation Speed (seconds per step)", 0.01, 0.5, 0.05, 0.01)

    if st.button("Animate Solution"):
        st.subheader("Solving Process:")
        with st.spinner("Generating animation frames..."):
            frames = find_path_bfs_animated(maze, 'O', 'X')
        
        if frames:
            for i, frame in enumerate(frames):
                # Use the placeholder to update the plot in place
                fig = draw_maze_matplotlib(maze, visited_cells=frame['visited'], current_path=frame['path'])
                animation_placeholder.pyplot(fig)
                plt.close(fig) # Close the figure immediately after displaying to free memory
                
                # Pause for the specified animation speed
                time.sleep(animation_speed) 
            
            # After loop, check if a path was actually found (last frame will have a path)
            if frames[-1]['path'] and maze[frames[-1]['path'][-1][0]][frames[-1]['path'][-1][1]] == 'X':
                 st.success(f"Path found in {len(frames)} steps!")
            else:
                 st.warning("No path found to the destination in the provided maze.") # Should be caught by find_path_bfs_animated
        else:
            st.error("Could not find a path or generate animation frames.")

# Run the Streamlit app function
run_pathfinder_app()