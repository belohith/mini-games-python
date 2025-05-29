import streamlit as st
import queue
import matplotlib.pyplot as plt
import numpy as np
import time
import random # NEW: For random maze generation

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Maze Pathfinder", page_icon="🗺️")
st.title("🗺️ Maze Pathfinder")
st.markdown("""
This app visualizes a pathfinding algorithm (Breadth-First Search) solving a maze.
Click **'Generate New Maze'** to create a unique solvable maze.
Then, click **'Animate Solution'** to see the BFS algorithm explore it step-by-step.
""")

# --- Maze Configuration (You can adjust these values) ---
MAZE_ROWS = 9
MAZE_COLS = 9
WALL_DENSITY = 0.35 # Percentage of cells that will become walls (approx.)

# --- Core Pathfinder Logic Functions (Unchanged from previous version) ---

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
        return []

    q = queue.Queue()
    q.put((start_pos, [start_pos])) # Store (current_position, path_to_current_position)

    visited = set()
    visited.add(start_pos)

    animation_frames = []
    animation_frames.append({'visited': visited.copy(), 'path': []}) # Initial frame

    while not q.empty():
        current_pos, path = q.get()
        row, col = current_pos

        animation_frames.append({'visited': visited.copy(), 'path': path.copy()}) # Frame for current step
        
        if maze[row][col] == end_char:
            animation_frames.append({'visited': visited.copy(), 'path': path.copy()}) # Final path frame
            return animation_frames

        neighbors = find_neighbors(maze, row, col)
        for neighbor in neighbors:
            r_n, c_n = neighbor
            if neighbor not in visited and maze[r_n][c_n] != "#":
                visited.add(neighbor)
                new_path = path + [neighbor]
                q.put((neighbor, new_path))
    
    return animation_frames # Return collected frames even if no path found (no path exists)

# --- NEW: Random Maze Generation Function ---
def generate_random_maze(rows, cols, wall_density, start_char='O', end_char='X', max_attempts=100):
    """Generates a random maze and ensures it's solvable."""
    if rows < 3 or cols < 3:
        st.error("Maze dimensions must be at least 3x3 for a meaningful maze.")
        return None

    for attempt in range(max_attempts):
        new_maze_grid = []
        for r in range(rows):
            row_chars = []
            for c in range(cols):
                if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                    row_chars.append("#") # Always make borders walls
                elif random.random() < wall_density:
                    row_chars.append("#") # Random wall
                else:
                    row_chars.append(" ") # Empty space
            new_maze_grid.append(row_chars)

        # Place start and end characters ensuring they are on pathable cells and distinct
        # We'll try to put them in fixed spots near corners for consistency, but if those are walls, find alternatives.
        possible_start_spots = [(0, 1), (1, 0), (1, 1)]
        possible_end_spots = [(rows - 1, cols - 2), (rows - 2, cols - 1), (rows - 2, cols - 2)]

        start_r, start_c = random.choice(possible_start_spots)
        end_r, end_c = random.choice(possible_end_spots)

        # Ensure start and end are within bounds for very small mazes and not same spot
        start_r = min(start_r, rows - 1)
        start_c = min(start_c, cols - 1)
        end_r = min(end_r, rows - 1)
        end_c = min(end_c, cols - 1)
        
        if (start_r, start_c) == (end_r, end_c): # If by chance they pick same spot
            continue # Try again

        temp_maze_for_check = [row[:] for row in new_maze_grid] # Deep copy for check
        temp_maze_for_check[start_r][start_c] = start_char
        temp_maze_for_check[end_r][end_c] = end_char
        
        # Check if a path exists
        # We'll use a simplified BFS (without animation frame capturing) for this check
        q_check = queue.Queue()
        initial_check_pos = find_start(temp_maze_for_check, start_char)
        if not initial_check_pos: # Start not placed correctly
            continue
        q_check.put(initial_check_pos)
        
        visited_check = set()
        visited_check.add(initial_check_pos)

        solvable = False
        while not q_check.empty():
            r_check, c_check = q_check.get()

            if temp_maze_for_check[r_check][c_check] == end_char:
                solvable = True
                break

            neighbors_check = find_neighbors(temp_maze_for_check, r_check, c_check)
            for nr_check, nc_check in neighbors_check:
                if (nr_check, nc_check) not in visited_check and temp_maze_for_check[nr_check][nc_check] != "#":
                    visited_check.add((nr_check, nc_check))
                    q_check.put((nr_check, nc_check))
        
        if solvable:
            # If solvable, set 'O' and 'X' in the actual maze grid to be returned
            new_maze_grid[start_r][start_c] = start_char
            new_maze_grid[end_r][end_c] = end_char
            return new_maze_grid
    
    st.error(f"Could not generate a solvable maze after {max_attempts} attempts. Returning a default maze.")
    # Fallback to a simple, known-solvable maze if random generation consistently fails
    return [
        ["#", "O", "#"],
        ["#", " ", "#"],
        ["#", "X", "#"]
    ]

# --- Matplotlib Visualization Function (Unchanged from previous version) ---
def draw_maze_matplotlib(maze_grid, visited_cells=None, current_path=None):
    """
    Draws the maze using matplotlib, highlighting visited cells and the current path.
    """
    rows = len(maze_grid)
    cols = len(maze_grid[0])

    numeric_maze_display = np.zeros((rows, cols))
    
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

    if visited_cells:
        for r, c in visited_cells:
            if numeric_maze_display[r][c] == 0:
                numeric_maze_display[r][c] = 4 # Visited (explored)

    if current_path:
        for r, c in current_path:
            if numeric_maze_display[r][c] in [0, 4]: 
                numeric_maze_display[r][c] = 5 # Current Path

    fig, ax = plt.subplots(figsize=(cols, rows))
    
    cmap_colors = ['#F0F0F0',    # 0: Empty (light gray)
                   '#333333',    # 1: Wall (dark gray/black)
                   '#4CAF50',    # 2: Start (green)
                   '#2196F3',    # 3: End (blue)
                   '#FFEB3B',    # 4: Visited (yellow - for exploration)
                   '#FF9800']    # 5: Current Path (orange - highlights the current path being evaluated)
    
    cmap = plt.matplotlib.colors.ListedColormap(cmap_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(numeric_maze_display, cmap=cmap, norm=norm, origin='upper', extent=[ -0.5, cols-0.5, rows-0.5, -0.5 ])
    
    ax.set_xticks(np.arange(cols+1)-0.5, minor=True)
    ax.set_yticks(np.arange(rows+1)-0.5, minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    for r in range(rows):
        for c in range(cols):
            if maze_grid[r][c] in ['O', 'X', '#']:
                ax.text(c, r, maze_grid[r][c], ha='center', va='center', color='white' if maze_grid[r][c] == '#' else 'black', fontsize=12, weight='bold')

    plt.tight_layout()
    return fig

# --- Streamlit App Main Logic ---

def run_pathfinder_app():
    # Initialize maze in session state if not already present
    if 'current_maze' not in st.session_state:
        st.session_state.current_maze = generate_random_maze(MAZE_ROWS, MAZE_COLS, WALL_DENSITY)
    
    # Placeholder for the animated maze visualization
    animation_placeholder = st.empty()

    # Display the current maze (either default or newly generated)
    initial_fig = draw_maze_matplotlib(st.session_state.current_maze)
    animation_placeholder.pyplot(initial_fig)
    plt.close(initial_fig)

    st.markdown("---") # Separator

    # Controls and buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Generate New Maze"):
            with st.spinner("Generating a new solvable maze..."):
                st.session_state.current_maze = generate_random_maze(MAZE_ROWS, MAZE_COLS, WALL_DENSITY)
            st.success("New maze generated! Click 'Animate Solution' to solve.")
            st.rerun() # Rerun to display the new maze immediately

    with col2:
        animation_speed = st.slider("Animation Speed (seconds per step)", 0.01, 0.5, 0.05, 0.01, key="speed_slider")

        if st.button("Animate Solution"):
            st.subheader("Solving Process:")
            with st.spinner("Generating animation frames..."):
                frames = find_path_bfs_animated(st.session_state.current_maze, 'O', 'X')
            
            if frames and frames[-1]['path'] and st.session_state.current_maze[frames[-1]['path'][-1][0]][frames[-1]['path'][-1][1]] == 'X':
                for i, frame in enumerate(frames):
                    fig = draw_maze_matplotlib(st.session_state.current_maze, visited_cells=frame['visited'], current_path=frame['path'])
                    animation_placeholder.pyplot(fig)
                    plt.close(fig)
                    time.sleep(animation_speed) 
                st.success(f"Path found in {len(frames)} steps!")
            else:
                 st.error("No path found to the destination in the current maze. Try generating a new one.")


# Run the Streamlit app function
run_pathfinder_app()