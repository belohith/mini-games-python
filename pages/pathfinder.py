import streamlit as st
import queue
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import heapq # For A*, Dijkstra, Greedy Best-First priority queue

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Maze Pathfinder", page_icon="🗺️")
st.title("🗺️ Maze Pathfinder")
st.markdown("""
This app visualizes various pathfinding algorithms solving a maze.
Click **'Generate New Maze'** to create a unique solvable maze.
Then, choose an algorithm and click **'Animate Solution'** to see it explore and find a path.
""")

# --- Maze Configuration (You can adjust these values) ---
MAZE_ROWS = 15 
MAZE_COLS = 15
WALL_DENSITY = 0.35 

# --- Core Pathfinder Logic Functions ---

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
    Finds the shortest path from start_char to end_char using Breadth-First Search (BFS),
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

def find_path_dfs_animated(maze, start_char, end_char):
    """
    Finds a path from start_char to end_char using Depth-First Search (DFS),
    and captures snapshots for animation.
    Does NOT guarantee the shortest path.
    """
    start_pos = find_start(maze, start_char)
    if not start_pos:
        return []

    stack = [(start_pos, [start_pos])] # Store (current_position, path_to_current_position)

    visited = set()
    visited.add(start_pos)

    animation_frames = []
    animation_frames.append({'visited': visited.copy(), 'path': []}) # Initial frame

    while stack: # While stack is not empty
        current_pos, path = stack.pop() # Pop from the end (LIFO)
        row, col = current_pos

        animation_frames.append({'visited': visited.copy(), 'path': path.copy()}) 
        
        if maze[row][col] == end_char:
            animation_frames.append({'visited': visited.copy(), 'path': path.copy()})
            return animation_frames

        neighbors = find_neighbors(maze, row, col)
        for neighbor in reversed(neighbors): # Process neighbors in reverse order to simulate LIFO exploration
            r_n, c_n = neighbor
            if neighbor not in visited and maze[r_n][c_n] != "#":
                visited.add(neighbor)
                new_path = path + [neighbor]
                stack.append((neighbor, new_path)) # Push to stack

    return animation_frames # No path found

# A* / Dijkstra / Greedy Best-First Helper Function
def heuristic(pos1, pos2):
    """Calculates the Manhattan distance heuristic between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def find_path_astar_animated(maze, start_char, end_char):
    """
    Finds the shortest path from start_char to end_char using A* Search,
    and captures snapshots for animation.
    Returns a list of dictionaries, each containing {'visited': set, 'path': list}.
    """
    start_pos = find_start(maze, start_char)
    end_pos = find_start(maze, end_char) 
    if not start_pos or not end_pos:
        return []

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start_pos, end_pos), 0, start_pos))
    count = 0 

    came_from = {} 

    g_score = { (r, c): float('inf') for r in range(len(maze)) for c in range(len(maze[0])) }
    g_score[start_pos] = 0
    
    visited_nodes_in_order = set() 
    animation_frames = []
    animation_frames.append({'visited': visited_nodes_in_order.copy(), 'path': []})

    while open_set:
        current_f_score, _, current_pos = heapq.heappop(open_set)

        if current_pos in visited_nodes_in_order: 
            continue

        visited_nodes_in_order.add(current_pos)
        
        current_path = []
        temp_node = current_pos
        while temp_node in came_from:
            current_path.insert(0, temp_node)
            temp_node = came_from[temp_node]
        if start_pos not in current_path: 
            current_path.insert(0, start_pos)

        animation_frames.append({'visited': visited_nodes_in_order.copy(), 'path': current_path.copy()})

        if current_pos == end_pos:
            final_path = []
            temp_node = end_pos
            while temp_node in came_from:
                final_path.insert(0, temp_node)
                temp_node = came_from[temp_node]
            final_path.insert(0, start_pos) 
            animation_frames.append({'visited': visited_nodes_in_order.copy(), 'path': final_path.copy()})
            return animation_frames

        neighbors = find_neighbors(maze, current_pos[0], current_pos[1])
        for neighbor in neighbors:
            r_n, c_n = neighbor
            if maze[r_n][c_n] == '#': 
                continue

            tentative_g_score = g_score[current_pos] + 1 

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current_pos
                g_score[neighbor] = tentative_g_score
                f_score_val = g_score[neighbor] + heuristic(neighbor, end_pos)
                
                count += 1 
                heapq.heappush(open_set, (f_score_val, count, neighbor))
        
    return animation_frames

def find_path_dijkstra_animated(maze, start_char, end_char):
    """
    Finds the shortest path from start_char to end_char using Dijkstra's Algorithm,
    and captures snapshots for animation.
    Returns a list of dictionaries, each containing {'visited': set, 'path': list}.
    """
    start_pos = find_start(maze, start_char)
    end_pos = find_start(maze, end_char)
    if not start_pos or not end_pos:
        return []

    open_set = []
    heapq.heappush(open_set, (0, 0, start_pos))
    count = 0 

    came_from = {} 

    g_score = { (r, c): float('inf') for r in range(len(maze)) for c in range(len(maze[0])) }
    g_score[start_pos] = 0
    
    visited_nodes_in_order = set() 
    animation_frames = []
    animation_frames.append({'visited': visited_nodes_in_order.copy(), 'path': []})

    while open_set:
        current_g_score, _, current_pos = heapq.heappop(open_set)

        if current_g_score > g_score[current_pos]: 
            continue

        visited_nodes_in_order.add(current_pos)
        
        current_path = []
        temp_node = current_pos
        while temp_node in came_from:
            current_path.insert(0, temp_node)
            temp_node = came_from[temp_node]
        if start_pos not in current_path: 
            current_path.insert(0, start_pos)

        animation_frames.append({'visited': visited_nodes_in_order.copy(), 'path': current_path.copy()})

        if current_pos == end_pos:
            final_path = []
            temp_node = end_pos
            while temp_node in came_from:
                final_path.insert(0, temp_node)
                temp_node = came_from[temp_node]
            final_path.insert(0, start_pos) 
            animation_frames.append({'visited': visited_nodes_in_order.copy(), 'path': final_path.copy()})
            return animation_frames

        neighbors = find_neighbors(maze, current_pos[0], current_pos[1])
        for neighbor in neighbors:
            r_n, c_n = neighbor
            if maze[r_n][c_n] == '#': 
                continue

            tentative_g_score = g_score[current_pos] + 1 

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current_pos
                g_score[neighbor] = tentative_g_score
                
                count += 1 
                heapq.heappush(open_set, (g_score[neighbor], count, neighbor))
        
    return animation_frames

def find_path_greedy_best_first_animated(maze, start_char, end_char):
    """
    Finds a path from start_char to end_char using Greedy Best-First Search,
    and captures snapshots for animation. Does NOT guarantee the shortest path.
    Returns a list of dictionaries, each containing {'visited': set, 'path': list}.
    """
    start_pos = find_start(maze, start_char)
    end_pos = find_start(maze, end_char)
    if not start_pos or not end_pos:
        return []

    open_set = []
    heapq.heappush(open_set, (heuristic(start_pos, end_pos), 0, start_pos))
    count = 0 

    came_from = {} 
    
    visited = set() 
    visited.add(start_pos) 

    animation_frames = []
    animation_frames.append({'visited': visited.copy(), 'path': []})

    while open_set:
        current_h_score, _, current_pos = heapq.heappop(open_set)

        current_path = []
        temp_node = current_pos
        while temp_node in came_from:
            current_path.insert(0, temp_node)
            temp_node = came_from[temp_node]
        if start_pos not in current_path: 
            current_path.insert(0, start_pos)

        animation_frames.append({'visited': visited.copy(), 'path': current_path.copy()})

        if current_pos == end_pos:
            final_path = []
            temp_node = end_pos
            while temp_node in came_from:
                final_path.insert(0, temp_node)
                temp_node = came_from[temp_node]
            final_path.insert(0, start_pos) 
            animation_frames.append({'visited': visited.nodes_in_order.copy(), 'path': final_path.copy()}) # Fix: use visited from current context, not some internal variable
            return animation_frames

        neighbors = find_neighbors(maze, current_pos[0], current_pos[1])
        for neighbor in neighbors:
            r_n, c_n = neighbor
            if maze[r_n][c_n] == '#': 
                continue

            if neighbor not in visited: 
                visited.add(neighbor) 
                came_from[neighbor] = current_pos
                
                count += 1 
                heapq.heappush(open_set, (heuristic(neighbor, end_pos), count, neighbor))
        
    return animation_frames


# NEW: Wall Follower (Right-Hand Rule) Algorithm
def find_path_wall_follower_animated(maze, start_char, end_char):
    """
    Solves the maze using the Right-Hand Rule (Wall Follower),
    and captures snapshots for animation.
    Guaranteed to find a path in simply connected mazes.
    Returns a list of dictionaries, each containing {'visited': set, 'path': list}.
    """
    start_pos = find_start(maze, start_char)
    end_pos = find_start(maze, end_char)
    if not start_pos or not end_pos:
        return []

    path = [start_pos]
    visited = {start_pos} # Set of unique visited cells
    current_pos = start_pos

    # Directions: (dr, dc) for (Up, Right, Down, Left)
    # This dictates the order of checking for the "right hand"
    # Starting facing Right (0,1)
    # The direction represents (row_change, col_change) for the *current heading*
    DIRECTIONS = {
        (0, 1): 0,  # Facing Right, index 0
        (1, 0): 1,  # Facing Down, index 1
        (0, -1): 2, # Facing Left, index 2
        (-1, 0): 3  # Facing Up, index 3
    }
    # List of directions to map index to (dr, dc)
    DIRECTION_VECTORS = [(0, 1), (1, 0), (0, -1), (-1, 0)] # R, D, L, U
    
    # Start facing right (0, 1) or whatever fits maze orientation
    # Find a valid starting direction if start is near a wall
    current_direction_index = -1 
    # Try to find a valid initial direction (e.g., away from the wall behind start)
    # Simplest: assume start is (0,1) and we move (0,1) -> (0,2) meaning facing Right (index 0)
    # Or, if start is (R, C) and end is (R_end, C_end), try to face towards end
    
    # A simple way to initialize direction: assume it points to the first valid neighbor on the way
    # from start to end (e.g., if end is to the right, try to face right)
    initial_dr, initial_dc = 0, 0
    if end_pos[0] > start_pos[0]: initial_dr = 1 # Towards end row
    elif end_pos[0] < start_pos[0]: initial_dr = -1 # Towards end row
    if end_pos[1] > start_pos[1]: initial_dc = 1 # Towards end col
    elif end_pos[1] < start_pos[1]: initial_dc = -1 # Towards end col

    # Default to facing right if no clear path or at the start
    if initial_dr == 0 and initial_dc == 0: # If start and end are same or adjacent
        initial_dr, initial_dc = 0, 1 # Face right
    elif initial_dr != 0 and initial_dc != 0: # If diagonal, pick one axis
        initial_dc = 0 # Prioritize vertical movement first
        
    current_direction_index = DIRECTIONS.get((initial_dr, initial_dc), 0) # Default to Right if not found


    # Animation frames
    animation_frames = []
    animation_frames.append({'visited': visited.copy(), 'path': path.copy()})

    # Loop until end is reached or stuck
    max_steps = len(maze) * len(maze[0]) * 4 # Prevent infinite loops in complex mazes/dead ends
    steps_taken = 0

    while current_pos != end_pos and steps_taken < max_steps:
        steps_taken += 1
        
        # Calculate the "right" turn relative to current_direction_index
        # If current_direction_index is 0 (Right), right_turn_index is 1 (Down)
        # If current_direction_index is 1 (Down), right_turn_index is 2 (Left)
        # etc.
        right_turn_index = (current_direction_index + 1) % 4
        
        # Check right (relative to current direction)
        dr_right, dc_right = DIRECTION_VECTORS[right_turn_index]
        next_r_right, next_c_right = current_pos[0] + dr_right, current_pos[1] + dc_right

        # Check forward (relative to current direction)
        dr_forward, dc_forward = DIRECTION_VECTORS[current_direction_index]
        next_r_forward, next_c_forward = current_pos[0] + dr_forward, current_pos[1] + dc_forward

        # Check left (relative to current direction)
        left_turn_index = (current_direction_index - 1 + 4) % 4
        dr_left, dc_left = DIRECTION_VECTORS[left_turn_index]
        next_r_left, next_c_left = current_pos[0] + dr_left, current_pos[1] + dc_left

        moved = False

        # 1. Try to turn right
        if 0 <= next_r_right < len(maze) and 0 <= next_c_right < len(maze[0]) and maze[next_r_right][next_c_right] != '#':
            current_pos = (next_r_right, next_c_right)
            current_direction_index = right_turn_index
            moved = True
        # 2. Else, try to move forward
        elif 0 <= next_r_forward < len(maze) and 0 <= next_c_forward < len(maze[0]) and maze[next_r_forward][next_c_forward] != '#':
            current_pos = (next_r_forward, next_c_forward)
            # current_direction_index remains the same
            moved = True
        # 3. Else, try to turn left
        elif 0 <= next_r_left < len(maze) and 0 <= next_c_left < len(maze[0]) and maze[next_r_left][next_c_left] != '#':
            current_pos = (next_r_left, next_c_left)
            current_direction_index = left_turn_index
            moved = True
        # 4. Else, turn around (dead end)
        else:
            current_direction_index = (current_direction_index + 2) % 4 # Turn 180 degrees
            # No movement, just change direction. Next iteration will try to move in new direction.
            # No 'moved = True' here, as we didn't move to a new cell.
            # To avoid getting stuck in a corner, we might need a small step back or re-evaluate.
            # For animation, we can add a frame even if just turning.
            animation_frames.append({'visited': visited.copy(), 'path': path.copy()}) # Add frame for turning around


        if moved:
            if current_pos not in visited:
                visited.add(current_pos)
            path.append(current_pos)
            animation_frames.append({'visited': visited.copy(), 'path': path.copy()})
        elif not moved and current_pos == path[-1]: # If didn't move but also not at destination, and path didn't change
            # This indicates being completely stuck or just turning in place repeatedly
            # Add a break condition to prevent infinite loops if wall follower can't make progress
            if steps_taken > 1 and path[-1] == path[-2]: # Check if stuck in last 2 steps
                 break # Stuck, can't move
            
    # Final check if end was reached
    if current_pos == end_pos:
        return animation_frames
    else:
        st.warning("Wall Follower might not find a path in mazes with islands or complex structures, or it got stuck. Try generating a new maze.")
        return [] # Return empty frames if path not found

# --- Random Maze Generation Function (Unchanged except for random start/end placement logic improvement) ---
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
        all_empty_spots = [(r,c) for r in range(rows) for c in range(cols) if new_maze_grid[r][c] == ' ']
        
        if len(all_empty_spots) < 2: # Not enough space for start and end
            continue

        start_pos = random.choice(all_empty_spots)
        all_empty_spots.remove(start_pos) # Remove chosen start_pos from consideration for end_pos
        end_pos = random.choice(all_empty_spots)

        temp_maze_for_check = [row[:] for row in new_maze_grid] # Deep copy for check
        temp_maze_for_check[start_pos[0]][start_pos[1]] = start_char
        temp_maze_for_check[end_pos[0]][end_pos[1]] = end_char
        
        # Check if a path exists using a simplified BFS (without animation frame capturing)
        q_check = queue.Queue()
        initial_check_pos = find_start(temp_maze_for_check, start_char)
        if not initial_check_pos: # Start not placed correctly (shouldn't happen with valid_start_options)
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
            new_maze_grid[start_pos[0]][start_pos[1]] = start_char
            new_maze_grid[end_pos[0]][end_pos[1]] = end_char
            return new_maze_grid
    
    st.error(f"Could not generate a solvable maze after {max_attempts} attempts. Returning a fixed default maze.")
    # Fallback to a simple, known-solvable maze if random generation consistently fails
    return [
        ["#", "O", "#"],
        ["#", " ", "#"],
        ["#", "X", "#"]
    ]

# --- Matplotlib Visualization Function (Unchanged) ---
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

    # Algorithm selection
    st.subheader("Algorithm Settings:")
    selected_algorithm = st.selectbox(
        "Choose Pathfinding Algorithm:",
        ("Breadth-First Search (BFS)", "Depth-First Search (DFS)", "A* Search",
         "Dijkstra's Algorithm", "Greedy Best-First Search", "Wall Follower (Right-Hand Rule)") # NEW: Wall Follower
    )
    
    st.markdown("---") # Separator

    # Controls and buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Generate New Maze"):
            with st.spinner("Generating a new solvable maze..."):
                st.session_state.current_maze = generate_random_maze(MAZE_ROWS, MAZE_COLS, WALL_DENSITY)
            st.success("New maze generated! Choose an algorithm and click 'Animate Solution' to solve.")
            st.rerun() # Rerun to display the new maze immediately

    with col2:
        animation_speed = st.slider("Animation Speed (seconds per step)", 0.01, 0.5, 0.05, 0.01, key="speed_slider")

        if st.button("Animate Solution"):
            st.subheader(f"Solving Process ({selected_algorithm}):") # Show chosen algorithm in header
            with st.spinner(f"Running {selected_algorithm}..."):
                # Call the appropriate algorithm based on selection
                if selected_algorithm == "Breadth-First Search (BFS)":
                    frames = find_path_bfs_animated(st.session_state.current_maze, 'O', 'X')
                elif selected_algorithm == "Depth-First Search (DFS)":
                    frames = find_path_dfs_animated(st.session_state.current_maze, 'O', 'X')
                elif selected_algorithm == "A* Search":
                    frames = find_path_astar_animated(st.session_state.current_maze, 'O', 'X')
                elif selected_algorithm == "Dijkstra's Algorithm":
                    frames = find_path_dijkstra_animated(st.session_state.current_maze, 'O', 'X')
                elif selected_algorithm == "Greedy Best-First Search":
                    frames = find_path_greedy_best_first_animated(st.session_state.current_maze, 'O', 'X')
                elif selected_algorithm == "Wall Follower (Right-Hand Rule)": # NEW: Wall Follower call
                    frames = find_path_wall_follower_animated(st.session_state.current_maze, 'O', 'X')
                else:
                    st.error("Invalid algorithm selected.")
                    frames = [] 
            
            # Animation loop and success/error messages
            if frames and frames[-1]['path'] and st.session_state.current_maze[frames[-1]['path'][-1][0]][frames[-1]['path'][-1][1]] == 'X':
                for i, frame in enumerate(frames):
                    fig = draw_maze_matplotlib(st.session_state.current_maze, visited_cells=frame['visited'], current_path=frame['path'])
                    animation_placeholder.pyplot(fig)
                    plt.close(fig)
                    time.sleep(animation_speed) 
                st.success(f"Path found using {selected_algorithm} in {len(frames)} steps!")
            else:
                 st.error(f"No path found to the destination using {selected_algorithm} in the current maze. Try generating a new one.")


# Run the Streamlit app function
run_pathfinder_app()