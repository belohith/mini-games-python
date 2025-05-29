import streamlit as st
import random
import time
from PIL import Image, ImageDraw # NEW: For drawing on images
import io # NEW: For image buffer

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Aim Trainer (Web Adaptation)", page_icon="🎯")
st.title("🎯 Aim Trainer (Web Adaptation)")
st.markdown("""
This is a web-adapted version of the classic Aim Trainer game.
Due to web browser limitations, it won't be as real-time or precise as a desktop Pygame version,
but it simulates the core mechanics.
""")

# --- Game Configuration ---
WIDTH, HEIGHT = 800, 600
TOP_BAR_HEIGHT = 50
BG_COLOR = (0, 25, 40) # Dark blue
TARGET_PADDING = 30
LIVES = 3 # Number of targets allowed to shrink to 0

# Target properties
TARGET_MAX_SIZE = 30
TARGET_GROWTH_RATE = 0.2
TARGET_COLOR = (255, 0, 0) # Red
TARGET_SECOND_COLOR = (255, 255, 255) # White

# Game timing
TARGET_SPAWN_INTERVAL = 0.8 # seconds between new targets appearing
GAME_DURATION = 30 # seconds

# --- Game State Management (using st.session_state) ---
# Initialize game state
if 'aim_game_started' not in st.session_state:
    st.session_state.aim_game_started = False
    st.session_state.aim_targets = []
    st.session_state.aim_targets_pressed = 0
    st.session_state.aim_misses = 0
    st.session_state.aim_start_time = 0
    st.session_state.aim_last_target_spawn_time = 0
    st.session_state.aim_game_over = False
    st.session_state.aim_clicks = 0 # Total clicks (for accuracy)

class Target:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = 0
        self.grow = True
        self.hit = False # New: flag if target was hit

    def update(self, delta_time):
        if self.hit: # If hit, shrink rapidly
            self.size -= self.GROWTH_RATE * 5 # Faster shrink
            if self.size <= 0:
                return False # Indicate it's gone
            return True

        # Normal grow/shrink
        if self.grow:
            self.size += self.GROWTH_RATE * delta_time * 60 # Scale growth by delta_time to be consistent
            if self.size >= TARGET_MAX_SIZE:
                self.grow = False
        else:
            self.size -= self.GROWTH_RATE * delta_time * 60 # Scale shrink
            if self.size <= 0:
                return False # Indicate it's gone (missed)
        return True

    def draw(self, draw_obj):
        # Draw concentric circles
        if self.size > 0:
            draw_obj.ellipse(
                (self.x - self.size, self.y - self.size,
                 self.x + self.size, self.y + self.size),
                fill=TARGET_COLOR
            )
            draw_obj.ellipse(
                (self.x - self.size * 0.8, self.y - self.size * 0.8,
                 self.x + self.size * 0.8, self.y + self.size * 0.8),
                fill=TARGET_SECOND_COLOR
            )
            draw_obj.ellipse(
                (self.x - self.size * 0.6, self.y - self.size * 0.6,
                 self.x + self.size * 0.6, self.y + self.size * 0.6),
                fill=TARGET_COLOR
            )
            draw_obj.ellipse(
                (self.x - self.size * 0.4, self.y - self.size * 0.4,
                 self.x + self.size * 0.4, self.y + self.size * 0.4),
                fill=TARGET_SECOND_COLOR
            )

    def collide(self, click_x, click_y):
        # Simple circular collision
        distance = math.sqrt((click_x - self.x)**2 + (click_y - self.y)**2)
        return distance <= self.size


def create_game_image(targets, elapsed_time, targets_pressed, misses):
    # Create a blank image
    img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
    draw_obj = ImageDraw.Draw(img)

    # Draw targets
    for target in targets:
        target.draw(draw_obj)

    # Draw top bar
    draw_obj.rectangle((0, 0, WIDTH, TOP_BAR_HEIGHT), fill=(150, 150, 150)) # Grey bar

    # Add text to top bar (using a simple font, PIL doesn't use Pygame fonts)
    # We'll use a default font or load one if needed. For simplicity, just use PIL's default.
    try:
        from PIL import ImageFont
        # Try to load a default font, or use a simpler one if not found
        font = ImageFont.truetype("arial.ttf", 20) # Common font on many systems
    except IOError:
        font = ImageFont.load_default()

    # Format time
    milli = math.floor(int(elapsed_time * 1000 % 1000) / 100)
    seconds = int(round(elapsed_time % 60, 0)) # Round to nearest second for display
    minutes = int(elapsed_time // 60)
    time_str = f"Time: {minutes:02d}:{seconds:02d}.{milli}"

    speed = round(targets_pressed / elapsed_time, 1) if elapsed_time > 0 else 0.0
    speed_str = f"Speed: {speed} t/s"

    hits_str = f"Hits: {targets_pressed}"
    lives_str = f"Lives: {LIVES - misses}"

    # Draw text with basic positioning
    draw_obj.text((5, 10), time_str, fill="black", font=font)
    draw_obj.text((200, 10), speed_str, fill="black", font=font)
    draw_obj.text((450, 10), hits_str, fill="black", font=font)
    draw_obj.text((650, 10), lives_str, fill="black", font=font)

    # Convert PIL Image to bytes for Streamlit
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def display_end_screen(elapsed_time, targets_pressed, clicks):
    st.subheader("Game Over!")
    st.markdown("---")

    accuracy = round(targets_pressed / clicks * 100, 1) if clicks > 0 else 0.0

    st.metric(label="Total Time", value=f"{int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d}")
    st.metric(label="Targets Hit", value=targets_pressed)
    st.metric(label="Accuracy", value=f"{accuracy}%")
    st.metric(label="Speed (Targets/sec)", value=round(targets_pressed / elapsed_time, 1) if elapsed_time > 0 else 0.0)

    if st.button("Play Again?", key="aim_play_again_btn"):
        # Reset all session state variables for a new game
        st.session_state.aim_game_started = False
        st.session_state.aim_targets = []
        st.session_state.aim_targets_pressed = 0
        st.session_state.aim_misses = 0
        st.session_state.aim_start_time = 0
        st.session_state.aim_last_target_spawn_time = 0
        st.session_state.aim_game_over = False
        st.session_state.aim_clicks = 0
        st.rerun()


def run_aim_trainer_app():
    if not st.session_state.aim_game_started and not st.session_state.aim_game_over:
        st.subheader("Instructions:")
        st.markdown("""
        Click 'Start Game' to begin. Targets will appear and shrink.
        Click the 'Shoot!' button to try and hit a target.
        Accuracy and speed are tracked. Don't let too many targets disappear!
        """)
        if st.button("Start Game", key="aim_start_game_btn"):
            st.session_state.aim_game_started = True
            st.session_state.aim_start_time = time.time()
            st.session_state.aim_last_target_spawn_time = time.time()
            st.rerun() # Rerun to start the game loop

    if st.session_state.aim_game_over:
        display_end_screen(
            time.time() - st.session_state.aim_start_time,
            st.session_state.aim_targets_pressed,
            st.session_state.aim_clicks
        )
        return # Stop execution here if game is over

    if st.session_state.aim_game_started:
        # Placeholder for the game display
        game_display_placeholder = st.empty()
        
        # --- Game Loop Simulation ---
        current_time = time.time()
        elapsed_game_time = current_time - st.session_state.aim_start_time
        
        # Check for game end by time
        if elapsed_game_time >= GAME_DURATION:
            st.session_state.aim_game_over = True
            st.rerun()
            return

        # Spawn new target
        if current_time - st.session_state.aim_last_target_spawn_time >= TARGET_SPAWN_INTERVAL:
            x = random.randint(TARGET_PADDING, WIDTH - TARGET_PADDING)
            y = random.randint(TOP_BAR_HEIGHT + TARGET_PADDING, HEIGHT - TARGET_PADDING)
            st.session_state.aim_targets.append(Target(x, y))
            st.session_state.aim_last_target_spawn_time = current_time

        # Update targets
        targets_to_keep = []
        delta_time = 1 / 60.0 # Simulate 60 FPS update for growth/shrink rate
        for target in st.session_state.aim_targets:
            if target.update(delta_time): # If target is still alive
                targets_to_keep.append(target)
            else: # Target disappeared (missed)
                if not target.hit: # Only count as miss if not hit
                    st.session_state.aim_misses += 1
        st.session_state.aim_targets = targets_to_keep

        # Check for game end by lives
        if st.session_state.aim_misses >= LIVES:
            st.session_state.aim_game_over = True
            st.rerun()
            return

        # Create and display the game image
        game_image_bytes = create_game_image(
            st.session_state.aim_targets,
            elapsed_game_time,
            st.session_state.aim_targets_pressed,
            st.session_state.aim_misses
        )
        game_display_placeholder.image(game_image_bytes, use_column_width=True)

        # --- User Interaction ---
        # We need a way to register a "click"
        # Since we can't get real-time mouse position on the image in Streamlit easily,
        # we'll simplify: a "Shoot!" button will try to hit the *first* visible target.
        # This is a major deviation from Pygame but necessary for web.
        if st.button("Shoot!", key="aim_shoot_btn"):
            st.session_state.aim_clicks += 1
            if st.session_state.aim_targets:
                # Try to hit the first target in the list
                target_to_hit = st.session_state.aim_targets[0]
                # Simulate a hit by marking it as hit, it will shrink rapidly
                target_to_hit.hit = True
                st.session_state.aim_targets_pressed += 1
                st.toast("🎯 Hit!")
            else:
                st.toast("Miss! No target to shoot.")
            st.rerun() # Rerun to update display after shot

        # --- Auto-rerun to simulate game loop ---
        # This is the trickiest part for real-time. We need to force Streamlit to rerun
        # periodically to update the target sizes and spawn new ones.
        # This is not a true game loop, but a series of rapid reruns.
        time.sleep(0.05) # Small delay to prevent excessive CPU usage
        st.rerun() # Force a rerun to update the game state and display

# Run the app
run_aim_trainer_app()