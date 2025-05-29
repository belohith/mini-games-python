import streamlit as st
import random

# --- Mastermind Game Configuration (can be put in Streamlit if you want users to change) ---
COLORS = ['R', 'G', 'B', 'Y', 'O', 'W']
TRIES = 10
CODE_LENGTH = 4

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Mastermind Game", page_icon="💡")
st.title("💡 Mastermind Game")
st.markdown("""
Guess the secret code! You have 10 attempts to guess the 4-color code.
Colors can be repeated.
""")

# --- Game Logic Functions (Adapted for Streamlit) ---

# Generate the secret code (cached so it doesn't change on rerun)
@st.cache_resource(show_spinner=False)
def generate_code():
    return [random.choice(COLORS) for _ in range(CODE_LENGTH)]

def check_code(guess, real_code):
    color_counts = {}
    correct_pos = 0
    incorrect_pos = 0
    checked_indices = [] # To prevent double counting a color for correct_pos and incorrect_pos

    # Count occurrences of each color in the real code
    for color in real_code:
        color_counts[color] = color_counts.get(color, 0) + 1

    # First pass: Check for correct position (red pegs)
    for i in range(len(guess)):
        if guess[i] == real_code[i]:
            correct_pos += 1
            color_counts[guess[i]] -= 1 # Consume the color from counts
            checked_indices.append(i) # Mark this index as checked

    # Second pass: Check for incorrect position (white pegs)
    for i in range(len(guess)):
        if i in checked_indices: # Skip colors already matched for correct position
            continue
        if guess[i] in color_counts and color_counts[guess[i]] > 0:
            incorrect_pos += 1
            color_counts[guess[i]] -= 1 # Consume the color

    return correct_pos, incorrect_pos

# --- Streamlit Game Flow ---

def game():
    # Initialize game state in session_state
    if 'mastermind_code' not in st.session_state:
        st.session_state.mastermind_code = generate_code()
        st.session_state.mastermind_attempts = 0
        st.session_state.mastermind_history = []
        st.session_state.mastermind_game_over = False

    code = st.session_state.mastermind_code

    if not st.session_state.mastermind_game_over:
        st.write(f"You have **{TRIES - st.session_state.mastermind_attempts}** attempts remaining.")
        st.write(f"Colors available: **{' '.join(COLORS)}**")

        # Display guess history
        if st.session_state.mastermind_history:
            st.subheader("Your Guesses:")
            for i, (g, cp, ip) in enumerate(st.session_state.mastermind_history):
                st.write(f"Attempt {i+1}: **{' '.join(g)}** | Correct positions: {cp}, Incorrect positions: {ip}")
            st.markdown("---")

        # User input for guess
        cols = st.columns(CODE_LENGTH)
        guess_input = []
        for i, col in enumerate(cols):
            # Using selectbox for color input
            guess_input.append(col.selectbox(f"Pos {i+1}", COLORS, key=f"guess_color_{i}"))

        if st.button("Submit Guess"):
            st.session_state.mastermind_attempts += 1
            correct_pos, incorrect_pos = check_code(guess_input, code)
            st.session_state.mastermind_history.append((guess_input, correct_pos, incorrect_pos))

            if correct_pos == CODE_LENGTH:
                st.success(f"🎉 Congratulations! You've guessed the code **{' '.join(code)}** in {st.session_state.mastermind_attempts} attempts.")
                st.session_state.mastermind_game_over = True
                st.balloons()
            elif st.session_state.mastermind_attempts >= TRIES:
                st.error(f"😔 Sorry, you've used all attempts. The correct code was **{' '.join(code)}**.")
                st.session_state.mastermind_game_over = True
            else:
                st.info(f"Guess: **{' '.join(guess_input)}** | Correct positions: {correct_pos}, Incorrect positions: {incorrect_pos}")
            
            # This causes a rerun to update the display
            st.rerun()
    else:
        st.subheader("Game Over!")
        if st.button("Play Again?"):
            # Reset game state
            del st.session_state.mastermind_code
            st.session_state.mastermind_game_over = False
            st.session_state.mastermind_attempts = 0
            st.session_state.mastermind_history = []
            st.rerun() # Rerun to start a new game

# Run the game
game()