import streamlit as st
import random

# --- Mastermind Game Configuration ---
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

# Generate the secret code (cached so it doesn't change on rerun unless explicitly reset)
@st.cache_resource(show_spinner=False)
def generate_code():
    return [random.choice(COLORS) for _ in range(CODE_LENGTH)]

def check_code(guess, real_code):
    color_counts = {}
    correct_pos = 0
    incorrect_pos = 0
    # Create a copy of the real_code to mark checked elements for incorrect_pos
    temp_real_code = list(real_code)
    
    # First pass: Check for correct position (red pegs)
    # Use a list to store checked indices for guess to avoid double counting
    guess_checked = [False] * CODE_LENGTH

    for i in range(CODE_LENGTH):
        if guess[i] == temp_real_code[i]:
            correct_pos += 1
            temp_real_code[i] = None # Mark as used in real code
            guess_checked[i] = True # Mark as used in guess

    # Second pass: Check for incorrect position (white pegs)
    for i in range(CODE_LENGTH):
        if guess_checked[i]: # Skip if already matched for correct position
            continue
        if guess[i] in temp_real_code:
            incorrect_pos += 1
            # Mark the first occurrence of the color in temp_real_code as used
            temp_real_code[temp_real_code.index(guess[i])] = None 

    return correct_pos, incorrect_pos

# --- Streamlit Game Flow ---

def game():
    # Initialize game state in session_state
    if 'mastermind_code' not in st.session_state:
        st.session_state.mastermind_code = generate_code()
        st.session_state.mastermind_attempts = 0
        st.session_state.mastermind_history = []
        st.session_state.mastermind_game_over = False
        st.session_state.last_guess_feedback = "" # NEW: To hold feedback for the immediate previous guess

    code = st.session_state.mastermind_code

    # Display guess history (always visible)
    if st.session_state.mastermind_history:
        st.subheader("Your Guesses:")
        for i, (g, cp, ip) in enumerate(st.session_state.mastermind_history):
            st.write(f"Attempt {i+1}: **{' '.join(g)}** | Correct positions: {cp}, Incorrect positions: {ip}")
        st.markdown("---")

    # Display the result of the *last* non-game-ending guess
    # This ensures feedback appears before input fields if the game is ongoing
    if st.session_state.last_guess_feedback and not st.session_state.mastermind_game_over:
        st.info(st.session_state.last_guess_feedback)
        # Clear it immediately after display, so it doesn't show again on subsequent non-guess reruns
        st.session_state.last_guess_feedback = "" 

    # Main game play section (visible only if game is NOT over)
    if not st.session_state.mastermind_game_over:
        st.write(f"You have **{TRIES - st.session_state.mastermind_attempts}** attempts remaining.")
        st.write(f"Colors available: **{' '.join(COLORS)}**")

        # User input for guess
        cols = st.columns(CODE_LENGTH)
        guess_input = []
        for i, col in enumerate(cols):
            # Key changes with attempt number to force re-render of selectboxes
            guess_input.append(col.selectbox(f"Pos {i+1}", COLORS, key=f"guess_color_{i}_{st.session_state.mastermind_attempts}"))

        if st.button("Submit Guess"):
            st.session_state.mastermind_attempts += 1
            correct_pos, incorrect_pos = check_code(guess_input, code)
            st.session_state.mastermind_history.append((guess_input, correct_pos, incorrect_pos))

            if correct_pos == CODE_LENGTH:
                st.session_state.last_guess_feedback = f"🎉 Congratulations! You've guessed the code **{' '.join(code)}** in {st.session_state.mastermind_attempts} attempts."
                st.session_state.mastermind_game_over = True
                st.balloons()
            elif st.session_state.mastermind_attempts >= TRIES:
                st.session_state.last_guess_feedback = f"😔 Sorry, you've used all attempts. The correct code was **{' '.join(code)}**."
                st.session_state.mastermind_game_over = True
            else:
                # Store feedback for the *next* rerun when the page updates
                st.session_state.last_guess_feedback = f"Guess: **{' '.join(guess_input)}** | Correct positions: {correct_pos}, Incorrect positions: {incorrect_pos}"
            
            st.rerun() # Rerun the app to update the display based on new state

    # Game Over screen (visible only if game IS over)
    else:
        st.subheader("Game Over!")
        if st.session_state.last_guess_feedback: # Display final win/lose message directly
            st.info(st.session_state.last_guess_feedback)
            st.session_state.last_guess_feedback = "" # Clear after displaying
        
        if st.button("Play Again?", key="play_again_btn"):
            # Reset all relevant game state variables
            del st.session_state.mastermind_code
            st.session_state.mastermind_game_over = False
            st.session_state.mastermind_attempts = 0
            st.session_state.mastermind_history = []
            st.session_state.last_guess_feedback = "" 
            st.rerun() # Rerun to start a fresh game

# Run the game function
game()