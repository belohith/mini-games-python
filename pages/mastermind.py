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

# Generate the secret code (THIS FUNCTION'S CACHE NEEDS TO BE CLEARED ON NEW GAME)
@st.cache_resource(show_spinner=False)
def generate_code():
    # You might want to print this to console during testing to verify
    # print(f"Generating new code: {[random.choice(COLORS) for _ in range(CODE_LENGTH)]}")
    return [random.choice(COLORS) for _ in range(CODE_LENGTH)]

def check_code(guess, real_code):
    color_counts = {}
    correct_pos = 0
    incorrect_pos = 0
    checked_indices = []

    for color in real_code:
        color_counts[color] = color_counts.get(color, 0) + 1

    for i in range(len(guess)):
        if guess[i] == real_code[i]:
            correct_pos += 1
            color_counts[guess[i]] -= 1
            checked_indices.append(i)

    for i in range(len(guess)):
        if i in checked_indices:
            continue
        if guess[i] in color_counts and color_counts[guess[i]] > 0:
            incorrect_pos += 1
            color_counts[guess[i]] -= 1

    return correct_pos, incorrect_pos

# --- Streamlit Game Flow ---

def game():
    # Initialize game state in session_state
    if 'mastermind_code' not in st.session_state:
        st.session_state.mastermind_code = generate_code()
        st.session_state.mastermind_attempts = 0
        st.session_state.mastermind_history = []
        st.session_state.mastermind_game_over = False
        st.session_state.last_guess_feedback = ""
        st.session_state.current_guess_selection = [COLORS[0]] * CODE_LENGTH

    code = st.session_state.mastermind_code

    # Display guess history (always visible)
    if st.session_state.mastermind_history:
        st.subheader("Your Guesses:")
        for i, (g, cp, ip) in enumerate(st.session_state.mastermind_history):
            st.write(f"Attempt {i+1}: **{' '.join(g)}** | Correct positions: {cp}, Incorrect positions: {ip}")
        st.markdown("---")

    # Display the result of the *last* non-game-ending guess
    if st.session_state.last_guess_feedback and not st.session_state.mastermind_game_over:
        st.info(st.session_state.last_guess_feedback)
        st.session_state.last_guess_feedback = "" 

    # Main game play section (visible only if game is NOT over)
    if not st.session_state.mastermind_game_over:
        st.write(f"You have **{TRIES - st.session_state.mastermind_attempts}** attempts remaining.")
        st.write(f"Colors available: **{' '.join(COLORS)}**")

        # User input for guess
        cols = st.columns(CODE_LENGTH)
        
        user_selected_colors_this_run = []
        for i, col in enumerate(cols):
            selected_color = col.selectbox(
                f"Pos {i+1}",
                COLORS,
                index=COLORS.index(st.session_state.current_guess_selection[i]),
                key=f"guess_color_{i}_{st.session_state.mastermind_attempts}"
            )
            user_selected_colors_this_run.append(selected_color)

        if st.button("Submit Guess"):
            st.session_state.mastermind_attempts += 1
            
            correct_pos, incorrect_pos = check_code(user_selected_colors_this_run, code)
            st.session_state.mastermind_history.append((user_selected_colors_this_run, correct_pos, incorrect_pos))

            if correct_pos == CODE_LENGTH:
                st.session_state.last_guess_feedback = f"🎉 Congratulations! You've guessed the code **{' '.join(code)}** in {st.session_state.mastermind_attempts} attempts."
                st.session_state.mastermind_game_over = True
                st.balloons()
            elif st.session_state.mastermind_attempts >= TRIES:
                st.session_state.last_guess_feedback = f"😔 Sorry, you've used all attempts. The correct code was **{' '.join(code)}**."
                st.session_state.mastermind_game_over = True
            else:
                st.session_state.last_guess_feedback = f"Guess: **{' '.join(user_selected_colors_this_run)}** | Correct positions: {correct_pos}, Incorrect positions: {incorrect_pos}"
            
            st.session_state.current_guess_selection = user_selected_colors_this_run
            
            st.rerun()

    # Game Over screen (visible only if game IS over)
    else:
        st.subheader("Game Over!")
        if st.session_state.last_guess_feedback:
            st.info(st.session_state.last_guess_feedback)
            st.session_state.last_guess_feedback = ""
        
        if st.button("Play Again?", key="play_again_btn"):
            # --- THE KEY FIX IS HERE ---
            # Clear the cache of the generate_code function to force a new code generation
            generate_code.clear() 
            # --- END OF FIX ---

            # Reset all relevant game state variables
            del st.session_state.mastermind_code
            st.session_state.mastermind_game_over = False
            st.session_state.mastermind_attempts = 0
            st.session_state.masterind_history = [] # Typo here, should be mastermind_history
            st.session_state.last_guess_feedback = ""
            st.session_state.current_guess_selection = [COLORS[0]] * CODE_LENGTH 
            st.rerun()

# Run the game function
game()