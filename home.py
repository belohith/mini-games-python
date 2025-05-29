import streamlit as st

st.set_page_config(
    page_title="My Python Projects Dashboard",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 Welcome to My Python Projects Dashboard!")

st.markdown("""
This is a collection of my Python projects, brought together into a single Streamlit web application.
Use the sidebar to navigate between different projects.
""")

st.subheader("Available Projects:")
st.markdown("- **Sudoku Solver:** An interactive tool to solve Sudoku puzzles step-by-step.")
st.markdown("- **Maze Pathfinder:** Visualize various algorithms finding paths through a maze.")
st.markdown("- **Mastermind Game:** Test your logic and deduction skills.")

st.info("""
**How to use:**
1. Select a project from the sidebar on the left.
2. Follow the instructions on each project's page.
""")

st.markdown("---")

st.subheader("Important Notes & Disclaimers:")

st.warning("""
**Performance & Responsiveness:**
Please note that these are simple projects running in a web environment.
Complex animations or highly interactive elements (like the Aim Trainer game we discussed)
might not feel as smooth or responsive as a native desktop application.
This is a limitation of web frameworks designed for data applications.
""")

st.info("""
**Browser Compatibility:**
While Streamlit aims for broad browser compatibility, optimal performance and visual fidelity
may vary slightly across different web browsers and devices.
""")

st.write("Built with ❤️ and Streamlit.")