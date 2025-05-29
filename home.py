import streamlit as st

st.set_page_config(
    page_title="My Python Projects Dashboard",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 Welcome to My Python Projects Dashboard!")

st.markdown("""
This is a collection of my simple Python projects, brought together into a single Streamlit web application.
Use the sidebar to navigate between different projects.
""")

st.subheader("Available Projects:")
st.markdown("- **Mastermind Game:** Test your logic and deduction skills.")
st.markdown("- *More projects coming soon!*")

st.info("""
**How to use:**
1. Select a project from the sidebar on the left.
2. Follow the instructions on each project's page.
""")

st.markdown("---")
st.write("Built with ❤️ and Streamlit.")