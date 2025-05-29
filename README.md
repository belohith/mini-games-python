# 🎮 My Python Projects Dashboard

Welcome to my personal collection of Python projects, showcased as an interactive web application built with Streamlit! This repository brings together various coding challenges and games into a single, easy-to-navigate dashboard.

## ✨ Features

* **Sudoku Solver:** An interactive tool that allows you to input Sudoku puzzles and watch various algorithms solve them step-by-step.
* **Maze Pathfinder:** Visualize classic pathfinding algorithms (BFS, DFS, A*, Dijkstra's, Greedy Best-First Search, and Wall Follower) as they explore and find paths through randomly generated mazes.
* **Mastermind Game:** A classic code-breaking game where you deduce a secret code using logic and feedback.

## 🚀 How to Use

### 🌐 Access Online (Recommended)

The easiest way to experience these projects is to visit the live Streamlit application:

**[Click here to open the Projects Dashboard!](https://mini-games-python.streamlit.app/)**

### 💻 Run Locally

If you prefer to run the application on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [hhttps://github.com/belohith/mini-games-python](https://github.com/belohith/mini-games-python)
    cd YOUR_REPO_NAME
    ```
    

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run Home.py
    ```
    (Assuming your main dashboard file is `Home.py`. If it's `my_games_dashboard.py` or similar, adjust the command accordingly.)

    This will open the application in your default web browser.

## 🚧 Important Notes & Disclaimers

* **Performance & Responsiveness:**
    Please note that these projects are designed to run in a web environment using Streamlit. While efforts have been made to optimize them, complex animations or highly interactive elements (especially for game-like simulations) might not feel as smooth or responsive as a native desktop application. This is a general characteristic of web frameworks designed primarily for data applications rather than high-performance gaming.

* **Browser Compatibility:**
    While Streamlit aims for broad browser compatibility, optimal performance and visual fidelity may vary slightly across different web browsers and devices. It's recommended to use modern browsers like Google Chrome, Mozilla Firefox, or Microsoft Edge for the best experience.

## 🛠️ Technologies Used

* **Python:** The core programming language for all projects.
* **Streamlit:** For building the interactive web dashboard.
* **Matplotlib:** For visualizing the Maze Pathfinder algorithm steps.
* **NumPy:** For numerical operations, especially in maze generation and Sudoku.

## ✍️ Author

Lohith Bollineni

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---
