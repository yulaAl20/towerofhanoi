import streamlit as st
import time
import random
import sqlite3
import pandas as pd
from datetime import datetime

# Initialize the database
def init_db():
    conn = sqlite3.connect('tower_of_hanoi.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS game_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_name TEXT,
        disk_count INTEGER,
        moves_count INTEGER,
        move_sequence TEXT,
        algorithm TEXT,
        execution_time REAL,
        timestamp TEXT
    )
    ''')
    conn.commit()
    conn.close()

# Save game results to the database
def save_result(player_name, disk_count, moves_count, move_sequence, algorithm, execution_time):
    conn = sqlite3.connect('tower_of_hanoi.db')
    c = conn.cursor()
    c.execute('''
    INSERT INTO game_results (player_name, disk_count, moves_count, move_sequence, algorithm, execution_time, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (player_name, disk_count, moves_count, move_sequence, algorithm, execution_time, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

# Get leaderboard data
def get_leaderboard():
    conn = sqlite3.connect('tower_of_hanoi.db')
    df = pd.read_sql_query('''
    SELECT player_name, disk_count, moves_count, algorithm, execution_time 
    FROM game_results 
    ORDER BY execution_time ASC
    LIMIT 10
    ''', conn)
    conn.close()
    return df

# Classic 3-peg Tower of Hanoi recursive solution
def solve_hanoi_recursive(n, source, auxiliary, destination):
    moves = []
    
    def hanoi(n, source, auxiliary, destination):
        if n == 1:
            moves.append(f"{source}->{destination}")
            return
        hanoi(n-1, source, destination, auxiliary)
        moves.append(f"{source}->{destination}")
        hanoi(n-1, auxiliary, source, destination)
    
    start_time = time.time()
    hanoi(n, source, auxiliary, destination)
    end_time = time.time()
    
    return moves, end_time - start_time

# Classic 3-peg Tower of Hanoi iterative solution
def solve_hanoi_iterative(n, source, auxiliary, destination):
    moves = []
    start_time = time.time()
    
    # If n is even, swap auxiliary and destination
    if n % 2 == 0:
        auxiliary, destination = destination, auxiliary
    
    total_moves = (1 << n) - 1  # 2^n - 1
    
    for i in range(1, total_moves + 1):
        if i % 3 == 1:
            # Move between source and destination
            if not moves or moves[-1].startswith(destination) and moves[-1].endswith(source):
                moves.append(f"{source}->{destination}")
            else:
                moves.append(f"{destination}->{source}")
        elif i % 3 == 2:
            # Move between source and auxiliary
            if not moves or moves[-1].startswith(auxiliary) and moves[-1].endswith(source):
                moves.append(f"{source}->{auxiliary}")
            else:
                moves.append(f"{auxiliary}->{source}")
        else:
            # Move between auxiliary and destination
            if not moves or moves[-1].startswith(destination) and moves[-1].endswith(auxiliary):
                moves.append(f"{auxiliary}->{destination}")
            else:
                moves.append(f"{destination}->{auxiliary}")
    
    end_time = time.time()
    return moves, end_time - start_time

# Frame-Stewart algorithm for 4 pegs
def solve_frame_stewart(n, source, aux1, aux2, destination):
    moves = []
    start_time = time.time()
    
    # Calculate k (optimal split for Frame-Stewart)
    k = int(n - (2*n)**(1/2))
    if k < 1:
        k = 1
    
    def frame_stewart_helper(n, source, aux1, aux2, destination):
        if n == 0:
            return
        if n == 1:
            moves.append(f"{source}->{destination}")
            return
        
        # Calculate k for this recursion level
        k = int(n - (2*n)**(1/2))
        if k < 1:
            k = 1
        
        # Move top k disks to aux1
        frame_stewart_helper(k, source, destination, aux2, aux1)
        # Move remaining n-k disks from source to destination using 3 pegs
        three_peg_hanoi(n-k, source, aux1, aux2, destination)
        # Move k disks from aux1 to destination
        frame_stewart_helper(k, aux1, source, aux2, destination)
    
    def three_peg_hanoi(n, source, auxiliary, not_used, destination):
        if n == 0:
            return
        if n == 1:
            moves.append(f"{source}->{destination}")
            return
        three_peg_hanoi(n-1, source, not_used, auxiliary, auxiliary)
        moves.append(f"{source}->{destination}")
        three_peg_hanoi(n-1, auxiliary, source, not_used, destination)
    
    frame_stewart_helper(n, source, aux1, aux2, destination)
    end_time = time.time()
    
    return moves, end_time - start_time

# Initialize game state
def init_game_state(n):
    return {
        'A': list(range(n, 0, -1)),
        'B': [],
        'C': [],
        'D': []
    }

# Validate a move
def is_valid_move(state, source, destination):
    if not state[source]:
        return False
    if not state[destination]:
        return True
    return state[source][-1] < state[destination][-1]

# Apply a move
def apply_move(state, source, destination):
    if is_valid_move(state, source, destination):
        disk = state[source].pop()
        state[destination].append(disk)
        return True
    return False

# Check if the game is solved
def is_solved(state, n, destination='C'):
    return len(state[destination]) == n and sorted(state[destination], reverse=True) == state[destination]

# Render the Tower of Hanoi game board
def render_game_board(state, n, pegs=3):
    max_disk_width = 180  # Max pixel width for the largest disk
    base_width = max_disk_width + 60
    
    # Create CSS for the game board
    st.markdown("""
<style>
    body {
    background-color: #1e1e1e;
}

.game-board {
    display: flex;
    justify-content: center;
    align-items: flex-end;
    padding: 20px 0;
    background-color: #1e1e1e;
}

.tower {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin: 0 30px;
    position: relative;
}

.peg {
    width: 10px;
    background: linear-gradient(180deg, #ff4e50, #6b5b95);
    border-radius: 5px;
    margin-bottom: 10px;
}

.disk {
    border-radius: 20px;
    text-align: center;
    color: white;
    font-weight: bold;
    height: 28px;
    line-height: 28px;
    box-shadow: 0 0 6px rgba(0,0,0,0.5);
    margin: 5px 0;
    transition: all 0.3s ease;
}

.base {
    background: linear-gradient(90deg, #ff4e50, #6b5b95);
    height: 12px;
    border-radius: 6px;
    margin-top: 10px;
}

.tower-label {
    color: white;
    font-size: 20px;
    margin-top: 10px;
}
    </style>
""", unsafe_allow_html=True)

    
    # Draw the board
    cols = st.columns(pegs)
    peg_names = ['A', 'B', 'C', 'D'][:pegs]
    
    for i, peg in enumerate(peg_names):
        with cols[i]:
            tower_html = f'<div class="tower" id="peg-{peg}">'
            
            # Calculate peg height based on number of disks
            peg_height = (n * 30) + 20
            tower_html += f'<div class="peg" style="height: {peg_height}px;"></div>'
            
            # Add disks
            for disk in reversed(state[peg]):
                # Calculate disk width proportional to its size
                disk_width = 40 + ((disk / n) * max_disk_width)
                # Generate a color based on disk size
                hue = int(120 + (240 * (disk / n)))
                tower_html += f'<div class="disk dragdrop" id="disk-{disk}" style="width: {disk_width+20}px; background-color: hsl({hue}, 70%, 50%);">{disk}</div>'
            
            # Add base
            tower_html += f'<div class="base" style="width: {base_width}px;"></div>'
            tower_html += f'<div style="text-align: center; margin-top: 10px;"><h2>{peg}</h2></div>'
            tower_html += '</div>'
            
            st.markdown(tower_html, unsafe_allow_html=True)
    
    # Add JavaScript for drag and drop
    drag_drop_js = """
    
    """
    st.markdown(drag_drop_js, unsafe_allow_html=True)

# Main application
def main():
    st.set_page_config(page_title="Tower of Hanoi Game", layout="wide")
    
    # Initialize database
    init_db()
    
    # App title
    st.title("Tower of Hanoi Game")
    
    # Sidebar for game options
    st.sidebar.header("Game Options")
    
    # Initialize session state
    if 'game_active' not in st.session_state:
        st.session_state.game_active = False
    if 'disk_count' not in st.session_state:
        st.session_state.disk_count = 0
    if 'game_state' not in st.session_state:
        st.session_state.game_state = {}
    if 'move_count' not in st.session_state:
        st.session_state.move_count = 0
    if 'moves_made' not in st.session_state:
        st.session_state.moves_made = []
    if 'optimal_moves' not in st.session_state:
        st.session_state.optimal_moves = []
    if 'peg_count' not in st.session_state:
        st.session_state.peg_count = 3
    
    # Menu options
    menu = st.sidebar.selectbox("Menu", ["Play Tower of Hanoi", "Leaderboard", "Algorithm Comparison"])
    
    if menu == "Play Tower of Hanoi":
        st.header("Play Tower of Hanoi")
        
        # Game setup
        col1, col2, col3 = st.columns(3)
        
        with col1:
            player_name = st.text_input("Your Name", key="player_name")
        
        with col2:
            peg_count = st.radio("Number of Pegs", [3, 4], key="peg_selection")
        
        with col3:
            if st.button("Start New Game"):
                # Generate random disk count between 5 and 10
                disk_count = random.randint(5, 10)
                st.session_state.disk_count = disk_count
                st.session_state.peg_count = peg_count
                st.session_state.game_state = init_game_state(disk_count)
                st.session_state.game_active = True
                st.session_state.move_count = 0
                st.session_state.moves_made = []
                
                # Calculate optimal moves
                if peg_count == 3:
                    st.session_state.optimal_moves, _ = solve_hanoi_recursive(disk_count, 'A', 'B', 'C')
                else:
                    st.session_state.optimal_moves, _ = solve_frame_stewart(disk_count, 'A', 'B', 'C', 'D')
                
                st.success(f"Started a new game with {disk_count} disks and {peg_count} pegs!")
        
        if st.session_state.game_active:
            st.write(f"Current game: {st.session_state.disk_count} disks with {st.session_state.peg_count} pegs")
            st.write(f"Minimum moves required: {len(st.session_state.optimal_moves)}")
            st.write(f"Moves made so far: {st.session_state.move_count}")
            
            # Display the game board
            render_game_board(st.session_state.game_state, st.session_state.disk_count, st.session_state.peg_count)
            
            # Move input
            st.subheader("Make a Move")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                source = st.selectbox("From Peg", ['A', 'B', 'C', 'D'][:st.session_state.peg_count])
            
            with col2:
                destination = st.selectbox("To Peg", ['A', 'B', 'C', 'D'][:st.session_state.peg_count])
            
            with col3:
                if st.button("Make Move"):
                    if source == destination:
                        st.error("Source and destination pegs cannot be the same!")
                    elif apply_move(st.session_state.game_state, source, destination):
                        st.session_state.move_count += 1
                        st.session_state.moves_made.append(f"{source}->{destination}")
                        
                        # Check if the game is solved
                        if is_solved(st.session_state.game_state, st.session_state.disk_count):
                            st.balloons()
                            st.success(f"Congratulations! You solved the puzzle in {st.session_state.move_count} moves!")
                            
                            # Compare with algorithms
                            recursive_moves, recursive_time = solve_hanoi_recursive(
                                st.session_state.disk_count, 'A', 'B', 'C')
                            
                            iterative_moves, iterative_time = solve_hanoi_iterative(
                                st.session_state.disk_count, 'A', 'B', 'C')
                            
                            if st.session_state.peg_count == 4:
                                fs_moves, fs_time = solve_frame_stewart(
                                    st.session_state.disk_count, 'A', 'B', 'C', 'D')
                                
                                st.write(f"Frame-Stewart algorithm (4 pegs) solved it in {len(fs_moves)} moves in {fs_time:.6f} seconds")
                                
                                # Save results
                                save_result(player_name, st.session_state.disk_count, st.session_state.move_count, 
                                           ",".join(st.session_state.moves_made), "Player Solution (4 pegs)", 0)
                                save_result("Algorithm", st.session_state.disk_count, len(fs_moves), 
                                           ",".join(fs_moves), "Frame-Stewart (4 pegs)", fs_time)
                            
                            st.write(f"Recursive algorithm solved it in {len(recursive_moves)} moves in {recursive_time:.6f} seconds")
                            st.write(f"Iterative algorithm solved it in {len(iterative_moves)} moves in {iterative_time:.6f} seconds")
                            
                            # Save results
                            save_result(player_name, st.session_state.disk_count, st.session_state.move_count, 
                                       ",".join(st.session_state.moves_made), "Player Solution (3 pegs)", 0)
                            save_result("Algorithm", st.session_state.disk_count, len(recursive_moves), 
                                       ",".join(recursive_moves), "Recursive", recursive_time)
                            save_result("Algorithm", st.session_state.disk_count, len(iterative_moves), 
                                       ",".join(iterative_moves), "Iterative", iterative_time)
                            
                            # Reset game
                            st.session_state.game_active = False
                        else:
                            st.success("Move successful!")
                    else:
                        st.error("Invalid move! Remember, you cannot place a larger disk on a smaller one.")
            
            # Option to enter full move sequence
            st.subheader("Enter Full Solution")
            col1, col2 = st.columns(2)
            
            with col1:
                move_count = st.number_input("Number of Moves", min_value=1, value=len(st.session_state.optimal_moves))
            
            with col2:
                move_sequence = st.text_input("Move Sequence (e.g., A->B,B->C,A->C)")
            
            if st.button("Submit Solution"):
                moves = move_sequence.split(',')
                if len(moves) != move_count:
                    st.error(f"You specified {move_count} moves but provided {len(moves)} moves!")
                else:
                    # Reset game state and apply moves
                    test_state = init_game_state(st.session_state.disk_count)
                    valid_solution = True
                    
                    for move in moves:
                        if '->' not in move:
                            st.error(f"Invalid move format: {move}. Use 'Source->Destination' format.")
                            valid_solution = False
                            break
                        
                        source, destination = move.split('->')
                        if not apply_move(test_state, source, destination):
                            st.error(f"Invalid move: {move}. Check your solution.")
                            valid_solution = False
                            break
                    
                    if valid_solution:
                        if is_solved(test_state, st.session_state.disk_count):
                            st.balloons()
                            st.success("Your solution is correct!")
                            
                            # Compare with algorithms
                            recursive_moves, recursive_time = solve_hanoi_recursive(
                                st.session_state.disk_count, 'A', 'B', 'C')
                            
                            iterative_moves, iterative_time = solve_hanoi_iterative(
                                st.session_state.disk_count, 'A', 'B', 'C')
                            
                            if st.session_state.peg_count == 4:
                                fs_moves, fs_time = solve_frame_stewart(
                                    st.session_state.disk_count, 'A', 'B', 'C', 'D')
                                
                                st.write(f"Frame-Stewart algorithm (4 pegs) solved it in {len(fs_moves)} moves in {fs_time:.6f} seconds")
                                
                                # Save results
                                save_result(player_name, st.session_state.disk_count, len(moves), 
                                           move_sequence, "Player Solution (4 pegs)", 0)
                                save_result("Algorithm", st.session_state.disk_count, len(fs_moves), 
                                           ",".join(fs_moves), "Frame-Stewart (4 pegs)", fs_time)
                            
                            st.write(f"Recursive algorithm solved it in {len(recursive_moves)} moves in {recursive_time:.6f} seconds")
                            st.write(f"Iterative algorithm solved it in {len(iterative_moves)} moves in {iterative_time:.6f} seconds")
                            
                            # Save results
                            save_result(player_name, st.session_state.disk_count, len(moves), 
                                       move_sequence, "Player Solution (3 pegs)", 0)
                            save_result("Algorithm", st.session_state.disk_count, len(recursive_moves), 
                                       ",".join(recursive_moves), "Recursive", recursive_time)
                            save_result("Algorithm", st.session_state.disk_count, len(iterative_moves), 
                                       ",".join(iterative_moves), "Iterative", iterative_time)
                            
                            # Reset game
                            st.session_state.game_active = False
                        else:
                            st.error("Your solution does not solve the puzzle!")
            
            # Get a hint
            if st.button("Get Hint"):
                if st.session_state.move_count < len(st.session_state.optimal_moves):
                    hint = st.session_state.optimal_moves[st.session_state.move_count]
                    st.info(f"Hint: Try moving from {hint.split('->')[0]} to {hint.split('->')[1]}")
                else:
                    st.info("You've already made more moves than the optimal solution!")
        
    elif menu == "Leaderboard":
        st.header("Leaderboard")
        leaderboard = get_leaderboard()
        st.dataframe(leaderboard)
    
    elif menu == "Algorithm Comparison":
        st.header("Algorithm Comparison")
        
        # Compare algorithms for different disk counts
        st.subheader("Comparison by Disk Count")
        
        disk_count = st.slider("Number of Disks", min_value=3, max_value=20, value=10)
        
        if st.button("Run Comparison"):
            st.write("Running comparison...")
            
            # 3 pegs
            recursive_moves, recursive_time = solve_hanoi_recursive(disk_count, 'A', 'B', 'C')
            iterative_moves, iterative_time = solve_hanoi_iterative(disk_count, 'A', 'B', 'C')
            
            # 4 pegs
            fs_moves, fs_time = solve_frame_stewart(disk_count, 'A', 'B', 'C', 'D')
            
            # Results
            data = {
                'Algorithm': ['Recursive (3 pegs)', 'Iterative (3 pegs)', 'Frame-Stewart (4 pegs)'],
                'Move Count': [len(recursive_moves), len(iterative_moves), len(fs_moves)],
                'Execution Time (s)': [recursive_time, iterative_time, fs_time]
            }
            
            df = pd.DataFrame(data)
            st.dataframe(df)
            
            # Visualization
            st.subheader("Move Count Comparison")
            st.bar_chart(df.set_index('Algorithm')['Move Count'])
            
            st.subheader("Execution Time Comparison")
            st.bar_chart(df.set_index('Algorithm')['Execution Time (s)'])


# Unit tests
class TestTowerOfHanoi:
    def test_recursive_solution(self):
        """Test that the recursive solution produces the correct number of moves."""
        for n in range(1, 10):
            moves, _ = solve_hanoi_recursive(n, 'A', 'B', 'C')
            assert len(moves) == (2**n - 1), f"Expected {2**n - 1} moves for {n} disks, got {len(moves)}"
    
    def test_iterative_solution(self):
        """Test that the iterative solution produces the correct number of moves."""
        for n in range(1, 10):
            moves, _ = solve_hanoi_iterative(n, 'A', 'B', 'C')
            assert len(moves) == (2**n - 1), f"Expected {2**n - 1} moves for {n} disks, got {len(moves)}"
    
    def test_frame_stewart(self):
        """Test that the Frame-Stewart algorithm produces a solution with fewer moves than the classic solution."""
        for n in range(5, 15):
            classic_moves, _ = solve_hanoi_recursive(n, 'A', 'B', 'C')
            fs_moves, _ = solve_frame_stewart(n, 'A', 'B', 'C', 'D')
            assert len(fs_moves) <= len(classic_moves), f"Expected Frame-Stewart to use fewer moves than classic for {n} disks"
    
    def test_move_validation(self):
        """Test that move validation works correctly."""
        state = {'A': [3, 2, 1], 'B': [], 'C': []}
        
        # Valid move: smaller disk onto empty peg
        assert is_valid_move(state, 'A', 'B') == True
        
        # Apply the move
        apply_move(state, 'A', 'B')
        assert state == {'A': [3, 2], 'B': [1], 'C': []}
        
        # Valid move: smaller disk onto larger disk
        assert is_valid_move(state, 'A', 'C') == True
        
        # Apply the move
        apply_move(state, 'A', 'C')
        assert state == {'A': [3], 'B': [1], 'C': [2]}
        
        # Invalid move: larger disk onto smaller disk
        assert is_valid_move(state, 'A', 'B') == False
        
        # Try to apply the invalid move (should not change state)
        result = apply_move(state, 'A', 'B')
        assert result == False
        assert state == {'A': [3], 'B': [1], 'C': [2]}
    
    def test_is_solved(self):
        """Test that the is_solved function works correctly."""
        # Not solved
        state1 = {'A': [3, 2, 1], 'B': [], 'C': []}
        assert is_solved(state1, 3) == False
        
        # Solved
        state2 = {'A': [], 'B': [], 'C': [3, 2, 1]}
        assert is_solved(state2, 3) == True
        
        # Wrong order (not solved)
        state3 = {'A': [], 'B': [], 'C': [1, 2, 3]}
        assert is_solved(state3, 3) == False

def run_tests():
    test = TestTowerOfHanoi()
    test.test_recursive_solution()
    test.test_iterative_solution()
    test.test_frame_stewart()
    test.test_move_validation()
    test.test_is_solved()
    st.success("All tests passed!")

if __name__ == "__main__":
    main()