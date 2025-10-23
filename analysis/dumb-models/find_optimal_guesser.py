import numpy as np

# --- 1. DEFINE THE GROUND TRUTH VALUES FROM YOUR TEST SET ---
TOTAL_ROWS = 1504162
POS_BLUE = 52817
POS_ORANGE = 46864

NEG_BLUE = TOTAL_ROWS - POS_BLUE
NEG_ORANGE = TOTAL_ROWS - POS_ORANGE

# --- 2. DEFINE THE CORE CALCULATION LOGIC ---

def calculate_f1(guess_rate, actual_positives, total_rows):
    """
    Calculates the F1-Score for a random guesser with a given guess_rate.
    
    Args:
        guess_rate (float): The percentage of time the model guesses "Positive".
        actual_positives (int): The total number of true positive instances in the dataset.
        total_rows (int): The total number of rows in the dataset.
        
    Returns:
        float: The calculated F1-Score.
    """
    # For a random guesser, Precision is fixed and equals the natural probability of a positive class.
    # This is because the guess is independent of the data.
    precision = actual_positives / total_rows
    
    # For a random guesser, Recall is directly equal to the guess rate.
    # If you guess "Positive" 50% of the time, you will find 50% of the actual positives.
    recall = guess_rate
    
    # F1-Score formula, with a check to prevent division by zero.
    if (precision + recall) == 0:
        return 0.0
    
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

# --- 3. RUN THE OPTIMIZATION SEARCH ---

# We will test 10,000 different guessing percentages from 0.01% to 100%
# to find the one that yields the best F1-Score.
guess_rates_to_test = np.linspace(0.0001, 1.0, 10000)

# --- Scenario 1: Find Optimal Guessing Rate for BLUE team ---
best_f1_blue = 0.0
best_rate_blue = 0.0

for rate in guess_rates_to_test:
    f1 = calculate_f1(rate, POS_BLUE, TOTAL_ROWS)
    if f1 > best_f1_blue:
        best_f1_blue = f1
        best_rate_blue = rate

# --- Scenario 2: Find Optimal Guessing Rate for ORANGE team ---
best_f1_orange = 0.0
best_rate_orange = 0.0

for rate in guess_rates_to_test:
    f1 = calculate_f1(rate, POS_ORANGE, TOTAL_ROWS)
    if f1 > best_f1_orange:
        best_f1_orange = f1
        best_rate_orange = rate

# --- Scenario 3: Find Optimal Rate for the AVERAGE F1 of Both Teams ---
best_avg_f1 = 0.0
best_rate_avg = 0.0

for rate in guess_rates_to_test:
    f1_blue = calculate_f1(rate, POS_BLUE, TOTAL_ROWS)
    f1_orange = calculate_f1(rate, POS_ORANGE, TOTAL_ROWS)
    avg_f1 = (f1_blue + f1_orange) / 2
    
    if avg_f1 > best_avg_f1:
        best_avg_f1 = avg_f1
        best_rate_avg = rate

# --- 4. PRINT THE RESULTS ---

print("=" * 50)
print("   OPTIMAL RANDOM GUESSER ANALYSIS")
print("=" * 50)
print("\n--- Scenario 1: Maximizing F1-Score for BLUE Team ---")
print(f"Optimal Guessing Percentage: {best_rate_blue:.2%}")
print(f"Maximum Achievable F1-Score: {best_f1_blue:.4f} (or {best_f1_blue:.2%})")

print("\n--- Scenario 2: Maximizing F1-Score for ORANGE Team ---")
print(f"Optimal Guessing Percentage: {best_rate_orange:.2%}")
print(f"Maximum Achievable F1-Score: {best_f1_orange:.4f} (or {best_f1_orange:.2%})")

print("\n--- Scenario 3: Maximizing AVERAGE F1-Score for BOTH Teams ---")
print(f"Optimal Guessing Percentage: {best_rate_avg:.2%}")
print(f"Maximum Achievable Average F1-Score: {best_avg_f1:.4f} (or {best_avg_f1:.2%})")

print("\n" + "="*50)
print("Conclusion: The script numerically confirms that to maximize the")
print("F1-Score, a random model must always guess the positive class (100% rate).")
print("=" * 50)