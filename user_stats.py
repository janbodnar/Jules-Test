
import pandas as pd

def analyze_user_data(file_path):
    """
    Loads user data from a CSV file and prints descriptive statistics
    for numerical columns and value counts for key categorical columns.
    """
    try:
        # Load the dataset
        users_df = pd.read_csv(file_path)

        # --- Numerical Analysis ---
        print("Descriptive Statistics for Numerical Columns:")
        # Select only numerical columns for descriptive statistics
        numerical_stats = users_df.describe()
        print(numerical_stats)
        print("\n" + "="*50 + "\n")

        # --- Categorical Analysis ---
        print("Value Counts for Categorical Columns:")

        # Analyze 'occupation' column
        if 'occupation' in users_df.columns:
            print("Occupation:")
            occupation_counts = users_df['occupation'].value_counts()
            print(occupation_counts)
        else:
            print("'occupation' column not found.")

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    analyze_user_data('users.csv')
