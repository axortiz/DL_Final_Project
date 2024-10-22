import csv
import pandas as pd

# Define the essential columns based on your provided CSV structure
essential_columns = [
    'f_boxer', 's_boxer', 'f_boxer_result', 'f_boxer_age', 'f_boxer_height', 'f_boxer_reach',
    'f_boxer_won', 'f_boxer_lost', 'f_boxer_KOs',
    's_boxer_age', 's_boxer_height', 's_boxer_reach',
    's_boxer_won', 's_boxer_lost', 's_boxer_KOs',
    'matchRounds', 'fightEnd'
]


def clean_fight_data(df):
    # Remove columns not in the essential list
    df_cleaned = df[essential_columns].copy()

    # Drop rows with missing values (you can adjust this depending on your needs)
    df_cleaned.dropna(inplace=True)

    # Convert data types if necessary (for example, convert numeric columns to floats/integers)
    numeric_columns = [
        'f_boxer_age', 'f_boxer_height', 'f_boxer_reach', 'f_boxer_won', 'f_boxer_lost',
        'f_boxer_KOs', 's_boxer_age', 's_boxer_height', 's_boxer_reach',
        's_boxer_won', 's_boxer_lost', 's_boxer_KOs', 'matchRounds'
    ]

    df_cleaned[numeric_columns] = df_cleaned[numeric_columns].apply(
        pd.to_numeric, errors='coerce')

    # Determine winner based on f_boxer_result
    df_cleaned['winner'] = df_cleaned.apply(
        lambda row: row['f_boxer'] if row['f_boxer_result'] == 'won' else row['s_boxer'], axis=1)

    return df_cleaned


# Load CSV data
try:
    df = pd.read_csv(
        '/Users/lexoortiz/Desktop/CS/CS307/DL_Final_Project/raw_Data/cleandata_v3.csv', delimiter=",")
except FileNotFoundError:
    print("The file could not be found.")
except pd.errors.EmptyDataError:
    print("The file is empty or improperly formatted.")
except pd.errors.ParserError:
    print("An error occurred while parsing the file.")

# Clean the data
if 'df' in locals():
    cleaned_df = clean_fight_data(df)

    # Save cleaned data to a new CSV
    cleaned_df.to_csv(
        '/Users/lexoortiz/Desktop/CS/CS307/DL_Final_Project/data_src/extra.csv', index=False)
    print("Data cleaned and saved to extra.csv")
else:
    print("Failed to load the CSV file.")


def normalize_row(row):
    """
    Normalize a row by stripping whitespace and making everything lowercase for accurate comparison.
    """
    return [str(cell).strip().lower() for cell in row]


def read_and_clean_csv(file_path):
    """
    Read the CSV, clean and normalize the data, then return as a list of lists.
    """
    try:
        df = pd.read_csv(file_path)
        # Strip whitespace and lowercase
        df = df.applymap(lambda x: str(x).strip().lower())
        return df.values.tolist()  # Convert to list of lists
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except pd.errors.EmptyDataError:
        print(f"File {file_path} is empty or improperly formatted.")
    except pd.errors.ParserError:
        print(f"An error occurred while parsing {file_path}.")
    return []


def normalize_row(row):
    """
    Normalize a row by stripping whitespace and making everything lowercase for accurate comparison.
    """
    return [str(cell).strip().lower() for cell in row]


def read_and_clean_csv(file_path):
    """
    Read the CSV, clean and normalize the data, then return as a list of lists.
    """
    try:
        df = pd.read_csv(file_path)
        # Apply normalization (strip whitespace and convert to lowercase)
        # Ensure case insensitivity and whitespace normalization
        df = df.applymap(lambda x: str(x).strip().lower())
        return df.values.tolist()  # Convert to list of lists
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except pd.errors.EmptyDataError:
        print(f"File {file_path} is empty or improperly formatted.")
    except pd.errors.ParserError:
        print(f"An error occurred while parsing {file_path}.")
    return []


def normalize_row(row):
    """
    Normalize a row by stripping whitespace, making everything lowercase, 
    and converting integers to floats where applicable.
    """
    normalized_row = []
    for cell in row:
        # Convert to string, strip whitespace, and make lowercase
        cell = str(cell).strip().lower()
        try:
            # Try to convert to float if it's a number
            if '.' not in cell:  # If it's an integer, convert to float
                cell = float(cell)
            else:
                cell = float(cell)  # Keep floats as they are
        except ValueError:
            pass  # If conversion fails, keep the original string
        normalized_row.append(cell)
    return normalized_row


def read_and_clean_csv(file_path):
    """
    Read the CSV, clean and normalize the data, then return as a list of lists.
    """
    try:
        df = pd.read_csv(file_path)
        # Apply normalization to each row
        df = df.applymap(lambda x: normalize_row([x])[
                         0])  # Normalize each cell
        return df.values.tolist()  # Convert to list of lists
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except pd.errors.EmptyDataError:
        print(f"File {file_path} is empty or improperly formatted.")
    except pd.errors.ParserError:
        print(f"An error occurred while parsing {file_path}.")
    return []


def compare_csv_files(file1, file2, outliers_file):
    """
    Compare two CSV files and write rows that are in file2 but not in file1 to outliers_file.
    """
    # Read and clean both files
    file1_data = read_and_clean_csv(file1)
    file2_data = read_and_clean_csv(file2)

    # Convert file1 data to a set for faster comparison
    file1_set = {tuple(row) for row in file1_data}

    # Identify outliers (rows in file2 but not in file1)
    outliers = [row for row in file2_data if tuple(row) not in file1_set]

    # Write outliers to a new CSV file
    if outliers:
        pd.DataFrame(outliers).to_csv(outliers_file, index=False, header=False)
        print(f"Outliers written to {outliers_file}")
    else:
        print("No outliers found.")


# Example usage
file1 = '/Users/lexoortiz/Desktop/CS/CS307/DL_Final_Project/data_src/extra.csv'
file2 = '/Users/lexoortiz/Desktop/CS/CS307/DL_Final_Project/data_src/train.csv'
outliers_file = '/Users/lexoortiz/Desktop/CS/CS307/DL_Final_Project/data_src/outliers.csv'

compare_csv_files(file1, file2, outliers_file)
