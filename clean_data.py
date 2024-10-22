import pandas as pd

# Define the essential columns based on your provided CSV structure
essential_columns = [
    'f_boxer', 's_boxer', 'f_boxer_result', 'f_boxer_age', 'f_boxer_height', 'f_boxer_reach',
    'f_boxer_won', 'f_boxer_lost', 'f_boxer_KOs', 'f_boxer_ranking',
    's_boxer_age', 's_boxer_height', 's_boxer_reach',
    's_boxer_won', 's_boxer_lost', 's_boxer_KOs', 's_boxer_ranking',
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
        'f_boxer_KOs', 'f_boxer_ranking', 's_boxer_age', 's_boxer_height',
        's_boxer_reach', 's_boxer_won', 's_boxer_lost', 's_boxer_KOs',
        's_boxer_ranking', 'matchRounds'
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
        '/Users/lexoortiz/Desktop/CS/CS307/DL_Final_Project/data_src/cleandata_MikeTyson.csv', delimiter=",")
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
    cleaned_df.to_csv('cleaned_fight_data.csv', index=False)
    print("Data cleaned and saved to cleaned_fight_data.csv")
else:
    print("Failed to load the CSV file.")
