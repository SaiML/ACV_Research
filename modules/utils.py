import yaml
from holidays import UnitedStates
us_holidays = UnitedStates()
import pandas as pd
from pathlib import Path

def load_yaml_file(file_path):
    
    try:
        with open(file_path, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error: Failed to load YAML from '{file_path}': {e}")
        return None


def load_initial_files(basepath, file_list, save=True, filename='final_merge.csv'):
    dataframes = []  # List to store individual dataframes from files
    for file in file_list:
        # Read each CSV file and append its dataframe to the list
        filepath = basepath / file
        print("Loading file:", filepath)
        df = pd.read_csv(filepath)
        dataframes.append(df)
    # Concatenate all the dataframes into a single dataframe
    df = pd.concat(dataframes)
    if save:
        # Save the final concatenated dataframe to a CSV file
        df.to_csv(basepath / filename, index=False)
        print("File saved")
    return df

def missing_values_summary(df, drop=False):
    """
    Generate a summary of missing values in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
        drop (bool): Flag to drop rows with fewer than 1% missing values. Default is False.

    Returns:
        pd.DataFrame: A DataFrame summarizing the missing values in each column.
    """
    total_rows = len(df)
    missing_values_count = df.isnull().sum()
    missing_values_percentage = (missing_values_count / total_rows) * 100
    missing_values_summary = pd.DataFrame({
        'Missing Values Count': missing_values_count,
        'Missing Values Percentage': missing_values_percentage
    })

    if drop:
        # Determine rows to drop based on missing values percentage threshold
        rows_to_drop = missing_values_summary.index[missing_values_summary['Missing Values Percentage'] < 1]
        df = df.dropna(subset=rows_to_drop)
    # Exclude rows with 0% missing values from the summary
    missing_values_summary = missing_values_summary[missing_values_summary['Missing Values Percentage'] > 0]
    return df, missing_values_summary

def are_sets_non_overlapping(set1, set2, set3):
    set1=set(set1)
    set2=set(set2)
    set3=set(set3)
    
    # Check if the intersection between any two sets is empty
    assert len(set1.intersection(set2)) == 0, "Sets 1 and 2 are overlapping."
    assert len(set1.intersection(set3)) == 0, "Sets 1 and 3 are overlapping."
    assert len(set2.intersection(set3)) == 0, "Sets 2 and 3 are overlapping."

def get_best_performers(df, group_col, performance_col):
    """
    This function takes a dataframe, a column name for grouping, and a column name for performance. 
    It returns a new dataframe that contains the best-performing row from each group.

    Parameters:
    df (pandas.DataFrame): Input dataframe
    group_col (str): Column name to group by
    performance_col (str): Column name that indicates performance

    Returns:
    pandas.DataFrame: Output dataframe containing the best-performing row from each group
    """

    return df.loc[df.groupby(group_col)[performance_col].idxmax()]




# Assuming df is your DataFrame and 'date' is your date column
df = pd.DataFrame({
    'date': pd.date_range(start='2020-01-01', end='2020-12-31')
})



# Define a function to check if a date is a holiday
def is_holiday(date, us_holidays = UnitedStates()):
    holiday=date in us_holidays
    return int(holiday)

# Apply this function to your date column
df['is_holiday'] = df['date'].apply(is_holiday)

# Convert the boolean values to integers (1 for True, 0 for False)
df['is_holiday'] = df['is_holiday'].astype(int)