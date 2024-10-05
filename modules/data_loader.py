import pandas as pd
import os

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    print(data)
    # Handle missing values
    data.ffill(inplace=True)
    
    # Feature Engineering: Create new features
    data['bet_per_frequency'] = data['total_wagered'] / data['betting_frequency']
    data['win_loss_ratio'] = data['wins'] / (data['losses'] + 1)  # Add 1 to avoid division by zero
    
    # Encode categorical variables
    categorical_columns = data.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        data[column] = data[column].astype('category').cat.codes
    
    return data

def load_player_data(data_folder):
    player_data_path = os.path.join(data_folder, 'player_data.csv')
    bank_data_path = os.path.join(data_folder, 'bank_data.csv')

    player_data = pd.read_csv(player_data_path)
    bank_data = pd.read_csv(bank_data_path)

    return player_data, bank_data

def clean_and_analyze_data(data):
    # Remove duplicates
    data.drop_duplicates(inplace=True)
    
    # Remove outliers
    numeric_columns = data.select_dtypes(include=['number']).columns
    for column in numeric_columns:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    # Analyze data distribution
    print(data.describe())
    
    return data