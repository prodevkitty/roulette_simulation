import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE

# Generate synthetic player data
num_players = 500
player_data = {
    'player_id': [f'player{i}' for i in range(1, num_players + 1)],
    'total_wagered': np.random.randint(1000, 100000, num_players),
    'betting_frequency': np.random.randint(1, 50, num_players),
    'wins': np.random.randint(0, 20, num_players),
    'losses': np.random.randint(0, 20, num_players),
    'average_bet': np.random.randint(100, 5000, num_players),
    'win_rate': np.random.rand(num_players),
    'risk_profile': np.random.choice(['low', 'medium', 'high'], num_players)  # Add risk_profile column
}

player_df = pd.DataFrame(player_data)
player_df.to_csv('data/player_data.csv', index=False)

# Generate synthetic bank data
bank_data = {
    'bank_balance': [1000000],
    'profit_margin': [0.5]
}

bank_df = pd.DataFrame(bank_data)
bank_df.to_csv('data/bank_data.csv', index=False)

print("Synthetic data generated successfully.")

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

def train_model(data):
    df = pd.DataFrame(data)
    if 'risk_profile' not in df.columns:
        raise KeyError("The 'risk_profile' column is missing from the dataset.")
    
    # Clean and analyze data
    df = clean_and_analyze_data(df)
    
    X = df[['total_wagered', 'betting_frequency', 'wins', 'losses', 'average_bet', 'win_rate']]
    y = df['risk_profile']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Feature Selection using RFE
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    rfe = RFE(model, n_features_to_select=5)
    X_train_rfe = rfe.fit_transform(X_train, y_train)
    X_test_rfe = rfe.transform(X_test)
    
    # Define individual models
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    rf = RandomForestClassifier()
    lr = LogisticRegression(max_iter=1000)
    
    # Create a stacking classifier
    estimators = [
        ('xgb', xgb),
        ('rf', rf)
    ]
    stack_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    
    param_grid = {
        'xgb__n_estimators': [50, 100, 200],
        'xgb__learning_rate': [0.01, 0.05, 0.1],
        'xgb__max_depth': [3, 5, 7],
        'rf__n_estimators': [50, 100, 200],
        'rf__max_depth': [None, 10, 20],
        'final_estimator__C': [0.1, 1, 10]
    }
    
    grid_search = GridSearchCV(estimator=stack_model, param_grid=param_grid, scoring='accuracy', cv=StratifiedKFold(n_splits=5), n_jobs=-1, verbose=2)
    grid_search.fit(X_train_rfe, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_rfe)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    
    return best_model, label_encoder

def retrain_model(data):
    return train_model(data)