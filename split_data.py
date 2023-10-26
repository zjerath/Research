from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.3, random_state=42):
    '''
    Split data into training, validation, and test sets.
    
    Parameters:
    - X: Input features
    - y: Target variable
    - test_size: Proportion of data to be used for the test set
    - random_state: Random seed for reproducibility
    
    Returns:
    - X_train, X_val, X_test: Split input features
    - y_train, y_val, y_test: Split target variable
    '''
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test