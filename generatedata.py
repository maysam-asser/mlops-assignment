# generatedata.py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def generate_and_save_data():
    """Load Iris dataset, split it, and save to CSV files"""
    
    # Load Iris data
    iris = load_iris()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    # Create DataFrames with feature names
    feature_names = iris.feature_names
    
    # Training data
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['target'] = y_train
    
    # Test data
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['target'] = y_test
    
    # Save to CSV files
    train_df.to_csv('train_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)
    
    # Also save full dataset with target
    full_df = pd.DataFrame(iris.data, columns=feature_names)
    full_df['target'] = iris.target
    full_df.to_csv('iris_full.csv', index=False)
    
    print("Data files generated successfully:")
    print(f"  - train_data.csv: {len(train_df)} samples")
    print(f"  - test_data.csv: {len(test_df)} samples")
    print(f"  - iris_full.csv: {len(full_df)} samples")
    
    return train_df, test_df, full_df

if __name__ == "__main__":
    generate_and_save_data()