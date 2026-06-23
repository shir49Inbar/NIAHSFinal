import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


def train_and_evaluate_model(file_path='final_analytical_dataset.csv'):
    """
    Train a Decision Tree model to predict property value based on transit distance, and visualize the optimal "Sweet Spot"
    """
    print("Loading final analytical dataset.")
    df = pd.read_csv(file_path)

    # Clean missing values in our target and feature column
    df = df.dropna(subset=['price_per_sqrm', 'Distance_To_Hub_Meters']).copy()

    # Define feature and target (X, y)
    X = df[['Distance_To_Hub_Meters']]
    y = df['price_per_sqrm']

    # Splitting train, test batches 80%/20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Initialize and train the decision tree regressor
    tree_model = DecisionTreeRegressor(
        max_depth=4, min_samples_leaf=10, random_state=42)
    tree_model.fit(X_train, y_train)

    # Predict on the test set and evaluate using RMSE
    y_pred = tree_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model Evaluation Complete- Test RMSE: {rmse:.2f} ILS")

    print("Generating final visualization.")
    plt.figure(figsize=(10, 6))

    sns.scatterplot(data=df, x='Distance_To_Hub_Meters', y='price_per_sqrm',
                    alpha=0.4, color='coral', label='Actual Real Estate Data')

    # Create a smooth line for the decision tree predictions across all distances
    X_plot = np.linspace(X['Distance_To_Hub_Meters'].min(
    ), X['Distance_To_Hub_Meters'].max(), 500).reshape(-1, 1)

    X_plot_df = pd.DataFrame(X_plot, columns=['Distance_To_Hub_Meters'])
    y_plot = tree_model.predict(X_plot_df)

    plt.plot(X_plot_df, y_plot, color='teal', linewidth=3,
             label='Decision Tree Prediction')

    plt.title('Finding the Sweet Spot: Property Value vs. Distance to Transit Hub',
              fontsize=14, fontweight='bold')
    plt.xlabel('Distance to Nearest Busy Hub (Meters)', fontsize=12)
    plt.ylabel('Price per Square Meter (ILS)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.xlim(0, 4000)

    plt.savefig('visualization_3.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved 'visualization_3.png'. The project pipeline is officially complete!")


if __name__ == "__main__":
    train_and_evaluate_model()
