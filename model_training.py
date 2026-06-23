import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import haversine_distances


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


def run_greedy_transit_optimization(file_path='final_analytical_dataset.csv', num_new_stations=3, coverage_radius_meters=800):
    """
    Urban Optimization using a greedy algorithm for Maximum Coverage.
    Finds the optimal locations for new transit hubs to cover "transit deserts".
    """
    print("Loading property data for Urban Optimization")
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['Lat', 'Long', 'Distance_To_Hub_Meters']).copy()

    # Identify the "transit deserts" (properties that are too far from any hub)
    unserved_df = df[df['Distance_To_Hub_Meters']
                     > coverage_radius_meters].copy()
    print(
        f"Identified {len(unserved_df)} properties in 'Transit Deserts' (> {coverage_radius_meters}m from a hub).")

    # Extract coordinates in radians for Haversine distance calculations
    unserved_coords = np.radians(unserved_df[['Lat', 'Long']].values)

    # We consider every currently unserved property location as a potential candidate site for a new station
    candidate_coords = unserved_coords.copy()
    candidate_lat_long = unserved_df[['Lat', 'Long']].values

    earth_rad_meters = 6371000
    radius_rad = coverage_radius_meters / earth_rad_meters

    # to track the greedy algorithm
    covered_indices = set()
    new_station_locations = []
    properties_covered_per_step = []

    print(
        f"\nRunning Greedy Algorithm to place {num_new_stations} new stations.")

    for step in range(num_new_stations):
        best_candidate_idx = -1
        max_new_covered = -1
        best_new_covered_indices = set()

        for i, candidate in enumerate(candidate_coords):
            # Calculate distance from this candidate to all unserved properties
            dists = haversine_distances([candidate], unserved_coords)[0]

            # find which properties are within the radius
            within_radius_indices = set(np.where(dists <= radius_rad)[0])

            # Calculate the gain
            new_coverage = within_radius_indices - covered_indices

            if len(new_coverage) > max_new_covered:
                max_new_covered = len(new_coverage)
                best_candidate_idx = i
                best_new_covered_indices = new_coverage

        covered_indices.update(best_new_covered_indices)
        new_station_locations.append(candidate_lat_long[best_candidate_idx])
        properties_covered_per_step.append(max_new_covered)

        print(f"Step {step + 1}: Station placed at {candidate_lat_long[best_candidate_idx]}. "
              f"Gain: covered {max_new_covered} new properties")

    print("\nGenerating Urban Optimization Map.")
    plt.figure(figsize=(14, 10))
    plt.gca().set_aspect(1 / np.cos(np.radians(31.77)))

    served_df = df[df['Distance_To_Hub_Meters'] <= coverage_radius_meters]
    plt.scatter(served_df['Long'], served_df['Lat'],
                c='lightgrey', s=20, label='Already Served (<800m)', alpha=0.5)
    plt.scatter(unserved_df['Long'], unserved_df['Lat'], c='coral',
                s=35, label='Transit Desert (Unserved)', alpha=0.8)

    new_station_df = pd.DataFrame(
        new_station_locations, columns=['Lat', 'Long'])
    plt.scatter(new_station_df['Long'], new_station_df['Lat'], c='teal',
                marker='*', s=600, edgecolor='black', label='Proposed New Hubs')

    plt.xlim(35.10, 35.28)
    plt.ylim(31.70, 31.88)

    plt.title('Urban Optimization: Proposed Transit Hubs via Greedy Algorithm',
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Longitude', fontsize=14)
    plt.ylabel('Latitude', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig('visualization_4.png', dpi=400, bbox_inches='tight')
    plt.close()

    print("\nProof of Submodularity (Diminish Marginal Returns)")
    for i, count in enumerate(properties_covered_per_step):
        print(f"Station {i+1} added {count} to total coverage.")

    print("Saved 'visualization_4.png'.")


if __name__ == "__main__":
    # train_and_evaluate_model()
    run_greedy_transit_optimization()
