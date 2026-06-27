import os
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from sklearn.metrics.pairwise import haversine_distances
import kagglehub
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import seaborn as sns


def load_and_prepare_data():
    """
    Load the pre-filtered Jerusalem transit stations and fetch real estate data from Yad2 (via Kaggle).
    """
    print("Loading Jerusalem stations data from stations.csv")
    df_jerusalem_stops = pd.read_csv('stations.csv', encoding='utf-8')

    # Strip white spaces and string encapsulations
    for col in df_jerusalem_stops.select_dtypes(include=['object', 'string']).columns:
        df_jerusalem_stops[col] = df_jerusalem_stops[col].astype(
            str).str.strip('"').str.strip()

    jerusalem_stop_ids = set(df_jerusalem_stops['StationId'].astype(str))
    print(f"Successfully loaded {len(df_jerusalem_stops)} bus stations.")

    # Fetching the Hebrew Yad2 dataset directly from Kaggle servers
    print("Downloading updated real estate listring from Kaggle.")
    path = kagglehub.dataset_download(
        "amirjayousi/yad2-real-estate-listings-hebrew")
    yad2_file_path = os.path.join(path, "cleaned_yad2_data.csv")
    df_yad2 = pd.read_csv(yad2_file_path)

    for col in df_yad2.select_dtypes(include=['object']).columns:
        df_yad2[col] = df_yad2[col].astype(str).str.strip('"').str.strip()

    df_yad2_jerusalem = df_yad2[df_yad2['city'] == 'ירושלים'].copy()
    df_yad2_jerusalem.to_csv('yad2_jerusalem.csv',
                             index=False, encoding='utf-8-sig')
    print(f"Saved filtered real estate file to 'yad2_jerusalem.csv'.")

    return df_jerusalem_stops, jerusalem_stop_ids, df_yad2_jerusalem


def compute_transit_frequencies(df_jerusalem_stops, jerusalem_stop_ids):
    """
    Parse the GTFS stop_times.txt using chunking to calculate daily bus volume per station
    """
    print("Scanning stop_times.txt sequentially...")
    stop_counts = {}
    chunk_size = 100000

    # Iterate through the large text file line-by-line using context chunks
    chunk_iterator = pd.read_csv(
        'stop_times.txt', chunksize=chunk_size, dtype={'stop_id': str})
    for chunk in chunk_iterator:
        chunk['stop_id'] = chunk['stop_id'].astype(
            str).str.strip('"').str.strip()
        # Filter for rows that match our established Jerusalem Station IDs
        jerusalem_rows = chunk[chunk['stop_id'].isin(jerusalem_stop_ids)]
        for stop_id in jerusalem_rows['stop_id']:
            stop_counts[stop_id] = stop_counts.get(stop_id, 0) + 1

    # Convert the frequency dictionary into a structured dataframe
    df_frequencies = pd.DataFrame(list(stop_counts.items()), columns=[
                                  'StationId', 'Daily_Bus_Volume'])

    # Type normalization before executing the merge
    df_frequencies['StationId'] = df_frequencies['StationId'].astype(int)
    df_jerusalem_stops['StationId'] = df_jerusalem_stops['StationId'].astype(
        int)

    # Join to merge coordinates with frequency metrics
    df_stops_enriched = pd.merge(
        df_jerusalem_stops, df_frequencies, on='StationId', how='left')
    df_stops_enriched['Daily_Bus_Volume'] = df_stops_enriched['Daily_Bus_Volume'].fillna(
        0).astype(int)

    df_stops_enriched.to_csv(
        'jerusalem_stops_with_frequency.csv', index=False, encoding='utf-8-sig')
    print("Saved merged transit records to 'jerusalem_stops_with_frequencies.csv'.")

    return df_stops_enriched


def extract_busy_transit_hubs(df_stops_enriched, freq_threshold=400, eps_meters=300, min_samples=15):
    """
    Applying DBSCAN to isolate and extract dense spatial clusters of high-volume transit hubs, eliminating low traffic
    """
    print("Executing DBSCAN to discover dense geographical transit hubs.")

    # Filter using the frequency metric to mitigate right-skewed data bias
    busy_nodes = df_stops_enriched[df_stops_enriched['Daily_Bus_Volume']
                                   >= freq_threshold].copy()

    if busy_nodes.empty:
        print("Error: No transit stations meet the defines frequency threshold.")
        return busy_nodes

    coords = busy_nodes[['Lat', 'Long']].values
    coords_rad = np.radians(coords)

    earth_rad = 6371000
    eps_rad = eps_meters / earth_rad

    dbscan = DBSCAN(eps=eps_rad, min_samples=min_samples,
                    algorithm='ball_tree', metric='haversine')
    busy_nodes['Hub_Cluster_ID'] = dbscan.fit_predict(coords_rad)

    verified_hubs = busy_nodes[busy_nodes['Hub_Cluster_ID'] != -1].copy()

    distinct_hubs_count = len(set(verified_hubs['Hub_Cluster_ID']))
    print(
        f"Discovered {distinct_hubs_count} distinct high-volume transit hubs in Jerusalem.")

    return verified_hubs


def geocode_yad2_addresses(df_yad2):
    """
    Convert Street Addresses from YAD2 into geographical coordinates (Lat, Long).
    Uses the free OpenStreetMap Nomnatim API.
    """
    print("Starting Geocoding process for Yad2 addresses. This may take a while.")

    # Initialize the geocoder with a custom user agent
    geolocator = Nominatim(user_agent="jerusalem_transit_real_estate_research")
    # Use RateLimiter to avoid hitting the API too fast (1 req per second)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    # Create a full address string tailored for Jerusalem
    df_yad2['full_address'] = df_yad2['street'] + \
        ", " + df_yad2['neighborhood'] + ", " + df_yad2['city']

    def get_coordinates(address):
        try:
            location = geocode(address)
            if location:
                return pd.Series((location.latitude, location.longitude))
            return pd.Series((np.nan, np.nan))
        except Exception as e:
            return pd.Series((np.nan, np.nan))

    # Apply the function and create new Lat and Long columns
    df_yad2[['Lat', 'Long']] = df_yad2['full_address'].apply(get_coordinates)

    # Drop rows where geocoding failed
    df_yad2_clean = df_yad2.dropna(subset=['Lat', 'Long']).copy()

    print(
        f"Geocoding complete. Successfully geocoded {len(df_yad2_clean)} out of {len(df_yad2)} properties.")
    return df_yad2_clean


def calculate_distance_to_nearest_hub(df_properties, df_hubs):
    """
    Calculate the minimum Haversine distance from each propery to the nearest busy transit hub.
    """
    print("Calculating distances from properties to the nearest transit hubs.")

    # Extract coordinates and convert degress to radians for the Haversine formula
    prop_coords_rad = np.radians(df_properties[['Lat', 'Long']].values)
    hub_coords_rad = np.radians(df_hubs[['Lat', 'Long']].values)

    # Calculate pairwaise distances between all properties and all hubs
    distance_matrix_rad = haversine_distances(prop_coords_rad, hub_coords_rad)

    # Multiply by Earth's radius in meters to get actual physical distances
    earth_rad_meters = 6371000
    distance_matrix_meters = distance_matrix_rad * earth_rad_meters

    # find the minimum distance for each property
    min_distances = np.min(distance_matrix_meters, axis=1)

    # Add the result as a new feature to the real estate dataframe
    df_properties['Distance_To_Hub_Meters'] = min_distances

    # Save the final analytical dataset
    df_properties.to_csv('final_analytical_dataset.csv',
                         index=False, encoding='utf-8-sig')
    print("Successfully created 'final_analytical_dataset.csv' with distance features.")

    return df_properties


def visualize_dbscan_clusters(df_stops_enriched, df_dbscan_hubs, freq_threshold=400):
    """
    Visualize the transit hubs identified by the DBSCAN algorithm.
    Shows the clustered hubs in colors and the rejected 'noise' stations in gray.
    """
    print("Generating DBSCAN clusters visualization")

    # Get all busy stations that were evaluated by the algorithm.
    busy_nodes = df_stops_enriched[df_stops_enriched['Daily_Bus_Volume']
                                   >= freq_threshold].copy()

    # Identify the noise nodes (stations that were evaluated but rejected by DBSCAN)
    noise_nodes = busy_nodes[~busy_nodes['StationId'].isin(
        df_dbscan_hubs['StationId'])]

    plt.figure(figsize=(12, 10))
    plt.gca().set_aspect(1 / np.cos(np.radians(31.77)))

    plt.scatter(noise_nodes['Long'], noise_nodes['Lat'], c='lightgray',
                s=30, alpha=0.6, label='Noise (Unclustered Stations)')
    sns.scatterplot(data=df_dbscan_hubs, x='Long', y='Lat', hue='Hub_Cluster_ID',
                    palette='tab10', s=120, alpha=0.9, edgecolor='black', legend='full')
    plt.title('DBSCAN Clustering: Identified Major Transit Hubs in Jerusalem',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)

    plt.legend(title='DBSCAN Cluster ID',
               bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig('visualization_dbscan_hubs.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved 'visualization_dbscan_hubs.png'.")


if __name__ == "__main__":
    df_jeru_stops, jlm_stops_ids, df_real_estate = load_and_prepare_data()
    df_enriched_transit = compute_transit_frequencies(
        df_jeru_stops, jlm_stops_ids)
    df_final_hubs = extract_busy_transit_hubs(
        df_enriched_transit, freq_threshold=400, eps_meters=300, min_samples=5)
    visualize_dbscan_clusters(df_enriched_transit, df_final_hubs)
    df_yad2_geocoded = geocode_yad2_addresses(df_real_estate)
    final_dataset = calculate_distance_to_nearest_hub(
        df_yad2_geocoded, df_final_hubs)

    print("\n--- Pipeline Fully Completed ---")
    print(final_dataset[['street', 'neighborhood', 'price',
          'area_sqm', 'Distance_To_Hub_Meters']].head())
