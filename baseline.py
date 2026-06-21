"""
Project: A Needle in a Data Haystack
Authors: Shir Inbar, Noa Ruiz, [Team Member 3 Name]
Description: Analyzing the optimal distance between residential properties in Jerusalem and public transportation hubs.
"""

import pandas as pd
import numpy as np

# ==========================================
# Phase 1: Data & Feature Engineering
# ==========================================

def load_and_clean_real_estate(file_path):
    """
    TODO: 
    1. Load Yad2 residential listings.
    2. Filter outliers (e.g., the 10-room property price spike).
    3. Handle any data biases.
    """
    pass

def process_gtfs_data(file_path):
    """
    TODO:
    1. Extract relevant bus schedule data from the TXT file.
    2. Calculate daily bus frequencies per station.
    """
    pass

def calculate_aerial_distances(properties_df, stations_df):
    """
    TODO:
    Calculate Haversine distance between real estate coords and station coords.
    """
    pass

# ==========================================
# Phase 2: Solution & Modeling
# ==========================================

def train_baseline_model(X, y):
    """
    TODO:
    Implement a simple baseline model (e.g., Linear Regression)
    to compare against the advanced solution.
    """
    pass

def train_advanced_model(X, y):
    """
    TODO:
    Implement the main predictive algorithm (e.g., Deep Learning / Neural Network)
    that identifies the non-linear "optimal distance" sweet spot.
    """
    pass

# ==========================================
# Phase 3: Evaluation & Visualizations
# ==========================================

def evaluate_models(y_true, y_pred_baseline, y_pred_advanced):
    """
    TODO:
    Calculate metrics (like RMSE or MAE) to prove the advanced model's success.
    """
    pass

if __name__ == "__main__":
    print("Starting the A Needle in a Data Haystack Pipeline...")
    # Entry point for the scripts
