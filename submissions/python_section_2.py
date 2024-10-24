#!/usr/bin/env python
# coding: utf-8

# ### Question 9: Distance Matrix Calculation

import pandas as pd
from warnings import filterwarnings
filterwarnings('ignore')
from datetime import time
"""# Load the dataset
file_path = '/mnt/data/dataset-2.csv'
df = pd.read_csv(r"C:\Users\shiva\OneDrive\Desktop\MapUp-DA-Assessment-2024-main\datasets\dataset-2.csv")"""

# Display the first few rows to understand its structure
#df.head()





def calculate_distance_matrix(df) -> pd.DataFrame():
    # Extract unique toll IDs
    unique_ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))

    # Initialize a distance matrix with zeros
    distance_matrix = pd.DataFrame(0, index=unique_ids, columns=unique_ids)

    # Populate the distance matrix with known distances
    for _, row in df.iterrows():
        start = row['id_start']
        end = row['id_end']
        distance = row['distance']
        
        # Set distance for both directions to ensure symmetry
        distance_matrix.at[start, end] = distance
        distance_matrix.at[end, start] = distance

    return distance_matrix

"""# Apply the function to the dataset
distance_matrix = calculate_distance_matrix(df)
distance_matrix.head()"""


# ### Question - 10 - Unroll Distance Matrix



def unroll_distance_matrix(df) -> pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Unroll the matrix into a DataFrame
    unrolled_data = []
    
    # Iterate over all combinations of id_start and id_end
    for id_start in df.index:
        for id_end in df.columns:
            if id_start != id_end:  # Exclude same id_start to id_end (diagonal values)
                distance = df.at[id_start, id_end]
                if distance != 0:  # Only add entries where distance is non-zero
                    unrolled_data.append([id_start, id_end, distance])
    
    # Create the resulting DataFrame
    unrolled_df = pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])
    
    return unrolled_df

"""# Apply the function to the distance matrix
unrolled_distance_df = unroll_distance_matrix(distance_matrix)
unrolled_distance_df.head()"""


# ### Qusetion 11 - Finding IDs within Percentage Threshold



def find_ids_within_ten_percentage_threshold(df, reference_id) -> pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pd.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                      of the reference ID's average distance.
    """
    # Calculate the average distance for the reference_id
    ref_distances = df[df['id_start'] == reference_id]['distance']
    ref_avg_distance = ref_distances.mean()

    # Define the 10% threshold
    lower_threshold = ref_avg_distance * 0.90
    upper_threshold = ref_avg_distance * 1.10

    # Find average distances for all other IDs
    avg_distances = df.groupby('id_start')['distance'].mean().reset_index()

    # Filter IDs whose average distance falls within the 10% threshold
    ids_within_threshold = avg_distances[(avg_distances['distance'] >= lower_threshold) & 
                                         (avg_distances['distance'] <= upper_threshold)]

    # Sort the result by id_start and return the DataFrame
    return ids_within_threshold.sort_values(by='id_start')

"""# Apply the function to the unrolled DataFrame using a sample reference ID, e.g., 1001400
reference_id = 1001400
ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_distance_df, reference_id)
ids_within_threshold"""


# ### Question 12: Calculate Toll Rate


# Define the function to calculate toll rates for each vehicle type
def calculate_toll_rate(df):
    """
    Calculate toll rates for each vehicle type based on the distance.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: DataFrame with toll rates for each vehicle type.
    """
    # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Create new columns for each vehicle type's toll rate by multiplying distance with the respective rate coefficient
    for vehicle, rate in rate_coefficients.items():
        df[vehicle] = df['distance'] * rate

    return df

"""# Apply the function to the dataset to calculate toll rates
df_with_toll_rates = calculate_toll_rate(df)

# Display the updated DataFrame with toll rates
df_with_toll_rates.head()"""


# ### Question 13: Calculate Time-Based Toll Rates


def calculate_time_based_toll_rates(df):
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: DataFrame with time-based toll rates.
    """
    # Define discount factors for weekdays and weekends
    weekday_discount_factors = {
        (time(0, 0, 0), time(10, 0, 0)): 0.8,
        (time(10, 0, 0), time(18, 0, 0)): 1.2,
        (time(18, 0, 0), time(23, 59, 59)): 0.8
    }
    weekend_discount_factor = 0.7

    # List of days for reference
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekends = ['Saturday', 'Sunday']

    # Generate toll rates for each day and time interval
    records = []
    for idx, row in df.iterrows():
        for day in weekdays + weekends:
            for (start_time, end_time), discount in weekday_discount_factors.items():
                # Apply weekend discount factor
                if day in weekends:
                    discount = weekend_discount_factor
                # Calculate toll rates for each vehicle with the discount factor applied
                rates = { 
                    'moto': row['moto'] * discount,
                    'car': row['car'] * discount,
                    'rv': row['rv'] * discount,
                    'bus': row['bus'] * discount,
                    'truck': row['truck'] * discount
                }
                # Append record for the current (id_start, id_end), day, and time range
                records.append({
                    'id_start': row['id_start'],
                    'id_end': row['id_end'],
                    'distance': row['distance'],
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time,
                    **rates
                })

    # Create a new DataFrame with the generated records
    new_df = pd.DataFrame(records)

    return new_df

"""# Apply the function to the DataFrame
df_with_time_based_rates = calculate_time_based_toll_rates(df_with_toll_rates)

# Display the first few rows of the resulting DataFrame
df_with_time_based_rates.head()"""







