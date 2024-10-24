from typing import Dict, List

import pandas as pd
import re
import polyline
from geopy.distance import geodesic


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    
    Args:
    - lst (List[int]): The list to be reversed in groups.
    - n (int): The size of each group to reverse.
    
    Returns:
    - List[int]: The list with elements reversed in groups of n.
    """
    if n <= 0:
        raise ValueError("n must be greater than 0")
    
    result = []
    for i in range(0, len(lst), n):
        group = []
        for j in range(i, min(i + n, len(lst))):
            group.insert(0, lst[j])
        for item in group:
            result.append(item)
    return result

#print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8],3))

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary sorted by key in ascending order.
    
    Args:
    - lst (List[str]): The list of strings to be grouped by length.
    
    Returns:
    - Dict[int, List[str]]: A dictionary where the keys are lengths and the values are lists of strings of that length.
    """
    result = {}
    for string in lst:
        length = len(string)
        if length not in result:
            result[length] = []
        result[length].append(string)
    
    # Sort the dictionary by key in ascending order
    sorted_result = dict(sorted(result.items()))
    return sorted_result

#print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    def _flatten(obj, parent_key=''):
        items = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                items.extend(_flatten(v, new_key).items())
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                new_key = f"{parent_key}[{i}]"
                items.extend(_flatten(v, new_key).items())
        else:
            items.append((parent_key, obj))
        return dict(items)
    
    return _flatten(nested_dict)



def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(start=0):
        if start == len(nums):
            result.append(nums[:])
            return
        seen = set()
        for i in range(start, len(nums)):
            if nums[i] in seen:
                continue
            seen.add(nums[i])
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]

    result = []
    nums.sort()  # Sort the numbers to handle duplicates
    backtrack()
    return result

#print(unique_permutations([1, 1, 2]))

def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',  # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b' # yyyy.mm.dd
    ]
    
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        dates.extend(matches)
    
    return dates

#print(find_all_dates("I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."))

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Decode the polyline string into a list of (latitude, longitude) tuples
    coordinates = polyline.decode(polyline_str)
    
    # Initialize lists to store latitude, longitude, and distance
    latitudes = []
    longitudes = []
    distances = []
    
    # Iterate through the coordinates to populate the lists
    for i, (lat, lon) in enumerate(coordinates):
        latitudes.append(lat)
        longitudes.append(lon)
        if i == 0:
            distances.append(0)  # Distance for the first point is 0
        else:
            # Calculate the distance from the previous point
            prev_point = (latitudes[i-1], longitudes[i-1])
            current_point = (lat, lon)
            distance = geodesic(prev_point, current_point).meters
            distances.append(distance)
    
    # Create a DataFrame from the lists
    df = pd.DataFrame({
        'latitude': latitudes,
        'longitude': longitudes,
        'distance': distances
    })
    
    return df



def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    if not matrix or not matrix[0]:
        return []

    # Rotate the matrix by 90 degrees clockwise
    n = len(matrix)
    rotated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]

    # Multiply each element by the sum of its original row and column index
    transformed_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            original_row = n - 1 - j
            original_col = i
            transformed_matrix[i][j] = rotated_matrix[i][j] * (original_row + original_col)

    return transformed_matrix

#print(rotate_and_multiply_matrix([[1, 2, 3],[4, 5, 6],[7, 8, 9]]))
    

def time_check(df: pd.DataFrame) -> pd.Series:
    """
    You are given a dataset, `dataset-1.csv`, containing columns `id`, `id_2`, and timestamp (`startDay`, `startTime`, `endDay`, `endTime`). The goal is to verify the completeness of the time data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour period (from 12:00:00 AM to 11:59:59 PM) and span all 7 days of the week (from Monday to Sunday).

    Create a function that accepts `dataset-1.csv` as a DataFrame and returns a boolean series that indicates if each (`id`, `id_2`) pair has incorrect timestamps. The boolean series must have multi-index (`id`, `id_2`).
    """
    # Convert start and end times to datetime
    df['startTime'] = pd.to_datetime(df['startTime'], format='%H:%M:%S').dt.time
    df['endTime'] = pd.to_datetime(df['endTime'], format='%H:%M:%S').dt.time

    # Create a set of all possible days and times
    all_days = set(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    all_times = set(pd.date_range('00:00:00', '23:59:59', freq='S').time)

    def check_completeness(group):
        days_covered = set(group['startDay']) | set(group['endDay'])
        times_covered = set(group['startTime']) | set(group['endTime'])
        return not (days_covered == all_days and times_covered == all_times)

    result = df.groupby(['id', 'id_2']).apply(check_completeness)
    return result


"""path = "MapUp-DA-Assessment-2024-main\datasets\dataset-1.csv"
df = pd.read_csv(path)
result = time_check(df)
print(result.head())"""