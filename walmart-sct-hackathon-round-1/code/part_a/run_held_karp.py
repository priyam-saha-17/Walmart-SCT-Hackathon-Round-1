import sys
import csv
import math
import numpy as np
import itertools
import random



def held_karp(dists):
    """
    Implementation of Held-Karp, an algorithm that solves the Traveling
    Salesman Problem using dynamic programming with memoization.

    Parameters:
        dists: distance matrix

    Returns:
        A tuple, (cost, path).
    """
    n = len(dists)

    # Maps each subset of the nodes to the cost to reach that subset, as well
    # as what node it passed before reaching this subset.
    # Node subsets are represented as set bits.
    C = {}

    # Set transition cost from initial state
    for k in range(1, n):
        C[(1 << k, k)] = (dists[0][k], 0)

    # Iterate subsets of increasing length and store intermediate results
    # in classic dynamic programming manner
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            # Set bits for all nodes in this subset
            bits = 0
            for bit in subset:
                bits |= 1 << bit

            # Find the lowest cost to get to this subset
            for k in subset:
                prev = bits & ~(1 << k)

                res = []
                for m in subset:
                    if m == 0 or m == k:
                        continue
                    res.append((C[(prev, m)][0] + dists[m][k], m))
                C[(bits, k)] = min(res)

    # We're interested in all bits but the least significant (the start state)
    bits = (2**n - 1) - 1

    # Calculate optimal cost
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + dists[k][0], k))
    opt, parent = min(res)

    # Backtrack to find full path
    path = []
    for i in range(n - 1):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits

    # Add implicit start state
    path.append(0)

    return opt, list(reversed(path))






def haversine(lat1, lon1, lat2, lon2):
    R = 6371 # radius of Earth in kilometers
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    res = R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return np.round(res, 2)



def compute_distance_matrix(locations):
    dists = []
    n = len(locations.keys())
    for i in range (1, n+1):
        temp = []
        for j in range (1, n+1):
            distance = haversine(locations[i][0], locations[i][1], locations[j][0], locations[j][1])
            temp.append(distance)
        dists.append(temp)
    print("Distance Matrix:")
    print(dists)
    print("\n\n\n")
    return dists
      
    
    
    
    
    


def read_csv_dataset(dataset_path):
    """
    Read the dataset from a CSV file.

    Args:
    - dataset_path: Path to the CSV dataset file.

    Returns:
    - locations: A dictionary mapping location IDs to their coordinates.
    """
    locations = {}
    with open(dataset_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row if present
        flag = 0
        count = 1
        for row in reader:
            if flag == 0:
                depot_lat = float(row[3])
                depot_long = float(row[4])
                locations[count] = (depot_lat, depot_long)
                count += 1
                lat = float(row[2])
                long = float(row[1])
                locations[count] = (lat, long)
                count += 1
                flag = 1
            else:
                lat = float(row[2])
                long = float(row[1])
                locations[count] = (lat, long)
                count += 1
    print("Locations Lat-Long Dictionary:")
    print(locations)
    print("\n\n\n")
    return locations






def main(dataset_path):
    # Read dataset
    locations = read_csv_dataset(dataset_path)
    dists = compute_distance_matrix(locations)
    cost, path = held_karp(dists)
    print("Total Cost: ", cost)
    print("Path: ", path + [0])
    
    



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("ERROR!! Dataset Path not provided!!")
        sys.exit(1)
    dataset_path = sys.argv[1]
    main(dataset_path)