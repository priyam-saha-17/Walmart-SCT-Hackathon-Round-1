import sys
import csv
import math
import numpy as np
import itertools
import random



def nearest_neighbour(dists):
    n = len(dists)
    start= 0
    path=[start]
    cost=0

    for i in range(n-1):

        m=next(x for x in list(range(0,n)) if x not in path)

        for j in range(n):
            if j not in path:
                if dists[start][j] < dists[start][m]:
                    m=j

        cost+=dists[start][m]
        path.append(m)
        start = m


    path.append(0)
    cost+=dists[start][path[0]]

    return (cost, path)










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
    cost, path = nearest_neighbour(dists)
    print("Total Cost: ", cost)
    print("Path: ", path)
    
    



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("ERROR!! Dataset Path not provided!!")
        sys.exit(1)
    dataset_path = sys.argv[1]
    main(dataset_path)