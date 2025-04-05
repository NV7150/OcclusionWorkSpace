import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import csv

matplotlib.use('Qt5Agg') 

def visualize_depth(file_path):
    """
    Loads depth data from a .csv file and visualizes it as a grayscale image.
    """
    try:
        depth_data = []
        with open(file_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                depth_data.append([float(x) for x in row])
        depth_data = np.array(depth_data)
        print(f"Shape of depth data: {depth_data.shape}")

        # Normalize depth data to 0-255 range
        min_depth = np.min(depth_data)
        max_depth = np.max(depth_data)
        depth_data = 255 * (depth_data - min_depth) / (max_depth - min_depth)
        depth_data = depth_data.astype(np.uint8)

        plt.imshow(depth_data, cmap='gray')
        plt.colorbar()  # Add a colorbar to show depth values
        plt.title(f"Depth Map - {os.path.basename(file_path)}")
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Specify the directory containing the .csv files
    data_directory = "LocalData/RecordedData 18/"

    # Get a list of all .csv files in the directory
    csv_files = [f for f in os.listdir(data_directory) if f.endswith(".csv")]

    if not csv_files:
        print("No .csv files found in the specified directory.")
    else:
        # Iterate through all .csv files and visualize them
        for csv_file in csv_files:
            if csv_file.startswith("depth_"):
                file_path = os.path.join(data_directory, csv_file)
                visualize_depth(file_path)