import os
import pandas as pd
import glob
import numpy as np

def calculate_average_episode_length(csv_path):
    """Calculate the average episode length from the last 20 rows of a monitor.csv file."""
    try:
        # Read the CSV file, skipping the first row which contains metadata
        df = pd.read_csv(csv_path, skiprows=1)
        # Get only the last 20 rows
        df = df.tail(20)
        if 'l' in df.columns:
            return df['l'].mean()
        else:
            print(f"Warning: 'l' column not found in {csv_path}")
            return np.nan
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return np.nan

def process_directories(folder_path):
    """Process all directories and compute the episodic length ratio."""
    # Find all combined results files
    search_pattern = os.path.join(folder_path, "results.csv")
    combined_results_files = glob.glob(search_pattern)
    
    for combined_file in combined_results_files:
        print(f"Processing {combined_file}...")
        try:
            results_df = pd.read_csv(combined_file)
            
            # Initialize new column
            results_df['epilen_ratio'] = np.nan
            
            # Process each row/experiment
            for idx, row in results_df.iterrows():
                
                # Locate corresponding train and test monitor files
                train_monitor = f"{folder_path}/run_{idx}/train_monitor.monitor.csv"
                test_monitor = f"{folder_path}/run_{idx}/test_monitor.monitor.csv"
                
                if os.path.exists(train_monitor) and os.path.exists(test_monitor):
                    train_len = calculate_average_episode_length(train_monitor)
                    test_len = calculate_average_episode_length(test_monitor)
                    
                    if not np.isnan(train_len) and train_len > 0:
                        ratio = test_len / train_len
                        results_df.at[idx, 'epilen_ratio'] = ratio
                        print(f"test_len={test_len:.2f}, train_len={train_len:.2f}, ratio={ratio:.4f}")
                    else:
                        print(f"Invalid train length")
                else:
                    missing = []
                    if not os.path.exists(train_monitor):
                        missing.append(train_monitor)
                    if not os.path.exists(test_monitor):
                        missing.append(test_monitor)
                    print(f"Missing files: {', '.join(missing)}")
            
            # Save results with the new column
            output_file = combined_file.replace('.csv', '_with_epilen.csv')
            results_df.to_csv(output_file, index=False)
            print(f"Saved results to {output_file}")
        
        except Exception as e:
            print(f"Error processing {combined_file}: {e}")

if __name__ == "__main__":
    print("Starting episodic length ratio calculation...")
    process_directories("analysis/SAC/Walker2d-v5")
    # analysis/SAC/HalfCheetah-v5, analysis/SAC/Hopper-v5, 
    # analysis/SAC/InvertedPendulum-v5, analysis/SAC/Walker2d-v5
    print("Done!")