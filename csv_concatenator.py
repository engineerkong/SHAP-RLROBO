import pandas as pd
import os
import sys

def concatenate_csv_files(file_paths, output_path, keep_headers=True):
    """
    Concatenate multiple CSV files into a single CSV file.
    
    Parameters:
    - file_paths: List of paths to CSV files to concatenate
    - output_path: Path where the concatenated CSV will be saved
    - keep_headers: If True, keep headers only from the first file
    
    Returns:
    - True if successful, False otherwise
    """
    try:
        # Check if all files exist
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"Error: File {file_path} does not exist.")
                return False
        
        # Initialize an empty DataFrame to store the concatenated data
        concatenated_df = pd.DataFrame()
        
        # Process each file
        for i, file_path in enumerate(file_paths):
            print(f"Processing file: {file_path}")
            
            # Read the current CSV file
            current_df = pd.read_csv(file_path)
            
            # For all files except the first one, drop the header if specified
            if i > 0 and not keep_headers:
                current_df.columns = concatenated_df.columns
            
            # Concatenate with the main DataFrame
            concatenated_df = pd.concat([concatenated_df, current_df], ignore_index=True)
        
        # Save the concatenated DataFrame to a new CSV file
        concatenated_df.to_csv(output_path, index=False)
        print(f"Concatenation complete! Result saved to: {output_path}")
        print(f"Total rows in concatenated file: {len(concatenated_df)}")
        
        return True
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) < 6:
        print("Usage: python csv_concatenator.py file1.csv file2.csv file3.csv file4.csv output.csv")
        print("Or: python csv_concatenator.py file1.csv file2.csv file3.csv file4.csv output.csv --no-headers")
        return
    
    input_files = sys.argv[1:5]  # Get the 4 input file paths
    output_file = sys.argv[5]    # Get the output file path
    
    # Check if the --no-headers flag is provided
    keep_headers = True
    if len(sys.argv) > 6 and sys.argv[6] == "--no-headers":
        keep_headers = False
    
    # Call the concatenation function
    concatenate_csv_files(input_files, output_file, keep_headers)

if __name__ == "__main__":
    main()