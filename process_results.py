import pandas as pd
import glob
import os

def merge_folder_csvs(folder_path, output_filename='combined.csv'):
    """Combine all CSV files in a folder (alphabetically sorted)."""
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return
    
    print(f"Found {len(csv_files)} files")
    dfs = []
    first_columns = None
    
    for file in csv_files:
        df = pd.read_csv(file, index_col=0)
        
        if first_columns is None:
            first_columns = df.columns.tolist()
        elif df.columns.tolist() != first_columns:
            raise ValueError(f"Column mismatch in {file}")
        
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=False)
    output_path = os.path.join(folder_path, output_filename)
    combined_df.to_csv(output_path)
    
    print(f"Combined into {output_path} ({len(combined_df)} rows)")
    return combined_df

def merge_csv_list(csv_files, output_path='combined.csv'):
    """Combine specific CSV files in given order."""
    dfs = []
    first_columns = None
    
    for file in csv_files:
        if not os.path.exists(file):
            print(f"WARNING: File not found: {file}")
            continue
        
        df = pd.read_csv(file, index_col=0)
        
        if first_columns is None:
            first_columns = df.columns.tolist()
        elif df.columns.tolist() != first_columns:
            raise ValueError(f"Column mismatch in {file}")
        
        dfs.append(df)
    
    if not dfs:
        return None
    
    combined_df = pd.concat(dfs, ignore_index=False)
    combined_df.to_csv(output_path)
    print(f"Combined {len(dfs)} files into {output_path} ({len(combined_df)} rows)")
    return combined_df

def merge_with_source_tags(file_dict, output_path='combined.csv'):
    """Combine CSVs with source labels from dictionary keys."""
    dfs = []
    
    for source_name, file in file_dict.items():
        if not os.path.exists(file):
            print(f"WARNING: File not found: {file}")
            continue
        
        df = pd.read_csv(file)
        df.insert(0, 'source', source_name)
        dfs.append(df)
    
    if not dfs:
        return None
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(output_path, index=False)
    print(f"Combined {len(dfs)} files into {output_path} ({len(combined_df)} rows)")
    return combined_df

def merge_with_algorithm_codes(csv_files, algorithm_codes, output_path='combined.csv'):
    """Combine CSVs with algorithm codes, organizing columns by type."""
    if len(algorithm_codes) != len(csv_files):
        raise ValueError("Number of codes must match number of files")
    
    data_keywords = ['reward', 'seed', 'gap']
    config_cols, data_cols, seen = [], [], set()
    
    def is_data_col(name):
        return any(kw in name.lower() for kw in data_keywords)
    
    dfs = []
    for file, code in zip(csv_files, algorithm_codes):
        if not os.path.exists(file):
            continue
        
        df = pd.read_csv(file)
        
        for col in df.columns:
            if col not in seen and col != 'algorithm':
                (data_cols if is_data_col(col) else config_cols).append(col)
                seen.add(col)
        
        df['algorithm'] = code
        dfs.append(df)
    
    if not dfs:
        return None
    
    # Reorder: algorithm, config columns, data columns
    all_cols = ['algorithm'] + config_cols + data_cols
    aligned_dfs = [df.reindex(columns=all_cols) for df in dfs]
    
    combined_df = pd.concat(aligned_dfs, ignore_index=True)
    combined_df.to_csv(output_path, index=False)
    print(f"Combined {len(dfs)} files into {output_path} ({len(combined_df)} rows)")
    return combined_df

def recompute_gap(csv_file, output_file=None):
    """Recompute generalization_gap as train_reward - test_reward."""
    df = pd.read_csv(csv_file, index_col=0)
    
    if 'generalization_gap' in df.columns:
        df = df.drop('generalization_gap', axis=1)
    
    df['generalization_gap'] = df['train_reward'] - df['test_reward']
    
    output_file = output_file or csv_file
    df.to_csv(output_file)
    print(f"Saved to: {output_file}")
    return df

def average_by_config(csv_file, config_col_indices=[0, 1, 2, 3]):
    """Average results grouped by configuration columns."""
    df = pd.read_csv(csv_file)
    config_cols = df.columns[config_col_indices].tolist()
    
    seed_keywords = ['seed', 'repetition']
    output_cols = [
        col for col in df.columns 
        if col not in config_cols and not any(kw in col.lower() for kw in seed_keywords)
    ]
    
    result = df.groupby(config_cols, sort=False, dropna=False)[output_cols].mean().reset_index()
    result['n_seeds'] = df.groupby(config_cols, sort=False, dropna=False).size().values
    
    output_file = csv_file.replace('.csv', '_averaged.csv')
    result.to_csv(output_file, index=False)
    print(f"Averaged {len(df)} → {len(result)} rows, saved to {output_file}")
    return result

# Usage examples
if __name__ == "__main__":
    # Example 1: Merge all CSVs in a folder
    # merge_folder_csvs("./results/DDPG/")
    
    # Example 2: Merge specific files with algorithm codes
    # merge_with_algorithm_codes(
    #     ["./results/PPO/env1.csv", "./results/A2C/env1.csv"],
    #     algorithm_codes=[0, 1],
    #     output_path="./combined.csv"
    # )
    
    # Example 3: Merge and tag with source names
    merge_with_source_tags({
        "HalfCheetah-v5": "./results/combined_HalfCheetah-v5_averaged.csv",
        "Hopper-v5": "./results/combined_Hopper-v5_averaged.csv",
        "Walker2d-v5": "./results/combined_Walker2d-v5_averaged.csv"
    }, output_path="./results/combined_all.csv")