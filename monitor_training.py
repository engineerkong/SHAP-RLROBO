#!/usr/bin/env python
"""
Monitor training status for RL hyperparameter analysis jobs.
"""

import os
import argparse
import time
from datetime import datetime
import glob

def read_status_file(status_file):
    """Read and return the contents of a status file."""
    if not os.path.exists(status_file):
        return f"Status file not found: {status_file}"
    
    with open(status_file, 'r') as f:
        content = f.read()
    
    return content

def find_status_files(log_dir):
    """Find all status files in the given directory and subdirectories."""
    return glob.glob(os.path.join(log_dir, '**/status.txt'), recursive=True)

def monitor_single_job(status_file, follow=False, interval=5):
    """Monitor a single job."""
    if not os.path.exists(status_file):
        print(f"Status file not found: {status_file}")
        return
    
    print(f"Monitoring job status from: {status_file}")
    
    # Print current content
    print("\n" + "="*80)
    print(read_status_file(status_file))
    
    # If follow mode is enabled, keep checking for updates
    if follow:
        last_position = os.path.getsize(status_file)
        try:
            while True:
                time.sleep(interval)
                
                # Check if file exists (could be deleted)
                if not os.path.exists(status_file):
                    print(f"Status file no longer exists: {status_file}")
                    break
                
                # Check for new content
                current_size = os.path.getsize(status_file)
                if current_size > last_position:
                    with open(status_file, 'r') as f:
                        f.seek(last_position)
                        new_content = f.read()
                    
                    print(new_content, end='')
                    last_position = current_size
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
    
def monitor_all_jobs(log_dir, follow=False, interval=5):
    """Monitor all jobs in the given directory."""
    status_files = find_status_files(log_dir)
    
    if not status_files:
        print(f"No status files found in {log_dir}")
        return
    
    print(f"Found {len(status_files)} job(s) in {log_dir}")
    
    # Print summary of all jobs
    for i, status_file in enumerate(status_files):
        job_dir = os.path.dirname(status_file)
        job_name = os.path.basename(job_dir)
        
        # Read the last line to get current status
        with open(status_file, 'r') as f:
            lines = f.readlines()
            last_line = lines[-1].strip() if lines else "No status available"
        
        print(f"Job {i+1}: {job_name} - {last_line}")
    
    # If follow is enabled, let the user select which job to monitor
    if follow:
        try:
            choice = int(input("\nEnter job number to monitor (0 for all): "))
            
            if choice == 0:
                print("Monitoring all jobs...")
                last_positions = {file: os.path.getsize(file) for file in status_files}
                
                try:
                    while True:
                        time.sleep(interval)
                        for status_file in status_files:
                            job_dir = os.path.dirname(status_file)
                            job_name = os.path.basename(job_dir)
                            
                            # Check if file still exists
                            if not os.path.exists(status_file):
                                continue
                            
                            # Check for new content
                            current_size = os.path.getsize(status_file)
                            if current_size > last_positions[status_file]:
                                with open(status_file, 'r') as f:
                                    f.seek(last_positions[status_file])
                                    new_content = f.read()
                                
                                print(f"\n[{job_name}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:")
                                print(new_content.strip())
                                last_positions[status_file] = current_size
                except KeyboardInterrupt:
                    print("\nMonitoring stopped by user.")
            
            elif 1 <= choice <= len(status_files):
                monitor_single_job(status_files[choice-1], follow=True, interval=interval)
            else:
                print("Invalid choice.")
                
        except ValueError:
            print("Invalid input. Please enter a number.")
    else:
        # If not in follow mode, print detailed status for all jobs
        for i, status_file in enumerate(status_files):
            job_dir = os.path.dirname(status_file)
            job_name = os.path.basename(job_dir)
            
            print(f"\n{'='*40} Job {i+1}: {job_name} {'='*40}")
            print(read_status_file(status_file))

def main():
    parser = argparse.ArgumentParser(description="Monitor RL training progress")
    parser.add_argument("log_dir", type=str, help="Log directory to monitor")
    parser.add_argument("-f", "--follow", action="store_true", help="Follow logs in real-time")
    parser.add_argument("-i", "--interval", type=int, default=5, help="Update interval in seconds when following logs")
    parser.add_argument("-j", "--job", type=str, help="Path to specific status file to monitor")
    
    args = parser.parse_args()
    
    if args.job:
        monitor_single_job(args.job, follow=args.follow, interval=args.interval)
    else:
        monitor_all_jobs(args.log_dir, follow=args.follow, interval=args.interval)

if __name__ == "__main__":
    main()
