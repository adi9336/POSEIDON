import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional
import argparse

class CSVVisualizer:
    def __init__(self, data_dir: str = None):
        """Initialize the visualizer with a data directory."""
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent / 'data'
        else:
            self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        self.df_dict = {}
        self.load_data()
    
    def load_data(self):
        """Load all CSV files from the data directory."""
        csv_files = list(self.data_dir.glob('*.csv'))
        if not csv_files:
            print(f"No CSV files found in {self.data_dir}")
            return
        
        print(f"Found {len(csv_files)} CSV files:")
        for i, file in enumerate(csv_files, 1):
            try:
                df = pd.read_csv(file)
                self.df_dict[file.stem] = df
                print(f"{i}. {file.name}: {len(df)} rows x {len(df.columns)} columns")
                print(f"   Columns: {', '.join(df.columns)}\n")
            except Exception as e:
                print(f"Error loading {file.name}: {str(e)}")
    
    def plot_time_series(self, file_key: str, x_col: str = None, y_cols: List[str] = None, 
                        title: str = None, figsize: tuple = (12, 6)):
        """Plot time series data from a loaded CSV file."""
        if not self.df_dict:
            print("No data loaded. Cannot create plot.")
            return
            
        if file_key not in self.df_dict:
            print(f"File key '{file_key}' not found. Available keys: {list(self.df_dict.keys())}")
            return
            
        df = self.df_dict[file_key]
        
        # Set default columns if not provided
        if x_col is None:
            x_col = df.columns[0]  # Use first column as x-axis by default
        
        if y_cols is None:
            y_cols = [col for col in df.columns if col != x_col]
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        for y_col in y_cols:
            if y_col in df.columns:
                plt.plot(df[x_col], df[y_col], marker='o', linestyle='-', label=y_col)
        
        plt.title(title or f"Time Series: {file_key}")
        plt.xlabel(x_col)
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_ocean_profiles(self, file_key: str):
        """Plot oceanographic profiles for ARGO float data."""
        if not self.df_dict:
            print("No data loaded. Cannot create plot.")
            return
            
        if file_key not in self.df_dict:
            print(f"File key '{file_key}' not found. Available keys: {list(self.df_dict.keys())}")
            return
            
        df = self.df_dict[file_key]
        
        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Temperature vs Pressure (oceanographic profile)
        ax1.plot(df['temp'], df['pres'], 'r-')
        ax1.invert_yaxis()  # Invert y-axis for depth/pressure
        ax1.grid(True)
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Pressure (dbar)')
        ax1.set_title('Temperature Profile')
        
        # Salinity vs Pressure
        ax2.plot(df['psal'], df['pres'], 'b-')
        ax2.invert_yaxis()  # Invert y-axis for depth/pressure
        ax2.grid(True)
        ax2.set_xlabel('Practical Salinity (PSU)')
        ax2.set_ylabel('Pressure (dbar)')
        ax2.set_title('Salinity Profile')
        
        # Add a title for the entire figure
        plt.suptitle(f'ARGO Float Profiles\nLat: {df["latitude"].iloc[0]:.4f}°, Lon: {df["longitude"].iloc[0]:.4f}°', 
                    fontsize=14, y=1.02)
        
        plt.tight_layout()
        
        # Create a second figure for the temperature-salinity diagram
        plt.figure(figsize=(10, 8))
        plt.scatter(df['psal'], df['temp'], c=df['pres'], cmap='viridis_r')
        plt.colorbar(label='Pressure (dbar)')
        plt.xlabel('Practical Salinity (PSU)')
        plt.ylabel('Temperature (°C)')
        plt.title('Temperature-Salinity Diagram')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Create a map of the float's position
        plt.figure(figsize=(10, 8))
        plt.scatter(df['longitude'], df['latitude'], c='red', marker='o', s=100, 
                   label=f'Float {df["platform_number"].iloc[0]}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('ARGO Float Position')
        plt.grid(True)
        plt.legend()
        
        # Add a small buffer around the point
        lat_buffer = 0.1
        lon_buffer = 0.1
        plt.xlim(df['longitude'].iloc[0] - lon_buffer, df['longitude'].iloc[0] + lon_buffer)
        plt.ylim(df['latitude'].iloc[0] - lat_buffer, df['latitude'].iloc[0] + lat_buffer)
        
        plt.tight_layout()
        plt.show()
    
    def show_summary_statistics(self, file_key: str):
        """Display summary statistics for a loaded CSV file."""
        if file_key not in self.df_dict:
            print(f"File key '{file_key}' not found. Available keys: {list(self.df_dict.keys())}")
            return
            
        df = self.df_dict[file_key]
        print(f"\nSummary statistics for {file_key}:")
        print("-" * 50)
        print(df.describe())
        print("\nFirst 5 rows:")
        print(df.head())

def main():
    parser = argparse.ArgumentParser(description='Visualize CSV data from the data directory.')
    parser.add_argument('--dir', type=str, help='Path to the data directory')
    args = parser.parse_args()
    
    try:
        visualizer = CSVVisualizer(args.dir)
        
        if not visualizer.df_dict:
            print("No data loaded. Exiting.")
            return
            
        # Example usage with the first available file
        file_key = list(visualizer.df_dict.keys())[0]
        print(f"\nVisualizing data from: {file_key}")
        
        # Show summary statistics
        visualizer.show_summary_statistics(file_key)
        
        # Create oceanographic profile plots
        visualizer.plot_ocean_profiles(file_key)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()