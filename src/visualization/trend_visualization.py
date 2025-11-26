import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.utils.file_operations import read_csv

def load_topic_data(csv_path):
    """Load topic frequency data"""
    df = read_csv(csv_path)
    df['发表日期'] = pd.to_datetime(df['发表日期'])
    return df

def plot_trend(df, output_path):
    """Plot time trend"""
    plt.figure(figsize=(15, 8))
    ax = df.plot(kind='line', figsize=(15,8))
    
    # Customize plot with English labels
    ax.set_title('Topic Trend Over Time', fontsize=14)  
    ax.set_xlabel('Date', fontsize=12)               
    ax.set_ylabel('Frequency', fontsize=12)          
    
    # Format x-axis with English format
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    # Save figure with English filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Trend plot saved to {output_path}")

def main():
    """Main visualization workflow"""
    df = load_topic_data("results/csv/topic_frequency.csv")
    plot_trend(df, "results/plots/topic_frequency_time_trend.png")  

if __name__ == "__main__":
    main()