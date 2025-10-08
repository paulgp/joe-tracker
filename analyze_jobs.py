#!/usr/bin/env python3
"""
Analyze cumulative job postings by week across different years.
"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import glob

def parse_excel_files(data_folder='data'):
    """Parse all Excel files in the data folder and combine them."""
    excel_files = glob.glob(f'{data_folder}/*.xlsx')
    print(f"Found {len(excel_files)} Excel files")

    all_data = []
    for file in excel_files:
        try:
            df = pd.read_excel(file)
            all_data.append(df)
            print(f"Loaded {file}: {len(df)} rows")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal rows: {len(combined_df)}")

    return combined_df

def calculate_cumulative_by_week(df, start_week=31):
    """Calculate cumulative job postings by week for each year.

    Args:
        df: DataFrame with job posting data
        start_week: The week number to start from (default 31 for August)
    """
    # Convert Date_Active to datetime
    df['Date_Active'] = pd.to_datetime(df['Date_Active'], errors='coerce')

    # Drop rows with invalid dates
    df = df.dropna(subset=['Date_Active'])

    # Extract year and week number
    df['year'] = df['Date_Active'].dt.year
    df['week'] = df['Date_Active'].dt.isocalendar().week

    # Adjust week numbers to start from start_week (e.g., week 31)
    # Weeks >= start_week stay in current year, weeks < start_week are shifted to next year
    df['adjusted_week'] = df['week'].apply(
        lambda w: w - start_week + 1 if w >= start_week else w + (52 - start_week + 1)
    )

    # Adjust year: if week < start_week, it belongs to previous academic year
    df['academic_year'] = df.apply(
        lambda row: row['year'] - 1 if row['week'] < start_week else row['year'],
        axis=1
    )

    # Group by academic year and adjusted week, count postings
    weekly_counts = df.groupby(['academic_year', 'adjusted_week']).size().reset_index(name='count')
    weekly_counts.columns = ['year', 'week', 'count']  # Rename for consistency

    # Calculate cumulative sum for each year
    cumulative_data = {}
    for year in weekly_counts['year'].unique():
        year_data = weekly_counts[weekly_counts['year'] == year].copy()
        year_data = year_data.sort_values('week')
        year_data['cumulative'] = year_data['count'].cumsum()
        cumulative_data[year] = year_data

    return cumulative_data

def calculate_rolling_four_week(cumulative_data):
    """Calculate rolling 4-week flow of new postings from cumulative data."""
    rolling_data = {}

    for year, data in cumulative_data.items():
        year_data = data.copy()
        # Calculate new postings each week (difference in cumulative)
        year_data['new_postings'] = year_data['count']

        # Calculate rolling 4-week sum
        year_data['rolling_4wk'] = year_data['new_postings'].rolling(window=4, min_periods=1).sum()
        rolling_data[year] = year_data

    return rolling_data

def plot_cumulative_by_week(cumulative_data, output_file='job_postings_by_week.png', max_week=54):
    """Plot cumulative job postings by week for each year."""
    plt.figure(figsize=(14, 8))

    # Plot each year
    for year, data in sorted(cumulative_data.items()):
        # Shift weeks back to calendar weeks (add 30 to convert adjusted week to calendar week)
        calendar_weeks = data['week'] + 30
        plt.plot(calendar_weeks, data['cumulative'], label=str(year), linewidth=2)

    plt.xlabel('Calendar Week Number (Week 31 = August)', fontsize=12)
    plt.ylabel('Cumulative Job Postings', fontsize=12)
    plt.title('Cumulative Job Postings by Week (Comparison Across Years)', fontsize=14, fontweight='bold')
    plt.legend(title='Academic Year', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(31, max_week)  # Set x-axis limit from week 31 to 54
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")
    plt.close()

def plot_rolling_four_week(rolling_data, output_file='job_postings_rolling_4wk.png', max_week=54):
    """Plot rolling 4-week flow of new job postings."""
    plt.figure(figsize=(14, 8))

    # Plot each year
    for year, data in sorted(rolling_data.items()):
        # Shift weeks back to calendar weeks (add 30 to convert adjusted week to calendar week)
        calendar_weeks = data['week'] + 30
        plt.plot(calendar_weeks, data['rolling_4wk'], label=str(year), linewidth=2)

    plt.xlabel('Calendar Week Number (Week 31 = August)', fontsize=12)
    plt.ylabel('Rolling 4-Week Job Postings', fontsize=12)
    plt.title('Rolling 4-Week Flow of New Job Postings (Comparison Across Years)', fontsize=14, fontweight='bold')
    plt.legend(title='Academic Year', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(31, max_week)  # Set x-axis limit from week 31 to 54
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()

def plot_cumulative_interactive(cumulative_data, output_file='job_postings_by_week.html', max_week=54):
    """Create interactive plot of cumulative job postings."""
    fig = go.Figure()

    # Add trace for each year
    for year, data in sorted(cumulative_data.items()):
        calendar_weeks = data['week'] + 30
        fig.add_trace(go.Scatter(
            x=calendar_weeks,
            y=data['cumulative'],
            mode='lines',
            name=str(year),
            line=dict(width=2),
            hovertemplate='<b>Year %{fullData.name}</b><br>' +
                         'Week: %{x}<br>' +
                         'Cumulative Postings: %{y}<br>' +
                         '<extra></extra>'
        ))

    fig.update_layout(
        title='Cumulative Job Postings by Week (Comparison Across Years)',
        xaxis_title='Calendar Week Number (Week 31 = August)',
        yaxis_title='Cumulative Job Postings',
        hovermode='closest',
        legend_title='Academic Year',
        template='plotly_white',
        width=650,
        height=850,
        xaxis=dict(range=[31, max_week])
    )

    fig.write_html(output_file)
    print(f"Interactive plot saved to {output_file}")

def plot_rolling_interactive(rolling_data, output_file='job_postings_rolling_4wk.html', max_week=54):
    """Create interactive plot of rolling 4-week flow."""
    fig = go.Figure()

    # Add trace for each year
    for year, data in sorted(rolling_data.items()):
        calendar_weeks = data['week'] + 30
        fig.add_trace(go.Scatter(
            x=calendar_weeks,
            y=data['rolling_4wk'],
            mode='lines',
            name=str(year),
            line=dict(width=2),
            hovertemplate='<b>Year %{fullData.name}</b><br>' +
                         'Week: %{x}<br>' +
                         'Rolling 4-Week Postings: %{y:.0f}<br>' +
                         '<extra></extra>'
        ))

    fig.update_layout(
        title='Rolling 4-Week Flow of New Job Postings (Comparison Across Years)',
        xaxis_title='Calendar Week Number (Week 31 = August)',
        yaxis_title='Rolling 4-Week Job Postings',
        hovermode='closest',
        legend_title='Academic Year',
        template='plotly_white',
        width=650,
        height=850,
        xaxis=dict(range=[31, max_week])
    )

    fig.write_html(output_file)
    print(f"Interactive plot saved to {output_file}")

def print_summary_statistics(cumulative_data):
    """Print summary statistics for each year."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    for year, data in sorted(cumulative_data.items()):
        total_postings = data['cumulative'].iloc[-1]
        weeks_active = len(data)
        avg_per_week = total_postings / weeks_active if weeks_active > 0 else 0

        print(f"\nYear {year}:")
        print(f"  Total postings: {total_postings}")
        print(f"  Weeks with activity: {weeks_active}")
        print(f"  Average per week: {avg_per_week:.2f}")
        print(f"  Week range: {data['week'].min()} - {data['week'].max()}")

def main():
    """Main function to orchestrate the analysis."""
    print("="*60)
    print("JOB POSTINGS ANALYSIS")
    print("="*60)

    # Parse Excel files
    df = parse_excel_files('data')

    # Calculate cumulative postings by week
    cumulative_data = calculate_cumulative_by_week(df)

    # Print summary statistics
    print_summary_statistics(cumulative_data)

    # Plot the cumulative data (static)
    plot_cumulative_by_week(cumulative_data)

    # Calculate rolling 4-week flow
    rolling_data = calculate_rolling_four_week(cumulative_data)

    # Plot the rolling 4-week flow (static)
    plot_rolling_four_week(rolling_data)

    # Create interactive HTML plots
    print("\nCreating interactive plots...")
    plot_cumulative_interactive(cumulative_data)
    plot_rolling_interactive(rolling_data)

    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()
