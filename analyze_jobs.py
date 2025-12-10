#!/usr/bin/env python3
"""
Analyze cumulative job postings by week across different years.
"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from pathlib import Path
from datetime import datetime
import glob
import re
from collections import defaultdict

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

    # Get reversed viridis color palette from seaborn
    n_years = len(cumulative_data)
    colors = sns.color_palette("viridis", n_colors=n_years).as_hex()
    colors = colors[::-1]  # Reverse the palette

    # Define line styles (dash patterns)
    line_styles = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']

    # Add trace for each year
    for i, (year, data) in enumerate(sorted(cumulative_data.items())):
        calendar_weeks = data['week'] + 30
        # Always use solid line for 2025, cycle through styles for other years
        line_style = 'solid' if year == 2025 else line_styles[i % len(line_styles)]
        # Make 2025 line thicker
        line_width = 4 if year == 2025 else 2

        fig.add_trace(go.Scatter(
            x=calendar_weeks,
            y=data['cumulative'],
            mode='lines',
            name=str(year),
            line=dict(
                width=line_width,
                color=colors[i],
                dash=line_style
            ),
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

    # Get reversed viridis color palette from seaborn
    n_years = len(rolling_data)
    colors = sns.color_palette("viridis", n_colors=n_years).as_hex()
    colors = colors[::-1]  # Reverse the palette

    # Define line styles (dash patterns)
    line_styles = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']

    # Add trace for each year
    for i, (year, data) in enumerate(sorted(rolling_data.items())):
        calendar_weeks = data['week'] + 30
        # Always use solid line for 2025, cycle through styles for other years
        line_style = 'solid' if year == 2025 else line_styles[i % len(line_styles)]
        # Make 2025 line thicker
        line_width = 4 if year == 2025 else 2

        fig.add_trace(go.Scatter(
            x=calendar_weeks,
            y=data['rolling_4wk'],
            mode='lines',
            name=str(year),
            line=dict(
                width=line_width,
                color=colors[i],
                dash=line_style
            ),
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

def filter_finance_jobs(df):
    """Filter dataframe for finance-related jobs based on JEL codes."""
    # Filter for jobs with "G - Financial Economics" in JEL codes
    finance_mask = df['JEL_Classifications'].fillna('').str.contains('G - Financial Economics', case=False, na=False)
    finance_df = df[finance_mask].copy()

    print(f"\nFiltered for finance jobs: {len(finance_df)} out of {len(df)} total postings")
    return finance_df

def filter_fed_regulator_jobs(df):
    """Filter dataframe for Federal Reserve and bank regulator jobs."""
    # Keywords to match Federal Reserve and regulators
    fed_patterns = [
        'Federal Reserve Bank',
        'Federal Reserve Board',
        'Federal Reserve System',
        'Board of Governors',
        'Federal Deposit Insurance Corporation',
        'Office of the Comptroller of the Currency',
    ]

    # Create combined pattern
    pattern = '|'.join(fed_patterns)

    # Filter institutions
    fed_mask = df['jp_institution'].fillna('').str.contains(pattern, case=False, na=False, regex=True)

    fed_df = df[fed_mask].copy()

    print(f"\nFiltered for Fed/Regulator jobs: {len(fed_df)} out of {len(df)} total postings")
    print(f"\nTop institutions:")
    print(fed_df['jp_institution'].value_counts().head(15))

    return fed_df

def split_us_non_us_jobs(df):
    """Split dataframe into US and non-US jobs based on location."""
    # Check if locations contain "UNITED STATES"
    us_mask = df['locations'].fillna('').str.contains('UNITED STATES', case=False, na=False)

    us_df = df[us_mask].copy()
    non_us_df = df[~us_mask].copy()

    print(f"\nUS jobs: {len(us_df)} ({len(us_df)/len(df)*100:.1f}%)")
    print(f"Non-US jobs: {len(non_us_df)} ({len(non_us_df)/len(df)*100:.1f}%)")

    return us_df, non_us_df

def parse_jel_codes(jel_string):
    """Parse JEL codes from a classification string.

    Returns a set of JEL codes in format like 'C', 'C1', 'C01', 'G', 'G0', etc.
    Handles formats like:
    - 'G - Financial Economics' -> 'G'
    - 'C1 - Econometric and Statistical Methods' -> 'C1'
    - 'C01 - Econometrics' -> 'C01'
    - '00 - 00 - Default: Any Field' -> '00' (skipped as it's a default)
    """
    if pd.isna(jel_string) or not jel_string:
        return set()

    codes = set()
    # Split by newlines to get individual classifications
    lines = str(jel_string).split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Match pattern: letter optionally followed by 1-2 digits, then space-dash-space
        # Examples: "G - ", "C1 - ", "J0 - ", "C01 - "
        match = re.match(r'^([A-Z][0-9]{0,2})\s*-\s*', line)
        if match:
            code = match.group(1)
            # Skip the default "00" code
            if code != '00':
                codes.add(code)

    return codes


def get_all_jel_categories(df):
    """Get all unique JEL categories (single letters) from the dataframe."""
    all_categories = set()

    for jel_string in df['JEL_Classifications'].dropna():
        codes = parse_jel_codes(jel_string)
        for code in codes:
            # Extract just the letter (first character)
            all_categories.add(code[0])

    return sorted(all_categories)


def get_jel_category_labels():
    """Return a dictionary mapping JEL category letters to their labels."""
    return {
        'A': 'General Economics and Teaching',
        'B': 'History of Economic Thought',
        'C': 'Mathematical and Quantitative Methods',
        'D': 'Microeconomics',
        'E': 'Macroeconomics and Monetary Economics',
        'F': 'International Economics',
        'G': 'Financial Economics',
        'H': 'Public Economics',
        'I': 'Health, Education, and Welfare',
        'J': 'Labor and Demographic Economics',
        'K': 'Law and Economics',
        'L': 'Industrial Organization',
        'M': 'Business Administration and Business Econ',
        'N': 'Economic History',
        'O': 'Economic Development and Growth',
        'P': 'Economic Systems',
        'Q': 'Agricultural and Natural Resource Econ',
        'R': 'Urban, Rural, Regional, Real Estate',
        'Y': 'Miscellaneous Categories',
        'Z': 'Other Special Topics',
        '0': 'Any Field / Unclassified',
    }


def filter_by_jel_category(df, category):
    """Filter dataframe for jobs matching a JEL category (single letter).

    A job matches if any of its JEL codes start with the given category letter.
    Special case: category '0' matches jobs with no specific JEL category
    (only "00 - Default: Any Field" or empty/null JEL field).
    """
    if category == '0':
        # Match jobs with no real JEL codes
        def has_no_category(jel_string):
            codes = parse_jel_codes(jel_string)
            return len(codes) == 0
        mask = df['JEL_Classifications'].apply(has_no_category)
    else:
        def has_category(jel_string):
            codes = parse_jel_codes(jel_string)
            return any(code.startswith(category) for code in codes)
        mask = df['JEL_Classifications'].apply(has_category)

    filtered_df = df[mask].copy()

    return filtered_df


def count_jobs_by_jel_category(df):
    """Count jobs by JEL category. Jobs with multiple categories are counted in each.

    Also counts jobs with no specific category under '0' (Any Field / Unclassified).
    """
    category_counts = defaultdict(int)

    for jel_string in df['JEL_Classifications']:
        codes = parse_jel_codes(jel_string)
        if len(codes) == 0:
            # Job has no specific JEL category
            category_counts['0'] += 1
        else:
            categories_seen = set()
            for code in codes:
                cat = code[0]
                if cat not in categories_seen:
                    category_counts[cat] += 1
                    categories_seen.add(cat)

    return dict(category_counts)


def plot_jel_category_comparison(cumulative_data_by_category, categories_to_plot,
                                  output_file='job_postings_by_jel.html', max_week=54):
    """Create interactive plot comparing job postings across JEL categories for current year."""
    fig = go.Figure()

    jel_labels = get_jel_category_labels()
    n_categories = len(categories_to_plot)
    # Use husl palette which provides good distinction for many categories
    colors = sns.color_palette("husl", n_colors=n_categories).as_hex()

    # Find the most recent year
    all_years = set()
    for cat_data in cumulative_data_by_category.values():
        all_years.update(cat_data.keys())
    current_year = max(all_years)

    for i, category in enumerate(categories_to_plot):
        if category not in cumulative_data_by_category:
            continue
        cat_data = cumulative_data_by_category[category]
        if current_year not in cat_data:
            continue

        data = cat_data[current_year]
        calendar_weeks = data['week'] + 30
        label = f"{category} - {jel_labels.get(category, 'Unknown')}"

        fig.add_trace(go.Scatter(
            x=calendar_weeks,
            y=data['cumulative'],
            mode='lines',
            name=label,
            line=dict(width=2, color=colors[i]),
            hovertemplate=f'<b>{label}</b><br>' +
                         'Week: %{x}<br>' +
                         'Cumulative Postings: %{y}<br>' +
                         '<extra></extra>'
        ))

    fig.update_layout(
        title=f'Cumulative Job Postings by JEL Category ({current_year})',
        xaxis_title='Calendar Week Number (Week 31 = August)',
        yaxis_title='Cumulative Job Postings',
        hovermode='closest',
        legend_title='JEL Category',
        template='plotly_white',
        width=900,
        height=700,
        xaxis=dict(range=[31, max_week]),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        )
    )

    fig.write_html(output_file)
    print(f"JEL category comparison plot saved to {output_file}")


def plot_jel_single_category_years(cumulative_data, category,
                                    output_file=None, max_week=54):
    """Create interactive plot for a single JEL category across years."""
    if output_file is None:
        output_file = f'job_postings_jel_{category}.html'

    fig = go.Figure()

    jel_labels = get_jel_category_labels()
    n_years = len(cumulative_data)
    colors = sns.color_palette("viridis", n_colors=n_years).as_hex()
    colors = colors[::-1]

    line_styles = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']

    for i, (year, data) in enumerate(sorted(cumulative_data.items())):
        calendar_weeks = data['week'] + 30
        line_style = 'solid' if year == 2025 else line_styles[i % len(line_styles)]
        line_width = 4 if year == 2025 else 2

        fig.add_trace(go.Scatter(
            x=calendar_weeks,
            y=data['cumulative'],
            mode='lines',
            name=str(year),
            line=dict(
                width=line_width,
                color=colors[i],
                dash=line_style
            ),
            hovertemplate='<b>Year %{fullData.name}</b><br>' +
                         'Week: %{x}<br>' +
                         'Cumulative Postings: %{y}<br>' +
                         '<extra></extra>'
        ))

    category_label = jel_labels.get(category, 'Unknown')
    fig.update_layout(
        title=f'Cumulative Job Postings: {category} - {category_label}',
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
    print(f"JEL {category} plot saved to {output_file}")


def plot_us_vs_non_us_comparison(cumulative_data_us, cumulative_data_non_us,
                                  output_file='job_postings_us_vs_non_us.html', max_week=54):
    """Create interactive plot with separate subplots for US and non-US job postings."""
    from plotly.subplots import make_subplots

    # Create subplots with 2 rows
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('US Job Postings', 'Non-US Job Postings'),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5]
    )

    # Get reversed viridis color palette
    n_years = len(cumulative_data_us)
    colors = sns.color_palette("viridis", n_colors=n_years).as_hex()
    colors = colors[::-1]

    # Define line styles
    line_styles = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']

    # Add US traces to first subplot
    for i, (year, data) in enumerate(sorted(cumulative_data_us.items())):
        calendar_weeks = data['week'] + 30
        line_style = 'solid' if year == 2025 else line_styles[i % len(line_styles)]
        line_width = 4 if year == 2025 else 2

        fig.add_trace(go.Scatter(
            x=calendar_weeks,
            y=data['cumulative'],
            mode='lines',
            name=str(year),
            line=dict(
                width=line_width,
                color=colors[i],
                dash=line_style
            ),
            hovertemplate='<b>Year %{fullData.name}</b><br>' +
                         'Week: %{x}<br>' +
                         'Cumulative US Postings: %{y}<br>' +
                         '<extra></extra>',
            legendgroup=str(year),
            showlegend=True
        ), row=1, col=1)

    # Add non-US traces to second subplot
    for i, (year, data) in enumerate(sorted(cumulative_data_non_us.items())):
        calendar_weeks = data['week'] + 30
        line_style = 'solid' if year == 2025 else line_styles[i % len(line_styles)]
        line_width = 4 if year == 2025 else 2

        fig.add_trace(go.Scatter(
            x=calendar_weeks,
            y=data['cumulative'],
            mode='lines',
            name=str(year),
            line=dict(
                width=line_width,
                color=colors[i],
                dash=line_style
            ),
            hovertemplate='<b>Year %{fullData.name}</b><br>' +
                         'Week: %{x}<br>' +
                         'Cumulative Non-US Postings: %{y}<br>' +
                         '<extra></extra>',
            legendgroup=str(year),
            showlegend=False  # Don't duplicate legend
        ), row=2, col=1)

    # Update layout
    fig.update_xaxes(title_text='Calendar Week Number (Week 31 = August)', range=[31, max_week], row=2, col=1)
    fig.update_xaxes(range=[31, max_week], row=1, col=1)
    fig.update_yaxes(title_text='Cumulative Job Postings', row=1, col=1)
    fig.update_yaxes(title_text='Cumulative Job Postings', row=2, col=1)

    fig.update_layout(
        title='Cumulative Job Postings: US vs Non-US (Comparison Across Years)',
        hovermode='closest',
        legend_title='Academic Year',
        template='plotly_white',
        width=650,
        height=1200,
    )

    fig.write_html(output_file)
    print(f"US vs Non-US comparison plot saved to {output_file}")

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

    # Now do the same for finance jobs only
    print("\n" + "="*60)
    print("FINANCE JOBS ANALYSIS")
    print("="*60)

    finance_df = filter_finance_jobs(df)

    # Calculate cumulative postings by week for finance jobs
    cumulative_data_finance = calculate_cumulative_by_week(finance_df)

    # Print summary statistics
    print_summary_statistics(cumulative_data_finance)

    # Plot the cumulative data (static)
    plot_cumulative_by_week(cumulative_data_finance, output_file='job_postings_by_week_finance.png')

    # Calculate rolling 4-week flow
    rolling_data_finance = calculate_rolling_four_week(cumulative_data_finance)

    # Plot the rolling 4-week flow (static)
    plot_rolling_four_week(rolling_data_finance, output_file='job_postings_rolling_4wk_finance.png')

    # Create interactive HTML plots
    print("\nCreating interactive plots for finance jobs...")
    plot_cumulative_interactive(cumulative_data_finance, output_file='job_postings_by_week_finance.html')
    plot_rolling_interactive(rolling_data_finance, output_file='job_postings_rolling_4wk_finance.html')

    # Now do the same for Fed/Regulator jobs only
    print("\n" + "="*60)
    print("FEDERAL RESERVE & BANK REGULATOR JOBS ANALYSIS")
    print("="*60)

    fed_df = filter_fed_regulator_jobs(df)

    # Calculate cumulative postings by week for Fed/regulator jobs
    cumulative_data_fed = calculate_cumulative_by_week(fed_df)

    # Print summary statistics
    print_summary_statistics(cumulative_data_fed)

    # Plot the cumulative data (static)
    plot_cumulative_by_week(cumulative_data_fed, output_file='job_postings_by_week_fed.png')

    # Calculate rolling 4-week flow
    rolling_data_fed = calculate_rolling_four_week(cumulative_data_fed)

    # Plot the rolling 4-week flow (static)
    plot_rolling_four_week(rolling_data_fed, output_file='job_postings_rolling_4wk_fed.png')

    # Create interactive HTML plots
    print("\nCreating interactive plots for Fed/regulator jobs...")
    plot_cumulative_interactive(cumulative_data_fed, output_file='job_postings_by_week_fed.html')
    plot_rolling_interactive(rolling_data_fed, output_file='job_postings_rolling_4wk_fed.html')

    # Now analyze US vs Non-US job postings
    print("\n" + "="*60)
    print("US vs NON-US JOBS ANALYSIS")
    print("="*60)

    us_df, non_us_df = split_us_non_us_jobs(df)

    # Calculate cumulative postings for US jobs
    cumulative_data_us = calculate_cumulative_by_week(us_df)
    print("\n--- US Jobs Summary ---")
    print_summary_statistics(cumulative_data_us)

    # Calculate cumulative postings for non-US jobs
    cumulative_data_non_us = calculate_cumulative_by_week(non_us_df)
    print("\n--- Non-US Jobs Summary ---")
    print_summary_statistics(cumulative_data_non_us)

    # Create comparison plot
    print("\nCreating US vs Non-US comparison plot...")
    plot_us_vs_non_us_comparison(cumulative_data_us, cumulative_data_non_us)

    # Analyze by JEL classification
    print("\n" + "="*60)
    print("JEL CLASSIFICATION ANALYSIS")
    print("="*60)

    # Get all JEL categories present in the data
    jel_categories = get_all_jel_categories(df)
    jel_labels = get_jel_category_labels()
    print(f"\nFound {len(jel_categories)} JEL categories: {', '.join(jel_categories)}")

    # Count jobs by category
    category_counts = count_jobs_by_jel_category(df)
    print("\nJob counts by JEL category (jobs may be counted in multiple categories):")
    for cat in sorted(category_counts.keys(), key=lambda x: category_counts[x], reverse=True):
        label = jel_labels.get(cat, 'Unknown')
        print(f"  {cat} - {label}: {category_counts[cat]}")

    # Calculate cumulative data for each JEL category (including '0' for unclassified)
    cumulative_by_jel = {}
    all_categories = jel_categories + ['0']  # Add unclassified category
    for category in all_categories:
        cat_df = filter_by_jel_category(df, category)
        if len(cat_df) > 0:
            cumulative_by_jel[category] = calculate_cumulative_by_week(cat_df)
            print(f"\n{category} - {jel_labels.get(category, 'Unknown')}: {len(cat_df)} postings")

    # Create comparison plot for all categories in current year
    print("\nCreating JEL category comparison plot...")
    all_categories_sorted = sorted(category_counts.keys(), key=lambda x: category_counts[x], reverse=True)
    plot_jel_category_comparison(cumulative_by_jel, all_categories_sorted)

    # Create individual year-over-year plots for major categories (including unclassified)
    major_categories = ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N','O', 'P', 'Q', 'R', 'Y', 'Z']
    print("\nCreating individual JEL category plots (year-over-year)...")
    for category in major_categories:
        if category in cumulative_by_jel:
            plot_jel_single_category_years(cumulative_by_jel[category], category)

    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()
