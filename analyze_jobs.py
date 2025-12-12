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

def calculate_cumulative_by_week(df, start_week=31, interpolate_current=False):
    """Calculate cumulative job postings by week for each year.

    Args:
        df: DataFrame with job posting data
        start_week: The week number to start from (default 31 for August)
        interpolate_current: If True, interpolate the current week based on days elapsed
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

    # Interpolate current week if requested
    if interpolate_current:
        current_year = datetime.now().year
        if current_year in cumulative_data:
            # Get the current date and calculate day of week (0 = Monday, 6 = Sunday)
            now = datetime.now()
            current_week = now.isocalendar().week
            day_of_week = now.weekday()  # 0 = Monday

            # Calculate fraction of week completed (assuming week runs Monday-Sunday)
            days_elapsed = day_of_week + 1  # +1 because Monday is day 1, not day 0
            week_fraction = days_elapsed / 7.0

            # Find if current week exists in data
            year_data = cumulative_data[current_year]

            # Get the adjusted week for current calendar week
            if current_week >= start_week:
                adjusted_current_week = current_week - start_week + 1
                academic_year = current_year
            else:
                adjusted_current_week = current_week + (52 - start_week + 1)
                academic_year = current_year - 1

            # Only interpolate if we're in the right academic year
            if academic_year == current_year:
                current_week_data = year_data[year_data['week'] == adjusted_current_week]

                if not current_week_data.empty and week_fraction < 1.0:
                    # Get the current count for this week
                    current_count = current_week_data['count'].iloc[0]

                    # Interpolate what the full week count might be
                    interpolated_count = current_count / week_fraction if week_fraction > 0 else current_count
                    additional_count = interpolated_count - current_count

                    # Update the count and recalculate cumulative
                    year_data.loc[year_data['week'] == adjusted_current_week, 'count'] = interpolated_count
                    year_data['cumulative'] = year_data['count'].cumsum()
                    cumulative_data[current_year] = year_data

                    print(f"\nInterpolation applied to {current_year}, week {adjusted_current_week}:")
                    print(f"  Days elapsed in week: {days_elapsed}/7 ({week_fraction:.1%})")
                    print(f"  Current count: {current_count:.0f}")
                    print(f"  Interpolated full-week count: {interpolated_count:.1f}")
                    print(f"  Added: {additional_count:.1f} postings")

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

def plot_cumulative_by_ay(cumulative_data, output_file='job_postings_by_ay.png', selected_weeks=[35, 40, 45, 50, 55]):
    """Plot cumulative job postings by academic year for selected weeks.
    
    Args:
        cumulative_data: Dictionary mapping academic year to DataFrame with 'week' and 'cumulative' columns
        output_file: Output filename for the plot
        selected_weeks: List of calendar week numbers to plot (default: [30, 35, 40, 45, 50, 55])
    """
    plt.figure(figsize=(14, 8))
    
    # Convert calendar weeks to adjusted weeks
    # For weeks >= 31: adjusted_week = calendar_week - 30 (since adjusted week 1 = calendar week 31)
    # For week 30: adjusted_week = 52 (it's the last week of the academic year)
    week_to_adjusted = {}
    for cal_week in selected_weeks:
        if cal_week >= 31:
            week_to_adjusted[cal_week] = cal_week - 30
        else:
            # Week 30 is the last week of the academic year (adjusted week 52)
            week_to_adjusted[cal_week] = 52
    
    # Collect data: for each selected week, get cumulative value for each academic year
    plot_data = {}  # {calendar_week: {academic_year: cumulative_value}}
    
    for cal_week in selected_weeks:
        plot_data[cal_week] = {}
        adjusted_week = week_to_adjusted[cal_week]
        
        for year, data in sorted(cumulative_data.items()):
            week_data = data[data['week'] == adjusted_week]
            if not week_data.empty:
                plot_data[cal_week][year] = week_data['cumulative'].iloc[0]
    
    # Plot each week as a different colored line
    years = sorted(set().union(*[plot_data[week].keys() for week in selected_weeks]))
    
    for cal_week in selected_weeks:
        week_values = [plot_data[cal_week].get(year, None) for year in years]
        # Filter out None values and corresponding years
        valid_data = [(y, v) for y, v in zip(years, week_values) if v is not None]
        if valid_data:
            valid_years, valid_values = zip(*valid_data)
            plt.plot(valid_years, valid_values, label=f'Week {cal_week}', linewidth=2, marker='o')
    
    plt.xlabel('Academic Year', fontsize=12)
    plt.ylabel('Cumulative Job Postings', fontsize=12)
    plt.title('Cumulative Job Postings by Academic Year (Selected Weeks)', fontsize=14, fontweight='bold')
    plt.legend(title='Calendar Week', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()

def plot_cumulative_by_ay_interactive(cumulative_data, output_file='job_postings_by_ay.html', selected_weeks=[35, 40, 45, 50, 55]):
    """Create interactive plot of cumulative job postings by academic year for selected weeks.
    
    Args:
        cumulative_data: Dictionary mapping academic year to DataFrame with 'week' and 'cumulative' columns
        output_file: Output filename for the plot
        selected_weeks: List of calendar week numbers to plot (default: [35, 40, 45, 50, 55])
    """
    fig = go.Figure()
    
    # Convert calendar weeks to adjusted weeks
    week_to_adjusted = {}
    for cal_week in selected_weeks:
        if cal_week >= 31:
            week_to_adjusted[cal_week] = cal_week - 30
        else:
            week_to_adjusted[cal_week] = 52
    
    # Collect data: for each selected week, get cumulative value for each academic year
    plot_data = {}  # {calendar_week: {academic_year: cumulative_value}}
    
    for cal_week in selected_weeks:
        plot_data[cal_week] = {}
        adjusted_week = week_to_adjusted[cal_week]
        
        for year, data in sorted(cumulative_data.items()):
            week_data = data[data['week'] == adjusted_week]
            if not week_data.empty:
                plot_data[cal_week][year] = week_data['cumulative'].iloc[0]
    
    # Get color palette for the weeks
    n_weeks = len(selected_weeks)
    colors = sns.color_palette("viridis", n_colors=n_weeks).as_hex()
    
    # Plot each week as a different colored line
    years = sorted(set().union(*[plot_data[week].keys() for week in selected_weeks]))
    
    for i, cal_week in enumerate(selected_weeks):
        week_values = [plot_data[cal_week].get(year, None) for year in years]
        # Filter out None values and corresponding years
        valid_data = [(y, v) for y, v in zip(years, week_values) if v is not None]
        if valid_data:
            valid_years, valid_values = zip(*valid_data)
            fig.add_trace(go.Scatter(
                x=valid_years,
                y=valid_values,
                mode='lines+markers',
                name=f'Week {cal_week}',
                line=dict(
                    width=2,
                    color=colors[i]
                ),
                marker=dict(size=8),
                hovertemplate='<b>Week %{fullData.name}</b><br>' +
                             'Academic Year: %{x}<br>' +
                             'Cumulative Postings: %{y}<br>' +
                             '<extra></extra>'
            ))
    
    fig.update_layout(
        title='Cumulative Job Postings by Academic Year (Selected Weeks)',
        xaxis_title='Academic Year',
        yaxis_title='Cumulative Job Postings',
        hovermode='closest',
        legend_title='Calendar Week',
        template='plotly_white',
        width=650,
        height=850
    )
    
    fig.write_html(output_file)
    print(f"Interactive plot saved to {output_file}")

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

def filter_us_jobs(df):
    """Filter dataframe for US-based jobs."""
    us_mask = df['locations'].fillna('').str.startswith('UNITED STATES', na=False)
    us_df = df[us_mask].copy()

    print(f"\nFiltered for US jobs: {len(us_df)} out of {len(df)} total postings")
    return us_df

def extract_country(location):
    """Extract country name from location string."""
    if pd.isna(location) or location == '':
        return ''
    # Split by multiple spaces or newlines to separate country from city
    parts = location.split('  ') if '  ' in location else [location.split('\n')[0]]
    country_part = parts[0].strip()

    # Handle multi-word countries
    words = country_part.split()
    if len(words) >= 2 and words[0] == 'UNITED':
        # Check second word to distinguish "UNITED STATES" from "UNITED KINGDOM" and "UNITED ARAB EMIRATES"
        if words[1] == 'STATES':
            return 'UNITED STATES'
        elif len(words) >= 3 and words[1] == 'ARAB':
            return 'UNITED ARAB EMIRATES'
        else:
            return 'UNITED KINGDOM'
    elif len(words) >= 2 and words[0] == 'KOREA,':
        return 'SOUTH KOREA'
    elif len(words) >= 2 and words[0] == 'HONG':
        return 'HONG KONG'
    # For most countries, take everything before the first city (usually 1-2 words)
    # Cities typically start with capital letter followed by lowercase
    # But country names are often all caps, so take first word if all caps
    if words[0].isupper():
        return words[0]
    return country_part

def filter_non_us_jobs(df):
    """Filter dataframe for non-US jobs."""
    # Non-US means doesn't start with 'UNITED STATES' and isn't empty
    non_empty_mask = df['locations'].fillna('').str.len() > 0
    us_mask = df['locations'].fillna('').str.startswith('UNITED STATES', na=False)
    non_us_df = df[non_empty_mask & ~us_mask].copy()

    print(f"\nFiltered for non-US jobs: {len(non_us_df)} out of {len(df)} total postings")
    print(f"\nTop countries:")

    countries = non_us_df['locations'].apply(extract_country)
    print(countries.value_counts().head(15))

    return non_us_df

def filter_by_job_type(df, job_type):
    """Filter dataframe by job type.

    Args:
        df: DataFrame with job postings
        job_type: 'tenure_track', 'non_tenure_academic', 'industry'
    """
    if job_type == 'tenure_track':
        # Tenure track or tenured positions (both US and International)
        mask = df['jp_section'].fillna('').str.contains('Tenure Track or Tenured', case=False, na=False)
    elif job_type == 'non_tenure_academic':
        # Other academic positions (visiting, temporary, part-time, adjunct)
        mask = df['jp_section'].fillna('').str.contains('Other Academic', case=False, na=False)
    elif job_type == 'industry':
        # All nonacademic positions (full-time + temporary/consulting)
        mask = df['jp_section'].fillna('').str.contains('Nonacademic', case=False, na=False)
    else:
        raise ValueError(f"Unknown job_type: {job_type}")

    filtered_df = df[mask].copy()
    print(f"\nFiltered for {job_type.upper().replace('_', ' ')} jobs: {len(filtered_df)} out of {len(df)} total postings ({100*len(filtered_df)/len(df):.1f}%)")

    return filtered_df

def filter_region_jobs(df, region):
    """Filter dataframe by geographic region.

    Args:
        df: DataFrame with job postings
        region: 'us', 'canada_europe', or 'asia'
    """
    # Define country sets for each region
    european_countries = {
        'UNITED KINGDOM', 'GERMANY', 'SWITZERLAND', 'FRANCE', 'ITALY', 'SPAIN',
        'NETHERLANDS', 'BELGIUM', 'SWEDEN', 'NORWAY', 'DENMARK', 'FINLAND',
        'AUSTRIA', 'IRELAND', 'PORTUGAL', 'GREECE', 'POLAND', 'CZECH',
        'HUNGARY', 'ROMANIA', 'CROATIA', 'BULGARIA', 'SLOVAKIA', 'SLOVENIA',
        'LUXEMBOURG', 'ESTONIA', 'LATVIA', 'LITHUANIA', 'CYPRUS', 'MALTA',
        'ICELAND', 'TURKEY'
    }

    asian_countries = {
        'CHINA', 'JAPAN', 'SOUTH KOREA', 'KOREA,', 'HONG KONG', 'HONG', 'TAIWAN',
        'SINGAPORE', 'INDIA', 'THAILAND', 'INDONESIA', 'MALAYSIA', 'PHILIPPINES',
        'VIETNAM', 'PAKISTAN', 'BANGLADESH', 'SRI', 'ISRAEL', 'UNITED ARAB EMIRATES',
        'SAUDI', 'QATAR', 'KUWAIT', 'BAHRAIN', 'OMAN', 'JORDAN', 'LEBANON',
        'MONGOLIA', 'CAMBODIA', 'MYANMAR', 'NEPAL', 'KAZAKHSTAN', 'UZBEKISTAN'
    }

    # Add country column
    df_copy = df.copy()
    df_copy['country'] = df_copy['locations'].apply(extract_country)

    if region == 'us':
        mask = df_copy['locations'].fillna('').str.startswith('UNITED STATES', na=False)
    elif region == 'canada_europe':
        mask = (df_copy['country'] == 'CANADA') | (df_copy['country'].isin(european_countries))
    elif region == 'asia':
        mask = df_copy['country'].isin(asian_countries)
    else:
        raise ValueError(f"Unknown region: {region}")

    filtered_df = df_copy[mask].copy()
    print(f"\nFiltered for {region.upper().replace('_', ' + ')} jobs: {len(filtered_df)} out of {len(df)} total postings")

    return filtered_df

def main():
    """Main function to orchestrate the analysis."""
    import sys

    # Check if --interpolate flag is provided
    interpolate = '--interpolate' in sys.argv

    print("="*60)
    print("JOB POSTINGS ANALYSIS")
    if interpolate:
        print("(WITH CURRENT WEEK INTERPOLATION)")
    print("="*60)

    # Parse Excel files
    df = parse_excel_files('data')

    # Calculate cumulative postings by week
    cumulative_data = calculate_cumulative_by_week(df, interpolate_current=interpolate)

    # Print summary statistics
    print_summary_statistics(cumulative_data)

    # Plot the cumulative data (static)
    suffix = '_interpolated' if interpolate else ''
    plot_cumulative_by_week(cumulative_data, output_file=f'job_postings_by_week{suffix}.png')
    
    # Plot cumulative by academic year for selected weeks
    plot_cumulative_by_ay(cumulative_data, output_file=f'job_postings_by_ay{suffix}.png')

    # Calculate rolling 4-week flow
    rolling_data = calculate_rolling_four_week(cumulative_data)

    # Plot the rolling 4-week flow (static)
    plot_rolling_four_week(rolling_data, output_file=f'job_postings_rolling_4wk{suffix}.png')

    # Create interactive HTML plots
    print("\nCreating interactive plots...")
    plot_cumulative_interactive(cumulative_data, output_file=f'job_postings_by_week{suffix}.html')
    plot_cumulative_by_ay_interactive(cumulative_data, output_file=f'job_postings_by_ay{suffix}.html')
    plot_rolling_interactive(rolling_data, output_file=f'job_postings_rolling_4wk{suffix}.html')

    # Now do the same for finance jobs only
    print("\n" + "="*60)
    print("FINANCE JOBS ANALYSIS")
    print("="*60)

    finance_df = filter_finance_jobs(df)

    # Calculate cumulative postings by week for finance jobs
    cumulative_data_finance = calculate_cumulative_by_week(finance_df, interpolate_current=interpolate)

    # Print summary statistics
    print_summary_statistics(cumulative_data_finance)

    # Plot the cumulative data (static)
    plot_cumulative_by_week(cumulative_data_finance, output_file=f'job_postings_by_week_finance{suffix}.png')

    # Calculate rolling 4-week flow
    rolling_data_finance = calculate_rolling_four_week(cumulative_data_finance)

    # Plot the rolling 4-week flow (static)
    plot_rolling_four_week(rolling_data_finance, output_file=f'job_postings_rolling_4wk_finance{suffix}.png')

    # Create interactive HTML plots
    print("\nCreating interactive plots for finance jobs...")
    plot_cumulative_interactive(cumulative_data_finance, output_file=f'job_postings_by_week_finance{suffix}.html')
    plot_rolling_interactive(rolling_data_finance, output_file=f'job_postings_rolling_4wk_finance{suffix}.html')

    # Now do the same for Fed/Regulator jobs only
    print("\n" + "="*60)
    print("FEDERAL RESERVE & BANK REGULATOR JOBS ANALYSIS")
    print("="*60)

    fed_df = filter_fed_regulator_jobs(df)

    # Calculate cumulative postings by week for Fed/regulator jobs
    cumulative_data_fed = calculate_cumulative_by_week(fed_df, interpolate_current=interpolate)

    # Print summary statistics
    print_summary_statistics(cumulative_data_fed)

    # Plot the cumulative data (static)
    plot_cumulative_by_week(cumulative_data_fed, output_file=f'job_postings_by_week_fed{suffix}.png')

    # Calculate rolling 4-week flow
    rolling_data_fed = calculate_rolling_four_week(cumulative_data_fed)

    # Plot the rolling 4-week flow (static)
    plot_rolling_four_week(rolling_data_fed, output_file=f'job_postings_rolling_4wk_fed{suffix}.png')

    # Create interactive HTML plots
    print("\nCreating interactive plots for Fed/regulator jobs...")
    plot_cumulative_interactive(cumulative_data_fed, output_file=f'job_postings_by_week_fed{suffix}.html')
    plot_rolling_interactive(rolling_data_fed, output_file=f'job_postings_rolling_4wk_fed{suffix}.html')

    # Now do the same for US jobs only
    print("\n" + "="*60)
    print("US JOBS ANALYSIS")
    print("="*60)

    us_df = filter_us_jobs(df)

    # Calculate cumulative postings by week for US jobs
    cumulative_data_us = calculate_cumulative_by_week(us_df, interpolate_current=interpolate)

    # Print summary statistics
    print_summary_statistics(cumulative_data_us)

    # Plot the cumulative data (static)
    plot_cumulative_by_week(cumulative_data_us, output_file=f'job_postings_by_week_us{suffix}.png')

    # Calculate rolling 4-week flow
    rolling_data_us = calculate_rolling_four_week(cumulative_data_us)

    # Plot the rolling 4-week flow (static)
    plot_rolling_four_week(rolling_data_us, output_file=f'job_postings_rolling_4wk_us{suffix}.png')

    # Create interactive HTML plots
    print("\nCreating interactive plots for US jobs...")
    plot_cumulative_interactive(cumulative_data_us, output_file=f'job_postings_by_week_us{suffix}.html')
    plot_rolling_interactive(rolling_data_us, output_file=f'job_postings_rolling_4wk_us{suffix}.html')

    # Now do the same for non-US jobs only
    print("\n" + "="*60)
    print("NON-US JOBS ANALYSIS")
    print("="*60)

    non_us_df = filter_non_us_jobs(df)

    # Calculate cumulative postings by week for non-US jobs
    cumulative_data_non_us = calculate_cumulative_by_week(non_us_df, interpolate_current=interpolate)

    # Print summary statistics
    print_summary_statistics(cumulative_data_non_us)

    # Plot the cumulative data (static)
    plot_cumulative_by_week(cumulative_data_non_us, output_file=f'job_postings_by_week_non_us{suffix}.png')

    # Calculate rolling 4-week flow
    rolling_data_non_us = calculate_rolling_four_week(cumulative_data_non_us)

    # Plot the rolling 4-week flow (static)
    plot_rolling_four_week(rolling_data_non_us, output_file=f'job_postings_rolling_4wk_non_us{suffix}.png')

    # Create interactive HTML plots
    print("\nCreating interactive plots for non-US jobs...")
    plot_cumulative_interactive(cumulative_data_non_us, output_file=f'job_postings_by_week_non_us{suffix}.html')
    plot_rolling_interactive(rolling_data_non_us, output_file=f'job_postings_rolling_4wk_non_us{suffix}.html')

    # Regional analysis: US vs. Canada+Europe vs. Asia
    print("\n" + "="*60)
    print("REGIONAL COMPARISON: US vs. CANADA+EUROPE vs. ASIA")
    print("="*60)

    # Filter by region
    us_region_df = filter_region_jobs(df, 'us')
    canada_europe_df = filter_region_jobs(df, 'canada_europe')
    asia_df = filter_region_jobs(df, 'asia')

    # Calculate cumulative data for each region
    cumulative_us_region = calculate_cumulative_by_week(us_region_df, interpolate_current=interpolate)
    cumulative_canada_europe = calculate_cumulative_by_week(canada_europe_df, interpolate_current=interpolate)
    cumulative_asia = calculate_cumulative_by_week(asia_df, interpolate_current=interpolate)

    # Print summary statistics
    print("\n" + "-"*60)
    print("US REGION")
    print("-"*60)
    print_summary_statistics(cumulative_us_region)

    print("\n" + "-"*60)
    print("CANADA + EUROPE REGION")
    print("-"*60)
    print_summary_statistics(cumulative_canada_europe)

    print("\n" + "-"*60)
    print("ASIA REGION")
    print("-"*60)
    print_summary_statistics(cumulative_asia)

    # Create plots for each region
    plot_cumulative_by_week(cumulative_us_region, output_file=f'job_postings_by_week_region_us{suffix}.png')
    plot_cumulative_by_week(cumulative_canada_europe, output_file=f'job_postings_by_week_region_canada_europe{suffix}.png')
    plot_cumulative_by_week(cumulative_asia, output_file=f'job_postings_by_week_region_asia{suffix}.png')

    rolling_us_region = calculate_rolling_four_week(cumulative_us_region)
    rolling_canada_europe = calculate_rolling_four_week(cumulative_canada_europe)
    rolling_asia = calculate_rolling_four_week(cumulative_asia)

    plot_rolling_four_week(rolling_us_region, output_file=f'job_postings_rolling_4wk_region_us{suffix}.png')
    plot_rolling_four_week(rolling_canada_europe, output_file=f'job_postings_rolling_4wk_region_canada_europe{suffix}.png')
    plot_rolling_four_week(rolling_asia, output_file=f'job_postings_rolling_4wk_region_asia{suffix}.png')

    # Create interactive plots
    print("\nCreating interactive plots for regional comparison...")
    plot_cumulative_interactive(cumulative_us_region, output_file=f'job_postings_by_week_region_us{suffix}.html')
    plot_cumulative_interactive(cumulative_canada_europe, output_file=f'job_postings_by_week_region_canada_europe{suffix}.html')
    plot_cumulative_interactive(cumulative_asia, output_file=f'job_postings_by_week_region_asia{suffix}.html')

    plot_rolling_interactive(rolling_us_region, output_file=f'job_postings_rolling_4wk_region_us{suffix}.html')
    plot_rolling_interactive(rolling_canada_europe, output_file=f'job_postings_rolling_4wk_region_canada_europe{suffix}.html')
    plot_rolling_interactive(rolling_asia, output_file=f'job_postings_rolling_4wk_region_asia{suffix}.html')

    # Job type analysis: Tenure track vs. Non-tenure academic vs. Industry
    print("\n" + "="*60)
    print("JOB TYPE COMPARISON")
    print("="*60)

    # Filter by job type
    tenure_track_df = filter_by_job_type(df, 'tenure_track')
    non_tenure_df = filter_by_job_type(df, 'non_tenure_academic')
    industry_df = filter_by_job_type(df, 'industry')

    # Calculate cumulative data for each job type
    cumulative_tenure = calculate_cumulative_by_week(tenure_track_df, interpolate_current=interpolate)
    cumulative_non_tenure = calculate_cumulative_by_week(non_tenure_df, interpolate_current=interpolate)
    cumulative_industry = calculate_cumulative_by_week(industry_df, interpolate_current=interpolate)

    # Print summary statistics
    print("\n" + "-"*60)
    print("TENURE TRACK POSITIONS")
    print("-"*60)
    print_summary_statistics(cumulative_tenure)

    print("\n" + "-"*60)
    print("NON-TENURE ACADEMIC POSITIONS")
    print("-"*60)
    print_summary_statistics(cumulative_non_tenure)

    print("\n" + "-"*60)
    print("INDUSTRY POSITIONS (All Nonacademic)")
    print("-"*60)
    print_summary_statistics(cumulative_industry)

    # Create plots for each job type
    plot_cumulative_by_week(cumulative_tenure, output_file=f'job_postings_by_week_tenure_track{suffix}.png')
    plot_cumulative_by_week(cumulative_non_tenure, output_file=f'job_postings_by_week_non_tenure{suffix}.png')
    plot_cumulative_by_week(cumulative_industry, output_file=f'job_postings_by_week_industry{suffix}.png')

    rolling_tenure = calculate_rolling_four_week(cumulative_tenure)
    rolling_non_tenure = calculate_rolling_four_week(cumulative_non_tenure)
    rolling_industry = calculate_rolling_four_week(cumulative_industry)

    plot_rolling_four_week(rolling_tenure, output_file=f'job_postings_rolling_4wk_tenure_track{suffix}.png')
    plot_rolling_four_week(rolling_non_tenure, output_file=f'job_postings_rolling_4wk_non_tenure{suffix}.png')
    plot_rolling_four_week(rolling_industry, output_file=f'job_postings_rolling_4wk_industry{suffix}.png')

    # Create interactive plots
    print("\nCreating interactive plots for job type comparison...")
    plot_cumulative_interactive(cumulative_tenure, output_file=f'job_postings_by_week_tenure_track{suffix}.html')
    plot_cumulative_interactive(cumulative_non_tenure, output_file=f'job_postings_by_week_non_tenure{suffix}.html')
    plot_cumulative_interactive(cumulative_industry, output_file=f'job_postings_by_week_industry{suffix}.html')

    plot_rolling_interactive(rolling_tenure, output_file=f'job_postings_rolling_4wk_tenure_track{suffix}.html')
    plot_rolling_interactive(rolling_non_tenure, output_file=f'job_postings_rolling_4wk_non_tenure{suffix}.html')
    plot_rolling_interactive(rolling_industry, output_file=f'job_postings_rolling_4wk_industry{suffix}.html')

    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()
