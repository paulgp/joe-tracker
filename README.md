# JOE Job Market Tracker

Analysis tool for tracking economics job market postings from the American Economic Association's Job Openings for Economists (JOE).

## Features

- **Cumulative posting trends** by week across multiple years (2015-2025)
- **Rolling 4-week flow** analysis to smooth out weekly fluctuations
- **Multiple filter views**:
  - Overall market
  - Finance jobs (based on JEL G classifications)
  - Federal Reserve and bank regulator positions
  - US vs. Non-US jobs
  - Regional analysis (US, Canada+Europe, Asia)
  - Job type (Tenure Track, Non-Tenure Academic, Industry)
- **Interactive HTML plots** and static PNG charts
- **Current week interpolation** for more accurate mid-week comparisons

## Installation

```bash
pip install pandas matplotlib plotly seaborn openpyxl
```

## Usage

### Basic Analysis

Run the standard analysis without interpolation:

```bash
python3 analyze_jobs.py
```

This generates:
- Static PNG charts for all categories
- Interactive HTML visualizations
- Summary statistics for each year

### Interpolated Analysis

Run analysis with current week interpolation:

```bash
python3 analyze_jobs.py --interpolate
```

#### How Interpolation Works

When the `--interpolate` flag is used, the script:

1. Detects the current day of the week (Monday = day 1, Sunday = day 7)
2. Calculates what fraction of the current week has elapsed
3. Projects the current week's posting count to a full-week estimate
4. Updates all 2025 cumulative totals accordingly

**Example:**
- Current day: Monday (day 1 of 7, or 14.3% of week)
- Postings so far this week: 15
- Interpolated full-week count: 15 / 0.143 ≈ 105
- Added to cumulative: +90 postings

**When to use interpolation:**
- Mid-week comparisons to understand if 2025 is tracking ahead/behind prior years
- Blog posts or reports generated during the week
- Real-time monitoring of market trends

**When NOT to use interpolation:**
- Final end-of-season analysis (use actual data)
- Early in the week (Monday/Tuesday) when estimates are less reliable
- When postings are clustered at specific times during the week

**Note:** The interpolation assumes an even distribution of postings throughout the week, which may not reflect actual posting patterns. Use with appropriate context.

### Output Files

All outputs are saved to the current directory:

**Without interpolation:**
- `job_postings_by_week.html` - Interactive cumulative chart
- `job_postings_rolling_4wk.html` - Interactive rolling 4-week chart
- Plus category-specific files (finance, fed, us, non_us, regions, job types)

**With interpolation:**
- All files get `_interpolated` suffix (e.g., `job_postings_by_week_interpolated.html`)

## Data Format

The script expects Excel files in a `data/` directory with the following columns:
- `Date_Active` - When the posting became active
- `JEL_Classifications` - JEL classification codes
- `jp_institution` - Institution name
- `jp_section` - Job posting section (Tenure Track, Nonacademic, etc.)
- `locations` - Geographic location information

## Analysis Categories

### Overall Market
All postings across all categories

### Finance Jobs
Filtered to positions with "G - Financial Economics" JEL classification

### Fed/Regulator Jobs
Positions at:
- Federal Reserve Banks (all districts)
- Board of Governors
- FDIC
- OCC

### Geographic Analysis
- **US vs. Non-US**: Split between United States and international positions
- **Regional**: US, Canada+Europe, and Asia breakdowns

### Job Type Analysis
- **Tenure Track**: Tenure-track or tenured positions
- **Non-Tenure Academic**: Visiting, temporary, part-time, adjunct positions
- **Industry**: All nonacademic positions

## Technical Details

- **Academic year calculation**: Starts from week 31 (early August)
- **Week numbering**: ISO week numbers, adjusted relative to start of academic job market
- **Interpolation**: Linear projection based on elapsed days in current week
- **Rolling window**: 4-week moving average for smoothing

## Author

Created for tracking the economics job market. Data sourced from the AEA's Job Openings for Economists.
