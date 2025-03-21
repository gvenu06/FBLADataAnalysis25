import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
from matplotlib.collections import LineCollection

# Set up premium visualization style
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica Neue', 'Helvetica', 'Arial']
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['axes.edgecolor'] = '#333333'
mpl.rcParams['xtick.major.width'] = 0.8
mpl.rcParams['ytick.major.width'] = 0.8
mpl.rcParams['xtick.color'] = '#333333'
mpl.rcParams['ytick.color'] = '#333333'

# Color palettes for different pollutants
COLORS = {
    'ozone': ['#1A5276', '#2874A6', '#3498DB', '#5DADE2', '#85C1E9'],
    'no2': ['#641E16', '#922B21', '#C0392B', '#E74C3C', '#F1948A'],
    'pm25': ['#145A32', '#196F3D', '#229954', '#27AE60', '#58D68D'],
    'health': ['#6C3483', '#7D3C98', '#8E44AD', '#A569BD', '#BB8FCE'],
    'background': '#F8F9F9'
}


def plot_yearly_data(df, pollutant_name, color_key, filename, title_prefix=None, summer_only=False):
    """
    Create a premium line chart showing yearly average pollutant levels in NYC

    Args:
        df: DataFrame with air quality data
        pollutant_name: Name of the pollutant to plot
        color_key: Key for the COLORS dictionary
        filename: Output filename
        title_prefix: Optional prefix for the title
        summer_only: If True, filter for summer months only (June-August)
    """
    # Filter for specific pollutant data
    pollutant_data = df[df['Name'] == pollutant_name].copy()

    # Convert dates and extract year and month
    pollutant_data['Start_Date'] = pd.to_datetime(pollutant_data['Start_Date'])
    pollutant_data['Year'] = pollutant_data['Start_Date'].dt.year
    pollutant_data['Month'] = pollutant_data['Start_Date'].dt.month

    # Filter for summer months if requested
    if summer_only:
        pollutant_data = pollutant_data[(pollutant_data['Month'] >= 6) & (pollutant_data['Month'] <= 8)]
        season_text = "Summer "
    else:
        season_text = "Annual "

    # Calculate yearly averages
    yearly_avg = pollutant_data.groupby('Year')['Data Value'].mean().reset_index()

    # Create custom color gradient for the line
    cmap = LinearSegmentedColormap.from_list(f"{color_key}_gradient", COLORS[color_key])

    # Create the figure with a specific aspect ratio
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor(COLORS['background'])

    # Custom background with very subtle gradient
    gradient = np.linspace(0, 1, 100).reshape(-1, 1)
    ax.imshow(gradient, aspect='auto', extent=[yearly_avg['Year'].min() - 0.5, yearly_avg['Year'].max() + 0.5,
                                               min(yearly_avg['Data Value']) * 0.95,
                                               max(yearly_avg['Data Value']) * 1.05],
              alpha=0.05, cmap=LinearSegmentedColormap.from_list("bg_gradient", [COLORS['background'], '#ECF0F1']))

    # Plot line with gradient effect
    points = np.array([yearly_avg['Year'], yearly_avg['Data Value']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(yearly_avg['Year'].min(), yearly_avg['Year'].max())
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3.5, alpha=0.9)
    lc.set_array(yearly_avg['Year'])
    line = ax.add_collection(lc)

    # Add data points with style
    scatter = ax.scatter(yearly_avg['Year'], yearly_avg['Data Value'],
                         s=80,
                         color=cmap(norm(yearly_avg['Year'])),
                         edgecolor='white',
                         linewidth=1.5,
                         zorder=5,
                         alpha=0.9)

    # Add subtle shadow effect to the line
    shadow = ax.plot(yearly_avg['Year'], yearly_avg['Data Value'],
                     color='#333333',
                     linewidth=4.5,
                     alpha=0.15,
                     zorder=1)

    # Add value labels with style
    for i, row in yearly_avg.iterrows():
        label = ax.annotate(f"{row['Data Value']:.2f}",
                            (row['Year'], row['Data Value']),
                            xytext=(0, 10),
                            textcoords='offset points',
                            ha='center',
                            va='bottom',
                            fontsize=9,
                            color='#333333',
                            fontweight='bold')
        label.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])

    # Add trend line
    z = np.polyfit(yearly_avg['Year'], yearly_avg['Data Value'], 1)
    p = np.poly1d(z)
    trend = ax.plot(yearly_avg['Year'], p(yearly_avg['Year']),
                    linestyle='--',
                    linewidth=1.5,
                    color=COLORS[color_key][1],
                    alpha=0.6,
                    zorder=2)

    # Add trend annotation
    slope = z[0]
    trend_direction = "increasing" if slope > 0 else "decreasing"
    trend_color = '#7B241C' if slope > 0 else '#145A32'
    ax.text(0.02, 0.05,
            f"Trend: {trend_direction} at {abs(slope):.3f} units per year",
            transform=ax.transAxes,
            fontsize=9,
            color=trend_color,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.5'))

    # Set x-axis to show all years (no skipping)
    ax.set_xticks(yearly_avg['Year'])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)

    # Styling y-axis
    ax.yaxis.grid(True, linestyle='-', alpha=0.15, linewidth=1.5)
    ax.xaxis.grid(False)

    # Extract nice pollutant name for display
    pollutant_short = pollutant_name.split('(')[0].strip()

    # Override title if provided
    if title_prefix:
        main_title = title_prefix
    else:
        main_title = f"NYC {season_text}{pollutant_short} Index Trends"

    # Add labels and title with premium styling
    title = ax.set_title(main_title,
                         fontsize=18,
                         pad=20,
                         fontweight='bold',
                         color='#2C3E50')
    title.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='#EAECEE')])

    subtitle = ax.text(0.5, 0.97, f"{season_text}Average {pollutant_short} Concentration",
                       transform=ax.transAxes,
                       fontsize=12,
                       ha='center',
                       va='top',
                       color='#566573')

    ax.set_xlabel('Year', fontsize=12, labelpad=10, color='#2C3E50', fontweight='bold')
    ax.set_ylabel(f'{pollutant_short} Concentration', fontsize=12, labelpad=10, color='#2C3E50', fontweight='bold')

    # Add reference line for threshold
    threshold = yearly_avg['Data Value'].mean()
    ax.axhline(y=threshold, color=COLORS[color_key][2], linestyle='--', alpha=0.4, linewidth=1.5)
    ax.text(yearly_avg['Year'].max() + 0.1, threshold,
            f'Average: {threshold:.2f}',
            va='center',
            ha='left',
            fontsize=9,
            color=COLORS[color_key][2],
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # Remove spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Adjust y-axis limits for better spacing
    ymin, ymax = min(yearly_avg['Data Value']), max(yearly_avg['Data Value'])
    range_y = ymax - ymin
    ax.set_ylim(ymin - range_y * 0.1, ymax + range_y * 0.15)

    # Add source text
    fig.text(0.02, 0.02, 'Source: NYC Air Quality Data', fontsize=8, color='#7F8C8D')

    # Add minimal branded footer
    fig.text(0.98, 0.02, 'NYC Environmental Analytics', fontsize=9, color='#2C3E50',
             ha='right', fontweight='bold')

    # Tight layout for optimal spacing
    plt.tight_layout()

    # Save with high quality
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()

    return yearly_avg


def plot_health_impacts(df, health_metrics, filename):
    """
    Create a visualization showing changes in health impacts related to air pollution

    Args:
        df: DataFrame with air quality data
        health_metrics: List of health metrics to include
        filename: Output filename
    """
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor(COLORS['background'])

    # Consistent years across metrics for comparison
    years = [2005, 2009, 2013, 2017]  # Based on data availability
    cmap = LinearSegmentedColormap.from_list("health_gradient", COLORS['health'])

    # Plotting multiple metrics
    for i, metric in enumerate(health_metrics):
        metric_data = df[df['Name'] == metric].copy()
        metric_data['Start_Date'] = pd.to_datetime(metric_data['Start_Date'])
        metric_data['Year'] = metric_data['Start_Date'].dt.year

        # Get data for the consistent years
        yearly_avg = metric_data.groupby('Year')['Data Value'].mean()
        years_available = [y for y in years if y in yearly_avg.index]
        values = [yearly_avg[y] if y in yearly_avg.index else np.nan for y in years]

        # Normalize for comparison (2005 as baseline = 100%)
        base_year_value = yearly_avg[years_available[0]]
        norm_values = [v / base_year_value * 100 if not np.isnan(v) else np.nan for v in values]

        # Short name for display
        short_name = metric.split('(')[0].strip()
        if len(short_name) > 30:
            short_name = short_name[:27] + "..."

        # Plot line with markers
        line = ax.plot(years_available, [norm_values[years.index(y)] for y in years_available],
                       marker='o',
                       linewidth=2.5,
                       label=short_name,
                       color=cmap(i / len(health_metrics)))

        # Add value labels
        for j, y in enumerate(years_available):
            value = norm_values[years.index(y)]
            ax.annotate(f"{value:.1f}%",
                        (y, value),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=8,
                        fontweight='bold',
                        color=cmap(i / len(health_metrics)))

    # Styling
    ax.set_title("Health Impacts of Air Pollution in NYC (2005-2017)",
                 fontsize=18,
                 pad=20,
                 fontweight='bold',
                 color='#2C3E50')

    ax.set_xlabel('Year', fontsize=12, labelpad=10, color='#2C3E50', fontweight='bold')
    ax.set_ylabel('Relative Impact (2005 = 100%)', fontsize=12, labelpad=10, color='#2C3E50', fontweight='bold')

    # Add baseline reference
    ax.axhline(y=100, color='#7F8C8D', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(years[0] - 0.5, 100, '2005 Baseline', va='center', fontsize=8, color='#7F8C8D')

    # Customize legend
    ax.legend(bbox_to_anchor=(0.5, -0.15),
              loc='upper center',
              ncol=2,
              frameon=False,
              fontsize=10)

    # Set x-axis ticks to just the years we have data for
    ax.set_xticks(years)
    ax.set_xticklabels(years)

    # Remove spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Add the annotation of improvement
    rect = plt.Rectangle((0.02, 0.02), 0.3, 0.15,
                         transform=ax.transAxes,
                         facecolor='white',
                         alpha=0.8,
                         edgecolor='#CCD1D1',
                         zorder=5)
    ax.add_patch(rect)

    # Get overall improvement
    last_year_values = []
    base_year_values = []
    for metric in health_metrics:
        metric_data = df[df['Name'] == metric].copy()
        metric_data['Start_Date'] = pd.to_datetime(metric_data['Start_Date'])
        metric_data['Year'] = metric_data['Start_Date'].dt.year

        yearly_avg = metric_data.groupby('Year')['Data Value'].mean()

        if min(yearly_avg.index) <= 2005 and max(yearly_avg.index) >= 2017:
            base_year_values.append(yearly_avg[2005])
            last_year_values.append(yearly_avg[2017])

    if base_year_values and last_year_values:
        avg_change = (sum(last_year_values) / sum(base_year_values) - 1) * 100
        change_text = f"Overall improvement: {abs(avg_change):.1f}%"

        ax.text(0.03, 0.14,
                change_text,
                transform=ax.transAxes,
                fontsize=11,
                color='#2C3E50',
                fontweight='bold')

        ax.text(0.03, 0.08,
                "in pollution-related health impacts",
                transform=ax.transAxes,
                fontsize=9,
                color='#566573')

        ax.text(0.03, 0.04,
                "from 2005 to 2017",
                transform=ax.transAxes,
                fontsize=9,
                color='#566573')

    # Tight layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Adjust for legend

    # Save
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()


def plot_multi_pollutant_trends(df, filename):
    """
    Create a visualization showing all three major pollutants on the same chart
    """
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor(COLORS['background'])

    pollutants = [
        ('Ozone (O3)', 'ozone'),
        ('Nitrogen dioxide (NO2)', 'no2'),
        ('Fine particles (PM 2.5)', 'pm25')
    ]

    legend_handles = []

    for pollutant_name, color_key in pollutants:
        # Filter for this pollutant
        pollutant_data = df[df['Name'] == pollutant_name].copy()
        pollutant_data['Start_Date'] = pd.to_datetime(pollutant_data['Start_Date'])
        pollutant_data['Year'] = pollutant_data['Start_Date'].dt.year

        # Calculate yearly averages
        yearly_avg = pollutant_data.groupby('Year')['Data Value'].mean().reset_index()

        # Get range of years for this pollutant
        years = yearly_avg['Year'].tolist()

        # Create colormap for this pollutant
        cmap = LinearSegmentedColormap.from_list(f"{color_key}_gradient", COLORS[color_key])

        # Normalize the data for comparison across pollutants (first year = 100%)
        first_year_value = yearly_avg.loc[0, 'Data Value']
        yearly_avg['Normalized'] = yearly_avg['Data Value'] / first_year_value * 100

        # Plot line with gradient effect
        points = np.array([yearly_avg['Year'], yearly_avg['Normalized']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(yearly_avg['Year'].min(), yearly_avg['Year'].max())
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3, alpha=0.8)
        lc.set_array(yearly_avg['Year'])
        line = ax.add_collection(lc)

        # Add data points
        scatter = ax.scatter(yearly_avg['Year'], yearly_avg['Normalized'],
                             s=60,
                             color=cmap(norm(yearly_avg['Year'])),
                             edgecolor='white',
                             linewidth=1,
                             zorder=5,
                             alpha=0.8)

        # Get clean name for the legend
        pollutant_short = pollutant_name.split('(')[0].strip()

        # Add custom proxy for legend
        proxy = plt.Line2D([0], [0],
                           color=COLORS[color_key][2],
                           linewidth=2.5,
                           marker='o',
                           markersize=8,
                           markerfacecolor=COLORS[color_key][2],
                           markeredgecolor='white',
                           label=pollutant_short)

        legend_handles.append(proxy)

    # Styling
    title = ax.set_title("Air Quality Improvement in NYC (2008-2022)",
                         fontsize=18,
                         pad=20,
                         fontweight='bold',
                         color='#2C3E50')
    title.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='#EAECEE')])

    subtitle = ax.text(0.5, 0.97, "Normalized Annual Average Pollutant Levels (First Year = 100%)",
                       transform=ax.transAxes,
                       fontsize=12,
                       ha='center',
                       va='top',
                       color='#566573')

    ax.set_xlabel('Year', fontsize=12, labelpad=10, color='#2C3E50', fontweight='bold')
    ax.set_ylabel('Relative Pollutant Level (%)', fontsize=12, labelpad=10, color='#2C3E50', fontweight='bold')

    # Set x-axis
    # First ensure Start_Date is converted to datetime for the relevant rows
    filtered_df = df[df['Name'].isin([p[0] for p in pollutants])].copy()
    filtered_df['Start_Date'] = pd.to_datetime(filtered_df['Start_Date'])

    start_year = min(filtered_df['Start_Date'].dt.year)
    end_year = max(filtered_df['Start_Date'].dt.year)
    ax.set_xlim(start_year - 0.5, end_year + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)

    # Add reference line
    ax.axhline(y=100, color='#7F8C8D', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(start_year, 100, 'Baseline', va='center', fontsize=8, color='#7F8C8D')

    # Add legend
    ax.legend(handles=legend_handles,
              loc='upper right',
              frameon=False,
              fontsize=10)

    # Remove spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Calculate improvement for each pollutant
    improvements = []
    for pollutant_name, color_key in pollutants:
        pollutant_data = df[df['Name'] == pollutant_name].copy()
        pollutant_data['Start_Date'] = pd.to_datetime(pollutant_data['Start_Date'])
        pollutant_data['Year'] = pollutant_data['Start_Date'].dt.year

        yearly_avg = pollutant_data.groupby('Year')['Data Value'].mean()

        if len(yearly_avg) > 0:
            first_year = min(yearly_avg.index)
            last_year = max(yearly_avg.index)
            improvement = (yearly_avg[first_year] - yearly_avg[last_year]) / yearly_avg[first_year] * 100

            pollutant_short = pollutant_name.split('(')[0].strip()
            improvements.append((pollutant_short, improvement, first_year, last_year))

    # Add improvement annotation
    if improvements:
        y_pos = 0.2
        ax.text(0.02, y_pos, "Pollutant Reductions:",
                transform=ax.transAxes,
                fontsize=10,
                fontweight='bold',
                color='#2C3E50')

        for i, (name, imp, start, end) in enumerate(improvements):
            y_pos -= 0.05
            ax.text(0.02, y_pos,
                    f"{name}: {imp:.1f}% ({start}-{end})",
                    transform=ax.transAxes,
                    fontsize=9,
                    color=COLORS[pollutants[i][1]][2])

    # Tight layout
    plt.tight_layout()

    # Save
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()


def main():
    # Load data
    df = pd.read_csv('Air_Quality.csv')

    # 1. Create summer ozone trend plot (as requested)
    plot_yearly_data(df, 'Ozone (O3)', 'ozone', 'nyc_summer_ozone_trend.png', summer_only=True)

    # 2. Create annual average NO2 Index plot
    plot_yearly_data(df, 'Nitrogen dioxide (NO2)', 'no2', 'nyc_annual_no2_trend.png')

    # 3. Create annual average Fine Particles (PM2.5) plot
    plot_yearly_data(df, 'Fine particles (PM 2.5)', 'pm25', 'nyc_annual_pm25_trend.png')

    # 4. Create health impacts visualization
    health_metrics = [
        'Asthma emergency department visits due to PM2.5',
        'Respiratory hospitalizations due to PM2.5 (age 20+)',
        'Asthma hospitalizations due to Ozone',
        'Cardiovascular hospitalizations due to PM2.5 (age 40+)'
    ]
    plot_health_impacts(df, health_metrics, 'nyc_health_impacts.png')

    # 5. Create comparative visualization of all pollutants
    plot_multi_pollutant_trends(df, 'nyc_pollutant_comparison.png')

    # Print summary info
    print("Visualization files created:")
    print("1. nyc_summer_ozone_trend.png - Summer Ozone Index")
    print("2. nyc_annual_no2_trend.png - Annual NO2 Index")
    print("3. nyc_annual_pm25_trend.png - Annual Fine Particles (PM2.5) Index")
    print("4. nyc_health_impacts.png - Health Impacts of Air Pollution")
    print("5. nyc_pollutant_comparison.png - Comparative Pollutant Trends")


if __name__ == "__main__":
    main()