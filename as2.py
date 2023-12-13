 
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
years = []

def read_climate_change_data(file_path):
    """
    Reads climate change data from a CSV file and transposes it.

    Args:
    file_path (str): Path to the CSV file.

    Returns:
    tuple: A tuple containing the original DataFrame and the transposed DataFrame,
           or (None, None) if there's an error.
    """
    # Try reading the data from the specified file path
    try:
        original_dataframe = pd.read_csv(file_path, on_bad_lines='skip')
    except pd.errors.ParserError as parse_error:
        print(f"Error parsing file: {parse_error}")
        return None, None

    # Transpose the data and rename the first column to 'Years'
    transposed_dataframe = original_dataframe.transpose().reset_index()
    transposed_dataframe.columns = transposed_dataframe.iloc[0]
    transposed_dataframe = transposed_dataframe.drop(transposed_dataframe.index[0])

    return original_dataframe, transposed_dataframe

def get_indicator_dataframes(dataframe, indicators):
    """
    Filters the dataframe for the specified indicators.

    Args:
    dataframe (DataFrame): The original dataframe.
    indicators (list): List of indicators to filter.

    Returns:
    dict: Dictionary of dataframes for each indicator.
    """
    indicator_dfs = {}
    for indicator in indicators:
        indicator_dfs[indicator] = dataframe[dataframe['Indicator Name'] == indicator]
    return indicator_dfs

def merge_and_clean_dataframes(dataframes_dict):
    """
    Merges and cleans the dataframes.

    Args:
    dataframes_dict (dict): Dictionary of dataframes to merge and clean.

    Returns:
    DataFrame: The cleaned and merged dataframe.
    """
    merged_df = pd.concat(dataframes_dict.values())

    # Reset index, replace '..' with NaN, drop last 3 years, and drop redundant columns
    merged_df.reset_index(drop=True, inplace=True)
    merged_df.replace('..', np.nan, inplace=True)
    merged_df = merged_df.iloc[:, :-3]
    merged_df.fillna(0, inplace=True)
    merged_df.drop(['Indicator Code', 'Country Code'], axis=1, inplace=True)
    global years

    # Convert data to numeric and round off
    years = merged_df.columns[2:]
    merged_df[years] = np.abs(merged_df[years].astype('float').round(2))

    return merged_df
def plot_group_sizes(dataframe):
    """
    Plots the size of each group in the dataframe.

    Args:
    dataframe (DataFrame): The dataframe to plot.
    """
    dataframe.groupby(['Indicator Name'])['1960'].size().plot(kind='bar')
    plt.title('Size of each group')
    plt.xticks(rotation=90)
    plt.show()

# Usage
df, t_df = read_climate_change_data('climate.csv')
indicators = [
    'Renewable energy consumption (% of total final energy consumption)',
    'Access to electricity (% of population)',
    'Population growth (annual %)'
]
indicator_dfs = get_indicator_dataframes(df, indicators)
merged_df = merge_and_clean_dataframes(indicator_dfs)
plot_group_sizes(merged_df)
# Plotting the means of the 3 groups of the data over the years

def plot_group_means_by_year(dataframe, year, ax, color, title):
    """
    Plots the mean of each group for a specified year.

    Args:
    dataframe (DataFrame): The dataframe containing the data.
    year (str): The year for which to plot the data.
    ax (Axes): The matplotlib Axes object to plot on.
    color (str): The color of the bars in the plot.
    title (str): The title of the plot.
    """
    dataframe.groupby(['Indicator Name'])[year].mean().plot(
        kind='bar', ax=ax, color=color)
    ax.title.set_text(title)
    ax.set_ylabel('Mean Value')

# Plotting the data
fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Adjust the size as needed

plot_group_means_by_year(merged_df, '1990', axes[0], 'turquoise', 'Mean of Each Group in 1990')
plot_group_means_by_year(merged_df, '2018', axes[1], 'red', 'Mean of Each Group in 2018')

plt.subplots_adjust(wspace=0.3)  # Adjust the gap between subplots
plt.savefig('change.jpeg')
plt.show()


def calculate_skewness_kurtosis(dataframe, numeric_columns):
    """
    Calculates skewness and kurtosis for each numeric column in the dataframe.

    Args:
    dataframe (DataFrame): The dataframe to analyze.
    numeric_columns (list): List of numeric columns in the dataframe.
    """
    for col in numeric_columns:
        skewness = skew(dataframe[col])
        kurt = kurtosis(dataframe[col])
        print(f"Column {col}: Skewness = {skewness:.2f}, Kurtosis = {kurt:.2f}")

def split_data_by_indicator(dataframe, indicators):
    """
    Splits the dataframe into separate dataframes based on indicators.

    Args:
    dataframe (DataFrame): The original dataframe.
    indicators (list): List of indicators.

    Returns:
    dict: A dictionary of dataframes, each containing data for a specific indicator.
    """
    return {indicator: dataframe[dataframe['Indicator Name'] == indicator] for indicator in indicators}

def get_summary_statistics(dataframes, years):
    """
    Computes and scales the summary statistics for each dataframe.

    Args:
    dataframes (dict): Dictionary of dataframes to compute statistics for.
    years (list): List of years to consider.

    Returns:
    DataFrame: The scaled summary statistics.
    """
    stats = pd.DataFrame()
    for key, df in dataframes.items():
        stats[key] = df[years].mean(axis=0)

    # Scaling
    scaler = StandardScaler()
    stats_scaled = pd.DataFrame(scaler.fit_transform(stats), columns=stats.columns)
    stats_scaled.index = stats.index
    return stats_scaled

def plot_trends(stats_scaled):
    """
    Plots the general trend of indicators over the years.

    Args:
    stats_scaled (DataFrame): The dataframe containing scaled statistics.
    """
    plt.figure(figsize=(10, 6))
    for column in stats_scaled.columns:
        plt.plot(stats_scaled.index, stats_scaled[column], label=column)
    plt.legend(loc='best')
    plt.xticks(rotation=45)
    plt.ylabel('Relative Change')
    plt.title('Change of Indicators Over the Years Globally')
    plt.savefig('trend.jpeg')
    plt.show()

# Main execution
numeric_cols = merged_df.select_dtypes(include=np.number).columns.tolist()
calculate_skewness_kurtosis(merged_df, numeric_cols)

indicators = [
    'Renewable energy consumption (% of total final energy consumption)',
    'Access to electricity (% of population)',
    'Population growth (annual %)'
]
indicator_dfs = split_data_by_indicator(merged_df, indicators)

years = merged_df.columns[3:]  # Assuming year columns start from the fourth column
stats_scaled = get_summary_statistics(indicator_dfs, years)
plot_trends(stats_scaled)
def prepare_country_data(dataframe, country_name):
    """
    Prepares the data for a specific country.

    Args:
    dataframe (DataFrame): The original dataframe.
    country_name (str): The name of the country.

    Returns:
    DataFrame: The prepared dataframe for the specified country.
    """
    country_df = dataframe[dataframe['Country Name'] == country_name]
    country_df = country_df.drop(['Country Name'], axis=1)
    return country_df.set_index('Indicator Name').loc[:, '2000':'2018'].T

def plot_scaled_country_data(data, country_name):
    """
    Plots the scaled data for a specific country.

    Args:
    data (DataFrame): The dataframe containing the country's data.
    country_name (str): The name of the country.
    """
    plt.figure(figsize=(10, 6))
    scaler = StandardScaler()
    
    for column in data.columns:
        # Scaling the data
        scaled_data = pd.DataFrame(scaler.fit_transform(data[[column]]), columns=[column], index=data.index)
        plt.plot(scaled_data, label=column)
    
    plt.legend(loc='best')
    plt.xticks(rotation=45)
    plt.ylabel('Relative Change')
    plt.title(f'Change of Indicators over the Years in {country_name}')
    plt.show()

def plot_correlation_heatmap(dataframe):
    """
    Plots a correlation heatmap for the provided dataframe.

    Args:
    dataframe (DataFrame): The dataframe for which to plot the heatmap.
    """
    corr_mat = dataframe.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    cbar = ax.figure.colorbar(ax.imshow(corr_mat, cmap='plasma'), ax=ax)
    ax.set_xticks(np.arange(len(corr_mat.columns)))
    ax.set_yticks(np.arange(len(corr_mat.columns)))
    ax.set_xticklabels(corr_mat.columns)
    ax.set_yticklabels(corr_mat.columns)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Adding values to the cells
    for i in range(len(corr_mat)):
        for j in range(len(corr_mat)):
            ax.text(j, i, f"{corr_mat.iloc[i, j]:.2f}", ha="center", va="center", color="black")

    ax.set_title("Correlation Heatmap")
    plt.show()

# Preparing and plotting the data for Australia
Australia = prepare_country_data(merged_df, 'Australia')
plot_scaled_country_data(Australia, 'Australia')

# Preparing and plotting the data for Canada
Canada = prepare_country_data(merged_df, 'Canada')
plot_scaled_country_data(Canada, 'Canada')


plot_correlation_heatmap(stats_scaled)

# List of additional countries to plot
additional_countries = ['India', 'Brazil', 'Japan']

# Looping through each country and plotting the data
for country in additional_countries:
    country_data = prepare_country_data(merged_df, country)
    plot_scaled_country_data(country_data, country)
    

co2_emissions_indicator = 'CO2 emissions (metric tons per capita)'

# Filtering the DataFrame for CO2 emissions data
co2_df = df[df['Indicator Name'] == co2_emissions_indicator]

# Focusing on the year 2019
year = '2019'

# Identifying the top 10 countries with the highest CO2 emissions in 2019
top10_emitting_countries = co2_df.sort_values(by=year, ascending=False).head(10)

# Plotting the data
plt.figure(figsize=(12, 6))
plt.bar(top10_emitting_countries['Country Name'], top10_emitting_countries[year])
plt.title(f'Top 10 CO2 Emitting Countries in {year}')
plt.xlabel('Country')
plt.ylabel('CO2 Emissions (metric tons per capita)')
plt.xticks(rotation=45)
plt.show()