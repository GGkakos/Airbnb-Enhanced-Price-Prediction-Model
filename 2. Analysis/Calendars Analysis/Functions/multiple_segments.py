import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, kendalltau, spearmanr
from scipy import stats


## Function 1
def calculate_frequencies_and_assign_segments(df, primary_threshold, secondary_thresholds):
    
    # count occurrences in each segment for listings
    segment_counts = df.groupby(['listing_id', 'price_range']).size().reset_index(name='count')

    # calculate total counts for each listing
    total_counts = segment_counts.groupby('listing_id')['count'].sum().reset_index()
    total_counts.columns = ['listing_id', 'total_count']

    # merge total counts with segment counts
    segment_counts = segment_counts.merge(total_counts, on='listing_id')

    # calculate frequency for each segment
    segment_counts['frequency'] = segment_counts['count'] / segment_counts['total_count']

    # identify the most frequent segment for each listing
    most_frequent_segment = segment_counts.loc[segment_counts.groupby('listing_id')['frequency'].idxmax()]
    most_frequent_segment.columns = ['listing_id', 'most_frequent_segment', 'count', 'total_count', 'max_frequency']

    def assign_segments_dynamic(listing_id, segment_counts, primary_threshold):
        listing_data = segment_counts[segment_counts['listing_id'] == listing_id]
        most_frequent = listing_data.loc[listing_data['frequency'].idxmax()]
        max_frequency = most_frequent['frequency']
        
        assigned_segments = []

        # Case 1: If the most frequent segment is over the primary threshold, only one price segment is assigned since it is the most probable to represent the real price range of the listing
        if max_frequency >= primary_threshold:
            assigned_segments.append(most_frequent['price_range'])
        
        # Case 2: If the most frequent segment is between the secondary and primary thresholds
        elif secondary_thresholds <= max_frequency < primary_threshold:
            assigned_segments.append(most_frequent['price_range'])
            
            remaining_data = listing_data[listing_data['price_range'] != most_frequent['price_range']]
    
            # identify the next most frequent segment
            next_most_frequent = remaining_data.loc[remaining_data['frequency'].idxmax()]
            
            # check if the next most frequent segment meets the adjusted threshold
            if next_most_frequent['frequency'] >= secondary_thresholds:
                assigned_segments.append(next_most_frequent['price_range'])
        
        
        return assigned_segments

    # apply the dynamic assignment function
    most_frequent_segment['assigned_segments'] = most_frequent_segment.apply(
        lambda row: assign_segments_dynamic(row['listing_id'], segment_counts, primary_threshold), axis=1
    )

    return most_frequent_segment[['listing_id', 'most_frequent_segment', 'assigned_segments']]


## Function 2
def analyze_segments(n_price_segments):
 
    primary_thresholds = np.arange(0.50, 0.80, 0.05)  # from 50% to 75% with a step of 5%
    secondary_thresholds = np.arange(0.30, 0.50, 0.05)  # from 30% to 45% with a step of 5%

    results = []
    two_segments_frequency = []

    for p_th in primary_thresholds:
        for s_th in secondary_thresholds:
            if p_th > s_th:  # primary is always larger than secondary
                assigned_segments = calculate_frequencies_and_assign_segments(n_price_segments, p_th, s_th)
                results.append({'Primary Threshold': p_th, 'Secondary Threshold': s_th, 'Assigned Segments': assigned_segments})

    # Print the results
    for result in results:
        print("Primary Threshold: ", result['Primary Threshold'])
        print("Secondary Threshold: ", result['Secondary Threshold'])
        print(result['Assigned Segments']['assigned_segments'].value_counts())
        print("\n")

    # compute the frequency of assigned segments with two elements
    for result in results:
        two_segments_frequency.append(result['Assigned Segments']['assigned_segments'].apply(lambda x: len(x) == 2).sum())
        
    print(two_segments_frequency)
    print("Median frequency of assigned segments with two elements:", np.median(two_segments_frequency))
    


## Function 3
def analyze_price_segments(n_price_segments_calculated):
    # Print the number of empty lists in assigned_segments
    num_empty_lists = n_price_segments_calculated['assigned_segments'].apply(lambda x: len(x) == 0).sum()
    print("Number of empty lists in assigned_segments:", num_empty_lists)

    # Create a column for segment combinations as a string
    n_price_segments_calculated['segment_combinations'] = n_price_segments_calculated['assigned_segments'].apply(lambda x: ', '.join(sorted(x)))

    # Calculate the frequency of each combination
    combinations_count = n_price_segments_calculated['segment_combinations'].value_counts()

    # Plot the frequency of assigned price segment combinations
    plt.figure(figsize=(10, 6))
    sns.barplot(x=combinations_count.index, y=combinations_count.values, palette='viridis')
    plt.title('Frequency of Assigned Price Segment Combinations')
    plt.xlabel('Price Segment Combinations')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()
    
    
# Function 4 (To get the final dataframe after the adjustment of listings with mulitple price segments)
def merge_and_concatenate(exploded_df, cal_n_year_assigned_segments, cal_year_segments_assigned_one_price_ranges):
    # Function to merge prices with segments
    def merge_prices_with_segments(exploded_df, cal_n_year_assigned_segments):
        # Rename columns for clarity and consistency
        cal_n_year_assigned_segments = cal_n_year_assigned_segments.rename(columns={'price_range': 'assigned_segments'})
        
        # Merge the exploded_df with cal_2020_assigned_segments on listing_id and assigned_segments
        merged_df = pd.merge(exploded_df, cal_n_year_assigned_segments, on=['listing_id', 'assigned_segments'], how='left')
        
        return merged_df
    
    # From exploded_df keep only listing_id and assigned_segments columns
    exploded_df_only_listing_id_segments = exploded_df[['listing_id', 'assigned_segments']]
    
    # Merge prices with segments
    prices_after_assigned_segments = merge_prices_with_segments(exploded_df_only_listing_id_segments, cal_n_year_assigned_segments)

    
    # Concatenate the filtered data and the merged data
    cal_year_df = pd.concat([cal_year_segments_assigned_one_price_ranges, prices_after_assigned_segments], axis=0)
    
    # Fill NA values in price_range column with the corresponding values in assigned_segments column
    cal_year_df['price_range'] = cal_year_df['price_range'].fillna(cal_year_df['assigned_segments'])
    
    # Drop assigned_segments column from cal_year_df
    cal_year_df.drop('assigned_segments', axis=1, inplace=True)
    
    # Order price_range: Budget < Mid-Range < High-End < Luxury < Superior
    cat_dtype = pd.CategoricalDtype(categories=['Budget', 'Mid-Range', 'High-End', 'Luxury', 'Superior'], ordered=True)
    cal_year_df['price_range'] = cal_year_df['price_range'].astype(cat_dtype)
    
    return cal_year_df



## FOR COMPARISONS BEFORE AND AFTER THE ADJUSTMENT

### PLOT FOR PRICE AND COEFFICIENT OF VARIATION OF PRICE BY PRICE SEGMENT
def coefficient_price_variation_listing_segment_plot(calendar_df_with_price_range):
    # Calculate the coefficient of variation by listing_id and price_range
    price_cv_by_price_listing_id_price_segment = calendar_df_with_price_range.groupby(['listing_id', 'price_range'])['price'].agg(lambda x: np.std(x) / np.mean(x)).reset_index()
    
    # Drop rows with NaN price_cv values
    price_cv_by_price_listing_id_price_segment = price_cv_by_price_listing_id_price_segment.dropna()
    
    # Plot the price coefficient of variation by price segment
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(data=price_cv_by_price_listing_id_price_segment, x='price_range', y='price', palette='viridis', ci=None)
    plt.title('Price Coefficient of Variation by Price Segment')
    plt.xlabel('Price Segment')
    plt.ylabel('Price Coefficient of Variation')
    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.2f'), 
                          xy=(p.get_x() + p.get_width() / 2., p.get_height()), 
                          ha='center', va='center', 
                          size=12, xytext=(0, 8), 
                          textcoords='offset points')
    plt.show()

    # Calculate the mean price by listing_id and price_range
    mean_price_by_listing_id_price_segment = calendar_df_with_price_range.groupby(['listing_id', 'price_range'])['price'].mean()
    
    # Transform to a DataFrame
    mean_price_by_listing_id_price_segment_df = mean_price_by_listing_id_price_segment.reset_index()
    
    # Drop NaN values
    mean_price_by_listing_id_price_segment_df = mean_price_by_listing_id_price_segment_df.dropna()
    
    # Plot the mean price by price segment
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(data=mean_price_by_listing_id_price_segment_df, x='price_range', y='price', palette='viridis', ci=None)
    plt.title('Mean Price by Price Segment')
    plt.xlabel('Price Segment')
    plt.ylabel('Mean Price')
    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.2f'), 
                          xy=(p.get_x() + p.get_width() / 2., p.get_height()), 
                          ha='center', va='center', 
                          size=12, xytext=(0, 8), 
                          textcoords='offset points')
    plt.show()
    
### PLOT FOR COEFFICIENT OF VARIATION OF PRICE BY NEIGHBOURHOOD AND PRICE SEGMENT (returns the dataframe)
def plot_coefficient_of_variation(cal_2020_final):
    # Calculate the coefficient of variation by listing_id and price_range
    coefficient_of_variation_by_listing_neighbourhood_price_segment = cal_2020_final.groupby(['listing_id', 'price_range'])['price'].std() / cal_2020_final.groupby(['listing_id', 'price_range'])['price'].mean()
    
    # Transform to a DataFrame
    coefficient_of_variation_by_listing_neighbourhood_price_segment_df = coefficient_of_variation_by_listing_neighbourhood_price_segment.reset_index()
    
    # Drop NaN values
    coefficient_of_variation_by_listing_neighbourhood_price_segment_df = coefficient_of_variation_by_listing_neighbourhood_price_segment_df.dropna()
    
    # Merge neighborhood information
    coefficient_of_variation_by_listing_neighbourhood_price_segment_df = pd.merge(
        coefficient_of_variation_by_listing_neighbourhood_price_segment_df,
        cal_2020_final[['listing_id', 'neighbourhood']].drop_duplicates(),
        on='listing_id',
        how='left'
    )
    
    # Drop duplicates
    coefficient_of_variation_by_listing_neighbourhood_price_segment_df = coefficient_of_variation_by_listing_neighbourhood_price_segment_df.drop_duplicates()
    
    # Rename columns for clarity
    coefficient_of_variation_by_listing_neighbourhood_price_segment_df.rename(columns={0: 'coefficient_of_variation'}, inplace=True)
    
    # Plot the coefficient of variation by neighborhood and price_range
    plt.figure(figsize=(10, 6))
    sns.barplot(data=coefficient_of_variation_by_listing_neighbourhood_price_segment_df, x='neighbourhood', y='price', hue='price_range', palette='viridis', ci=None)
    plt.title('Coefficient of Variation by Neighbourhood and Price Segment')
    plt.xlabel('Neighbourhood')
    plt.ylabel('Coefficient of Variation')
    plt.xticks(rotation=90)
    plt.show()
    
    return coefficient_of_variation_by_listing_neighbourhood_price_segment_df



### Calculation of average_price by listing id and price segment
### Calculation of average coefficient of variation by listing id and price segment
### Single figure with two plots. One for average price by neighbourhood and one average coefficient of variation by neighbourhood
### Returns the dataframe
def price_by_listing_neighbourhood_price_segment_df_and_plot_mean_price_and_cv(cal_2020_final):
    # Calculate the mean price by listing_id and price_range
    mean_price_by_listing_id_price_segment = cal_2020_final.groupby(['listing_id', 'price_range'])['price'].mean()
    
    # Transform to a DataFrame
    mean_price_by_listing_id_price_segment_df = mean_price_by_listing_id_price_segment.reset_index()
    
    # Drop NaN values
    mean_price_by_listing_id_price_segment_df = mean_price_by_listing_id_price_segment_df.dropna()
    
    # Calculate mean coefficient of price variation by listing_id and price_range
    mean_coefficient_of_variation_by_listing_id_price_segment_merge = cal_2020_final.groupby(['listing_id', 'price_range'])['price'].std() / cal_2020_final.groupby(['listing_id', 'price_range'])['price'].mean()
    
    # Transform to DataFrame
    mean_coefficient_of_variation_by_listing_id_price_segment_merge = mean_coefficient_of_variation_by_listing_id_price_segment_merge.reset_index()
    mean_coefficient_of_variation_by_listing_id_price_segment_merge.rename(columns={'price': 'cv'}, inplace=True)
    
    # Merge the coefficient of variation with the mean price DataFrame
    mean_price_by_listing_id_price_segment_df = pd.merge(mean_price_by_listing_id_price_segment_df, mean_coefficient_of_variation_by_listing_id_price_segment_merge, on=['listing_id', 'price_range'], how='left')
    
    # Merge neighborhood information
    mean_price_by_listing_id_price_segment_df = pd.merge(mean_price_by_listing_id_price_segment_df, cal_2020_final[['listing_id', 'neighbourhood']], on='listing_id', how='left')
    
    # Drop duplicates
    mean_price_by_listing_id_price_segment_df = mean_price_by_listing_id_price_segment_df.drop_duplicates()
    
    # Plot the results
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    # Plot 1: Average Price by Neighbourhood
    sns.barplot(
        data=mean_price_by_listing_id_price_segment_df,
        x='neighbourhood',
        y='price',
        palette='viridis',
        ci=None,
        order=mean_price_by_listing_id_price_segment_df.groupby('neighbourhood')['price'].mean().sort_values(ascending=False).index,
        ax=axes[0]
    )
    axes[0].set_title('Average Price by Neighbourhood')
    axes[0].set_xlabel('Neighbourhood')
    axes[0].set_ylabel('Average Price')
    axes[0].tick_params(axis='x', rotation=90)

    # Plot 2: Average Coefficient of Variation by Neighbourhood
    sns.barplot(
        data=mean_price_by_listing_id_price_segment_df,
        x='neighbourhood',
        y='cv',
        palette='viridis',
        ci=None,
        order=mean_price_by_listing_id_price_segment_df.groupby('neighbourhood')['cv'].mean().sort_values(ascending=False).index,
        ax=axes[1]
    )
    axes[1].set_title('Average Coefficient of Variation by Neighbourhood')
    axes[1].set_xlabel('Neighbourhood')
    axes[1].set_ylabel('Average Coefficient of Variation')
    axes[1].tick_params(axis='x', rotation=90)

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    plt.show()
    
    return mean_price_by_listing_id_price_segment_df



## GETS AS INPUT THE DATAFRAME FROM THE FUNCTION ABOVE
### Correlations between price and coefficient of variation separately for the average values by listing ids and neighbourhoods 
def correlations_between_price_and_cv(mean_price_by_listing_id_price_segment_df):
    # Calculate the average price_x and average price_y by neighborhood
    mean_price_by_neighbourhood = mean_price_by_listing_id_price_segment_df.groupby('neighbourhood').agg({'price': 'mean', 'cv': 'mean'}).reset_index()

    # Transform to DataFrame
    mean_price_by_neighbourhood_df = mean_price_by_neighbourhood.reset_index(drop=True)
    print(mean_price_by_neighbourhood_df)
    
    # Scatterplot with trend line for price_x and price_y in mean_price_by_neighbourhood_df
    plt.figure(figsize=(10, 6))
    sns.regplot(data=mean_price_by_neighbourhood_df, x='price', y='cv', ci=None, line_kws={'color': 'red'})
    plt.title('Average Price vs. Coefficient of Variation of Price (Neighbourhood)')
    plt.xlabel('Average Price')
    plt.ylabel('Average Coefficient of Price Variation')
    plt.show()

    # Correlation of price_x and price_y in mean_price_by_neighbourhood_df
    correlation, p_value = pearsonr(mean_price_by_neighbourhood_df['price'], mean_price_by_neighbourhood_df['cv'])
    print(f'Neighbourhood Correlation: {correlation}')
    print(f'Neighbourhood P-Value: {p_value}')

    
    
    # Scatterplot of mean price and mean coefficient of price variation and add a trend line
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=mean_price_by_listing_id_price_segment_df, x='price', y='cv')
    sns.regplot(data=mean_price_by_listing_id_price_segment_df, x='price', y='cv', scatter=False, color='red')
    plt.title('Mean Price vs. Mean Coefficient of Variation (Listings)')
    plt.xlabel('Mean Price')
    plt.ylabel('Mean Coefficient of Variation')
    plt.show()

    # Calculate the Pearson correlation coefficient and the p-value for listings
    corr, p_value_listings = pearsonr(mean_price_by_listing_id_price_segment_df['price'], mean_price_by_listing_id_price_segment_df['cv'])
    print(f'Listing Pearson Correlation: {corr}')
    print(f'Listing P-Value: {p_value_listings}')
    
    # Calculate Kendall's Tau and Spearman correlation for listings
    kendall_tau = kendalltau(mean_price_by_listing_id_price_segment_df['price'], mean_price_by_listing_id_price_segment_df['cv'])
    spearman_corr = spearmanr(mean_price_by_listing_id_price_segment_df['price'], mean_price_by_listing_id_price_segment_df['cv'])
    print(f'Kendall Tau: {kendall_tau}')
    print(f'Spearman Correlation: {spearman_corr}')



### Calculate the average price by listing_id and price segment and save the final DataFrame to a CSV file if specified 
def calculate_average_price_by_listing_price_segment_neighbourhood(cal_2020_final, output_csv=False):
    # Calculate the average price by listing_id and price_range
    average_price_by_listing_id_price_range = cal_2020_final.groupby(['listing_id', 'price_range'])['price'].mean()

    # Transform to DataFrame
    average_price_by_listing_id_price_range_df = average_price_by_listing_id_price_range.reset_index()
    
    # Drop rows with NaN values in the price column
    average_price_by_listing_id_price_range_df = average_price_by_listing_id_price_range_df.dropna(subset=['price'])
    
    # Merge with cal_2020_final to get neighbourhood
    average_price_by_listing_id_price_range_df = pd.merge(
        average_price_by_listing_id_price_range_df,
        cal_2020_final[['listing_id', 'neighbourhood']],
        on='listing_id',
        how='left'
    )
    
    # Drop duplicates
    average_price_by_listing_id_price_range_df = average_price_by_listing_id_price_range_df.drop_duplicates()
    
    # Rename columns
    average_price_by_listing_id_price_range_df = average_price_by_listing_id_price_range_df.rename(
        columns={'price_range': 'price_segment', 'price': 'mean_price'}
    )
    
    # Save the final DataFrame to a CSV file if specified
    if output_csv:
        average_price_by_listing_id_price_range_df.to_csv('average_price_by_listing_id_price_range_df.csv', index=False)
    
    return average_price_by_listing_id_price_range_df


### plots for number of listing separately for price segment and neighbourhood, price segment and neighbourhood
## receives as dataframa the output from the function above
def number_of_listing_different_plots(average_price_by_listing_id_price_range_df):
    plt.figure(figsize=(15, 8))
    # Grouping and counting unique listings per neighbourhood
    grouped_data_neigh = average_price_by_listing_id_price_range_df.groupby(['neighbourhood'])['listing_id'].count().reset_index()
    # Sorting the data for better visualization
    grouped_data_neigh.sort_values(by='listing_id', ascending=False, inplace=True)
    # Creating the plot
    sns.barplot(data=grouped_data_neigh, y='listing_id', x='neighbourhood', palette='viridis', ci=None, order=grouped_data_neigh['neighbourhood'])
    plt.title('Number of Unique Listings by Neighbourhood')
    plt.xlabel('Neighbourhood')
    plt.ylabel('Number of Unique Listings')
    plt.xticks(rotation=90)
    plt.show()
    
    # Plot the counts of each price segment
    plt.figure(figsize=(10, 6))
    sns.countplot(data=average_price_by_listing_id_price_range_df, x='price_segment', palette='viridis')
    plt.title('Frequency of Price Segments')
    plt.xlabel('Price Segment')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(15, 8))
    # Grouping and counting unique listings per neighbourhood and price segment
    grouped_data = average_price_by_listing_id_price_range_df.groupby(['neighbourhood', 'price_segment'])['listing_id'].count().reset_index()
    # Sorting the data for better visualization
    grouped_data.sort_values(by='listing_id', ascending=False, inplace=True)
    # Creating the plot
    sns.barplot(data=grouped_data, y='listing_id', x='neighbourhood', hue='price_segment', palette='viridis')
    plt.title('Number of Unique Listings by Neighborhood and Price Segment')
    plt.xlabel('Neighbourhood')
    plt.ylabel('Number of Unique Listings')
    plt.legend(title='Price Segment')
    plt.xticks(rotation=90)
    plt.show()


#### PLOT THE DISTRIBUTION OF MEAN PRICE FOR EACH SEGMENT AND APPLY BOX-COX TRANSFORMATION
def plot_price_distributions_and_boxcox(average_price_by_listing_id_price_range_df):
    for segment in average_price_by_listing_id_price_range_df['price_segment'].unique():
        segment_data = average_price_by_listing_id_price_range_df[average_price_by_listing_id_price_range_df['price_segment'] == segment]
        
        # Plot distribution of mean_price for each segment
        plt.figure(figsize=(10, 6))
        sns.histplot(segment_data['mean_price'], bins=100, kde=True)
        plt.title(f'Mean Price Distribution for {segment}')
        plt.xlabel('Mean Price')
        plt.ylabel('Frequency')
        plt.show()

        # Apply Box-Cox transformation
        boxcox_transformed, _ = stats.boxcox(segment_data['mean_price'] + 1)  # Adding 1 to avoid issues with zero values
        
        # Plot the Box-Cox transformed distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(boxcox_transformed, bins=100, kde=True)
        plt.title(f'Box-Cox Transformed Mean Price Distribution for {segment}')
        plt.xlabel('Box-Cox Transformed Mean Price')
        plt.ylabel('Frequency')
        plt.show()