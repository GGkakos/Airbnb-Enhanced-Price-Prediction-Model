import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kstest, normaltest
import distfit

def tranformations_calendar_data(calendar):
    """
    Function to clean the calendar data
    :param calendar: the calendar data
    :return: the cleaned calendar data
    """

    # transform the price column to float type
    calendar['price'] = calendar['price'].str.replace('$', '').str.replace(',', '').astype(float)

    # transform adjusted_price column to float type
    calendar['adjusted_price'] = calendar['adjusted_price'].str.replace('$', '').str.replace(',', '').astype(float)

    # tranfrom the date column to datetime type
    calendar['date'] = pd.to_datetime(calendar['date'])

    # transform available column to boolean type
    calendar['available'] = calendar['available'].map({'t': True, 'f': False})

    return calendar


def price_descriptive_statistics(calendar):
    """
    Function to calculate the descriptive statistics for the price column
    :param calendar: the calendar data
    :return: the descriptive statistics
    """
    
    stats = {
        'Minimum of Price': calendar['price'].min(),
        '1st Quantile of Price': calendar['price'].quantile(0.25),
        'Mean of Price': calendar['price'].mean(),
        'Median of Price': calendar['price'].median(),
        '3rd Quantile of Price': calendar['price'].quantile(0.75),
        'Maximum of Price': calendar['price'].max(),
        'Standard Deviation of Price': calendar['price'].std()
    }
    
    return stats


def compare_bookings_per_neighbourhood_count(calendar_without_price_outliers, price_outliers, neighbourhood):
            
        print('Bookings in {} without price outliers'.format(neighbourhood))
        print(calendar_without_price_outliers[calendar_without_price_outliers['neighbourhood'] == neighbourhood]['available'].value_counts())
        
        print('Bookings in {} with price outliers'.format(neighbourhood))
        print(price_outliers[price_outliers['neighbourhood'] == neighbourhood]['available'].value_counts())
        

def time_series_month_occupancy_rate_neighbourhood(calendar, neighbourhood):
    """
    Function to calculate the monthly occupancy rate for a neighbourhood
    :param calendar: the calendar data
    :param neighbourhood: the neighbourhood
    :return: the monthly occupancy rate
    """
    calendar['date'] = pd.to_datetime(calendar['date'])
    
    # filter the calendar data for the neighbourhood
    calendar_neighbourhood = calendar[calendar['neighbourhood'] == neighbourhood]
    
    # calculate the monthly occupancy rate
    monthly_occupancy_rate = calendar_neighbourhood.groupby(calendar_neighbourhood['date'].dt.to_period('M'))['available'].mean()
    
    # Convert index to a datetime index to ensure correct ordering
    monthly_occupancy_rate.index = monthly_occupancy_rate.index.to_timestamp()

    # Sort by month to ensure correct order
    monthly_occupancy_rate = monthly_occupancy_rate.sort_index()
    
    # plot the monthly occupancy rate using seaborn
    sns.set(rc={'figure.figsize':(15, 5)})
    sns.lineplot(data=monthly_occupancy_rate)
    plt.title('Monthly Occupancy Rate in {}'.format(neighbourhood))
    plt.xlabel('Month')
    plt.ylabel('Occupancy Rate')
    plt.xticks(monthly_occupancy_rate.index, monthly_occupancy_rate.index.strftime('%B'), rotation=45)
    plt.show()
        
    return monthly_occupancy_rate

def time_series_month_occupancy_rate_dataset(calendar):
    """
    Function to calculate the monthly occupancy rate for the dataset
    :param calendar: the calendar data
    :return: the monthly occupancy rate
    """
    calendar['date'] = pd.to_datetime(calendar['date'])
    
    monthly_occupancy_rate = calendar.groupby(calendar['date'].dt.to_period('M'))['available'].mean()
    
    # Convert index to a datetime index to ensure correct ordering
    monthly_occupancy_rate.index = monthly_occupancy_rate.index.to_timestamp()

    # Sort by month to ensure correct order
    monthly_occupancy_rate = monthly_occupancy_rate.sort_index()

    # Plot the monthly occupancy rate using seaborn
    sns.set(rc={'figure.figsize':(15, 5)})
    sns.lineplot(data=monthly_occupancy_rate)
    plt.title('Monthly Occupancy Rate')
    plt.xlabel('Month')
    plt.ylabel('Occupancy Rate')
    plt.xticks(monthly_occupancy_rate.index, monthly_occupancy_rate.index.strftime('%B'), rotation=45)
    plt.show()
    
    return monthly_occupancy_rate



def time_series_month_price_neighbourhood(calendar, neighbourhood):
    """
    Function to calculate the monthly price for a neighbourhood
    :param calendar: the calendar data
    :param neighbourhood: the neighbourhood
    :return: the monthly price
    """
    
    # filter the calendar data for the neighbourhood
    calendar_neighbourhood = calendar[calendar['neighbourhood'] == neighbourhood]
    
    # calculate the monthly price
    monthly_price = calendar_neighbourhood.groupby(calendar_neighbourhood['date'].dt.to_period('M'))['price'].mean()
    
    # Convert index to a datetime index to ensure correct ordering
    monthly_price.index = monthly_price.index.to_timestamp()

    # Sort by month to ensure correct order
    monthly_price = monthly_price.sort_index()
    
    # plot the monthly price using seaborn
    sns.set(rc={'figure.figsize':(15, 5)})
    sns.lineplot(data=monthly_price)
    plt.title('Monthly Price in {}'.format(neighbourhood))
    plt.xlabel('Month')
    plt.ylabel('Price')
    plt.xticks(monthly_price.index, monthly_price.index.strftime('%B'), rotation=45)
    plt.show()
    
    return monthly_price



# a function to test the normality of price in a dataset

def test_normality_price(calendar):
    """
    Function to test the normality of the price in the dataset
    :param calendar: the calendar data
    :return: the normality test results
    """
    
    # test the normality of the price using Shapiro-Wilk test
    shapiro_test = shapiro(calendar['price'])
    
    # test the normality of the price using Kolmogorov-Smirnov test
    ks_test = kstest(calendar['price'], 'norm')
    
    # test the normality of the price using D'Agostino and Pearson's test
    normal_test = normaltest(calendar['price'])
    
    return {
        'Shapiro-Wilk Test': shapiro_test,
        'Kolmogorov-Smirnov Test': ks_test,
        "D'Agostino and Pearson's Test": normal_test
    }
    
def test_normality_price_neighbourhood(calendar, neighbourhood):
    """
    Function to test the normality of the price in a neighbourhood
    :param calendar: the calendar data
    :param neighbourhood: the neighbourhood
    :return: the normality test results
    """
    
    # filter the calendar data for the neighbourhood
    calendar_neighbourhood = calendar[calendar['neighbourhood'] == neighbourhood]
    
    # test the normality of the price using Shapiro-Wilk test
    shapiro_test = shapiro(calendar_neighbourhood['price'])
    
    # test the normality of the price using Kolmogorov-Smirnov test
    ks_test = kstest(calendar_neighbourhood['price'], 'norm')
    
    # test the normality of the price using D'Agostino and Pearson's test
    normal_test = normaltest(calendar_neighbourhood['price'])
    
    return {
        'Shapiro-Wilk Test': shapiro_test,
        'Kolmogorov-Smirnov Test': ks_test,
        "D'Agostino and Pearson's Test": normal_test
    }
    

from scipy.stats import probplot

def test_normality_price_qq_plot(calendar):
    """
    Function to test the normality of the price in the dataset using Q-Q plot
    :param calendar: the calendar data
    :return: the Q-Q plot
    """
    
    # create a Q-Q plot for the price
    probplot(calendar['price'], plot=plt)
    plt.title('Q-Q Plot for Price')
    plt.show()


 # apply distfit to the price column
def test_normality_price_distfit(calendar):
    """
    Function to test the normality of the price in the dataset using distfit
    :param calendar: the calendar data
    :return: the distfit plot
    """
    
    # apply distfit to the price column
    dist = distfit(alpha=0.05)
    dist.fit_transform(calendar['price'])
    dist.plot()
    plt.show()

# a function to analyze the price distribution in the calendar dataset using distfit
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from distfit import distfit

def analyze_price_distribution_distfit(calendar):
    """
    Analyze the price distribution in the calendar dataset.

    Parameters:
    - calendar: DataFrame containing the price data.
    - n_samples: Number of bootstrap samples.
    - sample_size_fraction: Fraction of data to sample for each bootstrap sample.
    """
    
    
    n_samples = 100 

    # Sample size for each distribution fit
    sample_size = int(0.01 * len(calendar))  

    def fit_distribution_sample(data):
        dist = distfit(alpha=0.05)
        dist.fit_transform(data)
        return dist.model

    def sample_and_fit(data, sample_size):
        sample = data.sample(sample_size, random_state=np.random.randint(0, 10000))
        return fit_distribution_sample(sample)

    # Parallel processing to fit distributions on multiple samples
    results = Parallel(n_jobs=-1)(delayed(sample_and_fit)(calendar['price'], sample_size) for _ in range(n_samples))

    # Aggregate results
    best_distributions = [result['name'] for result in results]

    # Count the frequency of each best-fitting distribution
    distribution_counts = pd.Series(best_distributions).value_counts()
    print("Best fitting distributions from bootstrap samples:")
    print(distribution_counts)

    # Determine the most frequent best fit distribution
    best_fit_distribution = distribution_counts.idxmax()
    print(f"Best fit distribution: {best_fit_distribution}")
    
    # Collect parameters for the most popular distribution
    params_list = [result['params'] for result in results if result['name'] == best_fit_distribution]

    # Aggregate the parameters (mean of each parameter)
    aggregated_params = np.mean(params_list, axis=0)
    print(f"Aggregated parameters: {aggregated_params}")
    
    # Fit the best distribution to the full data
    dist_fit_function = getattr(stats, best_fit_distribution)
    best_params = dist_fit_function.fit(calendar['price'])
    print(f"Best fit parameters: {best_params}")

    # Kolmogorov-Smirnov test
    ks_stat, ks_p_value = stats.kstest(calendar['price'], best_fit_distribution, args=best_params)
    print(f"Kolmogorov-Smirnov Test for {best_fit_distribution}: Statistic={ks_stat}, p-value={ks_p_value}")

    # Plot the empirical data distribution and the fitted PDF
    x = np.linspace(calendar['price'].min(), calendar['price'].max(), 1000)
    fitted_pdf = getattr(stats, best_fit_distribution).pdf(x, *best_params)
    fitted_cdf = getattr(stats, best_fit_distribution).cdf(x, *best_params)

    plt.figure(figsize=(12, 6))
    sns.histplot(calendar['price'], bins=50, kde=True, stat='density', label='Empirical Data')
    plt.plot(x, fitted_pdf, 'r-', label=f'Fitted {best_fit_distribution} PDF')
    plt.title(f'Empirical Data and Fitted {best_fit_distribution} PDF')
    plt.xlabel('Price')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
    # Plot the empirical data distribution and the fitted CDF
    plt.figure(figsize=(12, 6))
    sns.ecdfplot(calendar['price'], label='Empirical CDF')
    plt.plot(x, fitted_cdf, 'r-', label=f'Fitted {best_fit_distribution} CDF')
    plt.title(f'Empirical Data and Fitted {best_fit_distribution} CDF')
    plt.xlabel('Price')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.show()

    # Calculate residuals
    empirical_cdf = np.arange(1, len(calendar['price']) + 1) / len(calendar['price'])
    sorted_prices = np.sort(calendar['price'])
    fitted_cdf = getattr(stats, best_fit_distribution).cdf(sorted_prices, *best_params)
    residuals = empirical_cdf - fitted_cdf

    # Plot residuals
    plt.figure(figsize=(12, 6))
    plt.scatter(sorted_prices, residuals, s=1)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Residuals of the Fitted {best_fit_distribution} Distribution')
    plt.xlabel('Price')
    plt.ylabel('Residual')
    plt.show()