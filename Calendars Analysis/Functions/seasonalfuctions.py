import pandas as pd

def calculate_monthly_price_per_neighbourhood(df, year):
    # Ensure the date column is of datetime type
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter the data for the specified year
    df = df[df['date'].dt.year == year]
    
    # Extract year and month from the date
    df['year_month'] = df['date'].dt.to_period('M')
    
    # Calculate the average price per neighbourhood per month
    monthly_avg_prices = df.groupby(['neighbourhood', 'year_month'])['price'].mean().reset_index()
    
    # Pivot the DataFrame to have months as columns for easier percentage change calculation
    pivot_df = monthly_avg_prices.pivot(index='neighbourhood', columns='year_month', values='price')
    
    return pivot_df



def calculate_monthly_percentage_change(df, year):
    # Ensure the date column is of datetime type
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter the data for the specified year
    df = df[df['date'].dt.year == year]
    
    # Extract year and month from the date
    df['year_month'] = df['date'].dt.to_period('M')
    
    # Calculate the average price per neighbourhood per month
    monthly_avg_prices = df.groupby(['neighbourhood', 'year_month'])['price'].mean().reset_index()
    
    # Pivot the DataFrame to have months as columns for easier percentage change calculation
    pivot_df = monthly_avg_prices.pivot(index='neighbourhood', columns='year_month', values='price')
    
    # Calculate the percentage change month-to-month
    pct_change_df = pivot_df.pct_change(axis=1) * 100
    
    # Remove the first column (January) as we want to start from February
    pct_change_df = pct_change_df.iloc[:, 1:]
    
    # Format the DataFrame to have clean column names (months)
    pct_change_df.columns = pct_change_df.columns.to_series().astype(str)
    
    # Reset the index for a cleaner look
    pct_change_df.reset_index(inplace=True)
    
    return pct_change_df