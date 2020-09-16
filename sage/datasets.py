import os
import pandas as pd

github_data_url = 'https://github.com/iancovert/sage/raw/master/data/'


def airbnb():
    '''
    Airbnb listing data from Kaggle.

    Located at: https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data
    '''
    path = os.path.join(github_data_url, 'AB_NYC_2019.csv')
    df = pd.read_table(path, sep=',', header=0, index_col=None)

    # Type conversions
    df['name'] = df['name'].astype(str)
    df['host_name'] = df['host_name'].astype(str)
    df['last_review'] = pd.to_datetime(df['last_review'])
    return df


def bank():
    '''
    Bank marketing data from UCI dataset repository.

    Located at: https://archive.ics.uci.edu/ml/datasets/bank+marketing
    '''
    columns = [
        'Age', 'Job', 'Marital', 'Education', 'Default', 'Balance', 'Housing',
        'Loan', 'Contact', 'Day', 'Month', 'Duration', 'Campaign', 'Prev Days',
        'Prev Contacts', 'Prev Outcome', 'Success']
    path = os.path.join(github_data_url, 'bank-full.csv')
    df = pd.read_table(path, sep=';', header=None, index_col=None, skiprows=1,
                       names=columns)

    # Convert label.
    df['Success'] = (df['Success'] == 'yes')
    return df


def bike():
    '''
    Bike sharing dataset from Kaggle competition.

    Located at: https://www.kaggle.com/c/bike-sharing-demand
    '''
    path = os.path.join(github_data_url, 'bike.csv')
    df = pd.read_table(path, sep=',', header=0, index_col=None)
    columns = df.columns.tolist()

    # Split and remove datetime column.
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df = df.drop('datetime', axis=1)

    # Reorder and rename columns.
    df = df[['year', 'month', 'day', 'hour'] + columns[1:]]
    df.columns = list(map(str.title, df.columns))
    return df


def credit():
    '''
    German credit quality dataset from UCI dataset repository.

    Located at: https://archive.ics.uci.edu/ml/datasets/South+German+Credit+%28UPDATE%29
    '''
    columns = [
        'Checking Status', 'Duration', 'Credit History', 'Purpose',
        'Credit Amount', 'Savings Account/Bonds', 'Employment Since',
        'Installment Rate', 'Personal Status', 'Debtors/Guarantors',
        'Residence Duration', 'Property Type', 'Age',
        'Other Installment Plans', 'Housing Ownership',
        'Number Existing Credits', 'Job', 'Number Liable', 'Telephone',
        'Foreign Worker', 'Good Customer'
    ]
    path = os.path.join(github_data_url, 'SouthGermanCredit.asc')
    return pd.read_table(path, sep=' ', header=None, index_col=None,
                         names=columns, skiprows=1)
