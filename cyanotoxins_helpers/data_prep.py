def separate_date_into_mdy(df, col='DATE', drop=False, dropna=False):
    '''

    Separates the date column of a data frame into month, day, and year

    :param df: a pandas data frame
    :param col: a string of the column name containing the dates
    :param drop: a boolean of whether or not to drop col
    :param dropna: a boolean for whether or not to drop rows containing nas in column col
    :return: a pandas data frame containing the three new columns
    '''

    if dropna:
        df.dropna(subset=[col], inplace=True)

    df[col] = df[col].astype('datetime64')

    df['Month'] = df[col].dt.month.astype("int64")
    df['Year'] = df[col].dt.year
    df['Day'] = df[col].dt.day

    # df['Month'] = pd.DatetimeIndex(df[col]).month
    # df['Year'] = pd.DatetimeIndex(df[col]).year
    # df['Day'] = pd.DatetimeIndex(df[col]).day

    if drop:
        del df[col]

    return df


def generate_rolling_mean(df, col, window, rm_col_name, group_by_var, min_periods=1):
    '''
    Creates a new column in the data frame of the rolling mean of a column

    :param df: a pandas data frame
    :param col: a string of the column name to compute the rolling mean for
    :param window: an int for how big the rolling mean window should be
    :param rm_col_name: a string name for the new rolling mean column
    :param group_by_var: a string name for the variable to group by before calculating the rolling mean
    :param min_periods: an int for the minimum number of observations to calculate the rolling mean
    :return: a pandas data frame containing a new column called rm_col_name
    '''

    df[rm_col_name] = df.groupby(group_by_var)[col].rolling(window, min_periods=min_periods).mean().reset_index(0,
                                                                                                                drop=True)
    return df


def shift_columns(df, col, n, groupvar):
    '''

    :param df: a pandas data frame
    :param col: a string of the column name to shift
    :param n: the number of rows to shift by
    :param groupvar: the variable to group by before shifting rows
    :return: a pandas data frame with an additional column of the shifted values of col
    '''

    new_col_name = '{}_{}_ahead'.format(col, n)
    # df[new_col_name] = df[col].shift(-n)
    df[new_col_name] = df.groupby([groupvar])[col].shift(-n)

    return df


def read_station_lake_dictionary(file_name):
    '''


    :param file_name: the path to a tsv containing lakes and the station nearest to them
    :return: a dictionary with keys the name of the lake and values the nearest station
    '''

    station_lake_map = {}

    with open(file_name) as fin:
        for line in fin:
            line = line.rstrip().split('\t')
            station_lake_map[line[0].rstrip()] = line[1].rstrip()

    return station_lake_map


def main():
    pass


if __name__ == '__main__':
    import sys

    sys.exit(main())
