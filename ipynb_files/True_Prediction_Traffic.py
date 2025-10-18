import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import random

# Makes the only dates in the data frame during ski season
def make_dataframe_calender(data):
    data["FormattedDate"] = pd.to_datetime(data["FormattedDate"])
    data["WhatMonth"] = data["FormattedDate"].dt.month
    months = [11, 12, 1, 2, 3, 4, 5]
    data = data[data["WhatMonth"].isin(months)]
    return data

# Makes my data only weekend days
def to_weekend(data):
    data = data[data['FormattedDate'].dt.dayofweek >= 5]
    return data

# Generates the snow data that I will compare the traffic data with
def graphSnow_x(x=7):
    file = "GRANBY.csv"
    new_file = pd.read_csv(file)
    data = pd.DataFrame(new_file)
    file = "FRISCO.csv"
    new_file = pd.read_csv(file)
    other_data = pd.DataFrame(new_file)
    data = pd.concat([data, other_data])
    data['DATE'] = pd.to_datetime(data['DATE'])
    data['DOY'] = data['DATE'].dt.dayofyear
    data['YEAR'] = data['DATE'].dt.year
    data['PRCP'] = pd.to_numeric(data['PRCP'], errors='coerce')
    data['SNOW'] = pd.to_numeric(data['SNOW'], errors='coerce')
    data.dropna(subset=['SNOW'], inplace=True)
    data['SNOW_DAY_SUM'] = data['SNOW'].rolling(window=x).sum()
    return data

# Gets the data from 2014-2024 for the given type of data to extract from
# I didn't use 2020 because covid skewed my data so much it changed my answers
def weekdays_total_graph(type="AF"):
    ret_data = pd.DataFrame()
    if(type == "AF"):
        for j in range(2014, 2025):
            if j != 2020:
                base_data = AF_data(j)
                base_data = base_data[base_data["COUNTDIR"] == 'S']
                base_data.reset_index(drop=True, inplace=True)
                base_data = std_df(base_data)
                base_data = mean_df(base_data)
                base_data['FormattedDate'] = pd.to_datetime(base_data['FormattedDate'])
                # base_data['DOY'] = base_data['FormattedDate'].dt.dayofyear
                # base_data['YEAR'] = base_data['FormattedDate'].dt.year
                # base_data = base_data[base_data['FormattedDate'].dt.month.isin([datetime.now().month])]
                # base_data = base_data[base_data['FormattedDate'].dt.month.isin([11,12,1,2,3,4,5])]
                ret_data = pd.concat([ret_data, base_data])
    elif(type == "ET"):
        for j in range(2014, 2025):
            if j != 2020:
                base_data = ET_data(j)
                base_data = base_data[base_data["COUNTDIR"] == 'S']
                base_data.reset_index(drop=True, inplace=True)
                base_data = std_df(base_data)
                base_data = mean_df(base_data)
                base_data['FormattedDate'] = pd.to_datetime(base_data['FormattedDate'])
                base_data['DOY'] = base_data['FormattedDate'].dt.dayofyear
                base_data['YEAR'] = base_data['FormattedDate'].dt.year
                base_data = base_data[base_data['FormattedDate'].dt.month.isin([datetime.now().month])]
                base_data = base_data[base_data['FormattedDate'].dt.month.isin([11,12,1,2,3,4,5])]
                ret_data = pd.concat([ret_data, base_data])
    elif(type == "BP"):
        for j in range(2014, 2025):
            if j != 2020:
                base_data = BP_data(j)
                base_data = base_data[base_data["COUNTDIR"] == 'S']
                base_data.reset_index(drop=True, inplace=True)
                base_data = std_df(base_data)
                base_data = mean_df(base_data)
                base_data['FormattedDate'] = pd.to_datetime(base_data['FormattedDate'])
                base_data['DOY'] = base_data['FormattedDate'].dt.dayofyear
                base_data['YEAR'] = base_data['FormattedDate'].dt.year
                base_data = base_data[base_data['FormattedDate'].dt.month.isin([datetime.now().month])]
                base_data = base_data[base_data['FormattedDate'].dt.month.isin([11,12,1,2,3,4,5])]
                ret_data = pd.concat([ret_data, base_data])
    return ret_data


# Gets the after-floyd data set
def AF_data(inp):
    inp = str(inp)
    Csv_Import = "AFTERFLOYD_TD\\" + inp + ".csv"
    csv_read = pd.read_csv(Csv_Import)
    return csv_read

# Gets the Berthoud-Pass data set
def BP_data(inp):
    inp = str(inp)
    Csv_Import = "BERTHODPASS_TD\\"+ inp + ".csv"
    csv_read = pd.read_csv(Csv_Import)
    csv_output = pd.DataFrame(csv_read)
    return csv_output

# Gets the Eisenhower-Tunnel data set
def ET_data(inp):
    inp = str(inp)
    Csv_Import = "EISENHOWERTUNNEL_TD\\"+ inp + ".csv"
    csv_read = pd.read_csv(Csv_Import)
    csv_output = pd.DataFrame(csv_read)
    return csv_output

# Drops unused data
def remove_counts(data):
    return data.drop(["COUNTSTATIONID","COUNTDATE","COUNTDIR","FormattedDate"], axis=1)

# Adds a collumn called mean, that is the average of the hours
def mean_df(data):
    data['Mean'] = data.loc[:, 'HOUR0':'HOUR23'].mean(axis=1)
    return data

# Adds a collumn called Std, that is the Std of the hours
def std_df(data):
    data['Std'] = data.loc[:, 'HOUR0':'HOUR23'].std(axis=1)
    return data

# Makes a dataframe that I can use consistently 
def make_total_data(hour=7, t="AF",mean=False, window=7,weekend=False):
    if(not mean):
        hour = f"HOUR{hour}"
        traffic_data = weekdays_total_graph(type=t)
        snow_data = graphSnow_x(window)
        snow_data.rename(columns={'DOY': 'Snow_DOY', 'YEAR': 'Snow_YEAR'}, inplace=True)
        merged_data = pd.merge(traffic_data, snow_data, left_on=['DOY', 'YEAR'], right_on=['Snow_DOY', 'Snow_YEAR'], how='inner')
        if(weekend):
            merged_data = to_weekend(merged_data)        
        new_data = merged_data.loc[:, [hour,'SNOW_DAY_SUM', 'Snow_DOY']]
        new_data.dropna(subset=['SNOW_DAY_SUM'], inplace=True)
    else:
        traffic_data = weekdays_total_graph(type=t)
        snow_data = graphSnow_x(window)
        snow_data.rename(columns={'DOY': 'Snow_DOY', 'YEAR': 'Snow_YEAR'}, inplace=True)
        merged_data = pd.merge(traffic_data, snow_data, left_on=['DOY', 'YEAR'], right_on=['Snow_DOY', 'Snow_YEAR'], how='inner')
        if(weekend):
            merged_data = to_weekend(merged_data)
        new_data = merged_data.loc[:, ["Mean",'SNOW_DAY_SUM', 'Snow_DOY']]
        new_data.dropna(subset=['SNOW_DAY_SUM'], inplace=True)
    return new_data

def WRANGLE_FAT_DATA():
    ret_data = pd.DataFrame()
    for j in range(2014, 2025):
        if j != 2020:
            base_data = AF_data(j)
            base_data.reset_index(drop=True, inplace=True)
            # base_data = std_df(base_data)
            # base_data = mean_df(base_data)
            base_data['Date'] = pd.to_datetime(base_data['FormattedDate'])
            # base_data['DOY'] = base_data['FormattedDate'].dt.dayofyear
            # base_data['YEAR'] = base_data['FormattedDate'].dt.year
            # base_data = base_data[base_data['FormattedDate'].dt.month.isin([datetime.now().month])]
            # base_data = base_data[base_data['FormattedDate'].dt.month.isin([11,12,1,2,3,4,5])]
            ret_data = pd.concat([ret_data, base_data])

    # Melt the hourly columns into long format
    hourly_columns = [f"HOUR{i}" for i in range(24)]
    df_long = ret_data.melt(
        id_vars=["Date", "COUNTDIR"],
        value_vars=hourly_columns,
        var_name="HOUR",
        value_name="TRAFFIC"
    )

    # Convert HOUR column to numeric hour
    df_long["HOUR"] = df_long["HOUR"].str.extract("(\d+)").astype(int)

    # Rename columns for clarity
    df_long = df_long.rename(columns={"Date": "DATE", "COUNTDIR": "Direction"})

    # Optional: sort by date and hour
    df_long = df_long.sort_values(by=["DATE", "HOUR", "Direction"])

    # Save to new CSV
    snow_data = graphSnow_x()
    snow_data = snow_data.drop(columns={'STATION', 'DAPR' , 'MDPR', 'DOY', 'YEAR'})
    snow_data.to_csv(f'SNOW_DATA_PER_DAY.csv', index=False)
    
    merged = pd.merge(df_long, snow_data, on="DATE", how="left")
    
    # merged_data = pd.merge(traffic_data, snow_data, left_on=['DOY', 'YEAR'], right_on=['Snow_DOY', 'Snow_YEAR'], how='inner')
        # new_data = merged_data.loc[:, ["Mean",'SNOW_DAY_SUM', 'Snow_DOY']]
        # new_data.dropna(subset=['SNOW_DAY_SUM'], inplace=True)
        
    merged.to_csv(f'MERGED_DATA_TRAFFIC_SNOW.csv', index=False)


def combine_traffic_snow():
    traffic = pd.read_csv("TRAFFIC_DATA_PER_HOUR.csv")
    weather = pd.read_csv("SNOW_DATA.csv")

    traffic["DATE"] = pd.to_datetime(traffic["DATE"])
    weather["DATE"] = pd.to_datetime(weather["DATE"])

    weather_clean = weather.drop_duplicates(subset="DATE", keep="first")
    merged = pd.merge(traffic, weather_clean, on="DATE", how="left")
    merged.to_csv("traffic_with_weather.csv", index=False)


def look_for_simularities():
    traffic = pd.read_csv("CRASH_DATA_2021_2024.csv")
    
    # traffic = fix_driver_age(traffic)
    # traffic = make_values_for_conditions(traffic)
    # traffic = make_values_for_sex(traffic)
    
    # traffic = traffic.drop(columns={'Latitude','Longitude','Driver Sex', 'Condition'})
    
    traffic['Time'] = traffic['Time'].round().astype('Int64')
    traffic['Vehicles'] = traffic['Vehicles'].round().astype('Int64')
    
    traffic.to_csv('TATATTATTAATATTATTA.csv', index=False)


def fix_driver_age(traffic):
    traffic_conditions = traffic['Driver Age']
    traffic_conditions = traffic_conditions.map(lambda x: float('nan') if x < 16 else x)
    traffic_conditions_mean = traffic_conditions.dropna()
    mean = (traffic_conditions_mean.mean())
    std = (traffic_conditions_mean.std())
    
    
    def generate_age(mean, std_dev):
        return np.random.normal(loc=mean, scale=std_dev)
    
    
    traffic['Driver Age'] = traffic['Driver Age'].apply(
        lambda x: generate_age(mean, std) if ((pd.isna(x)) or (x < 16)) else x
    )
    traffic['Driver Age'] = traffic['Driver Age'].map(lambda x: 16 if x < 16 else x)
    traffic['Driver Age'] = traffic['Driver Age'].round().astype('Int64')
    return traffic


def make_values_for_sex(traffic):
    dict = {
        'M' : 0,
        'F' : 1
    }
    traffic['MorF'] = traffic['Driver Sex'].map(lambda x: dict.get(str(x), 0))
    traffic_driver = traffic['MorF']
    traffic_driver = traffic_driver.dropna()
    p_1 = traffic_driver.mean()
    p_0 = 1 - p_1
    
    
    def generate_binary(p_zero, p_one):
        return random.choices([0, 1], weights=[p_zero, p_one])[0]
    
    
    traffic['MorF'] = traffic['MorF'].apply(lambda x: generate_binary(p_0, p_1) if pd.isna(x) else x).astype('Int64')
    return traffic      


def make_values_for_conditions(traffic):
    weather_dict = {
        'Clear' : 0,
        'Dust' : 0,
        'NaN' : 0,
        'Cloudy' : 1,
        'Wind' : 2,
        'Rain' : 3,
        'Fog' : 4,
        'Sleet or Hail' : 5,
        'Snow' : 6,
        'Blowing Snow' : 7,
        'Freezing Rain or Freezing Drizzle' : 8
        }
    traffic['Time'] = traffic['Time'].str.extract(r'HOUR(\d+\.?\d*)').astype(float).astype('Int64')
    traffic['Condition_Code'] = traffic['Condition'].map(lambda x: weather_dict.get(str(x), 0))
    return traffic


# new_df = pd.DataFrame()
# for x in range(2021, 2025):
#     traffic = pd.read_csv(f"CRASH_DATA\CRASH_{x}.csv")
#     traffic.reset_index(drop=True, inplace=True)
#     new_df = pd.concat([new_df, traffic])
# new_df.to_csv('CRASH_2021_2024.csv', index=False)


def look_for_simularities_2014():
    traffic = pd.DataFrame()
    for x in range(2014, 2020):
        new_df = pd.read_csv(f"CRASH_DATA/CRASH_{x}.csv")
        traffic = pd.concat([traffic, new_df])
        
    traffic = fix_driver_age(traffic)
    traffic = make_values_for_conditions_2014(traffic)
    traffic = make_values_for_sex(traffic)
    traffic = make_better_hours(traffic)
    
    traffic = traffic.drop(columns={'CONDITION','Latitude','Longitude','Driver Sex'})
    
    traffic.to_csv('CRASH_DATA_2014_2019.csv', index=False)


def make_better_hours(traffic):
    def to_hour(x):
        return int(x // 100)
    traffic['Time'] = traffic['Time'].apply(lambda x: to_hour(x))
    return traffic
    

def make_values_for_conditions_2014(traffic):
    weather_dict = {
        'DRY' : 0,
        'FOREIGN MATERIAL' : 0,
        'NaN' : 0,
        'UNKNOWN' : 0,
        'Cloudy' : 1,
        'Wind' : 2,
        'WET' : 3,
        'ICY' : 3,
        'DRY W/VIS ICY ROAD TREATMENT' : 4,
        'SLUSHY W/VIS ICY ROAD TREATMENT' : 5,
        'SLUSHY' : 5,
        'SNOWY' : 6,
        'SNOWY W/VIS ICY ROAD TREATMENT' : 7,
        'WET W/VIS ICY ROAD TREATMENT' : 8
        }
    # traffic_conditions = traffic['CONDITION']
    # traffic_conditions = traffic_conditions.drop_duplicates(keep='first')
    # print(traffic_conditions)
    # traffic['Time'] = traffic['Time'].str.extract(r'HOUR(\d+\.?\d*)').astype(float).astype('Int64')
    traffic['Condition_Code'] = traffic['CONDITION'].map(lambda x: weather_dict.get(str(x), 0))
    return traffic

# df_2014 = pd.read_csv('CRASH_DATA_2014_2019.csv')
# df_2021 = pd.read_csv('CRASH_DATA_2021_2024.csv')
# df = pd.concat([df_2014, df_2021])
# df.to_csv('TOTAL_CRASH_DATA.csv', index=False)

# look_for_simularities()
# combine_traffic_snow()
# WRANGLE_FAT_DATA()

# traffic = pd.read_csv("TOTAL_CRASH_DATA.csv")
# weather = pd.read_csv("TRAFFIC_SNOW_DATA.csv")

# traffic["Date"] = pd.to_datetime(traffic["Date"])
# weather["Date"] = pd.to_datetime(weather["Date"])

# # traffic = traffic.drop_duplicates(subset="Date", keep="first")
# merged = pd.merge(weather, traffic, on=["Date", "Time"], how="left")
# merged.to_csv("DO_THIS_WORKK.csv", index=False)