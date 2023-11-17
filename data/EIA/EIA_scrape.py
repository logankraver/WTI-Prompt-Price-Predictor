import json
import pandas as pd

fields = [
    "U.S. Exports of Crude Oil, Weekly", 
    "U.S. Field Production of Crude Oil, Weekly", 
    "U.S. Net Imports of Crude Oil, Weekly", 
    "U.S. Refiner Net Input of Crude Oil, Weekly", 
    "U.S. Ending Stocks of Crude Oil, Weekly", 
    "U.S. Imports of Crude Oil, Weekly", 
]

file_names = [
    "US_exports",
    "US_field_prod",
    "US_net_imports",
    "US_refiner_input",
    "US_stocks",
    "US_imports"
]

data = []
with open("PET.txt", 'r') as f:
    out = f.readlines() 
for line in out:
    try:
        line_out = json.loads(line)
        if line_out['name'] in fields:
            line_out['file'] = file_names[fields.index(line_out['name'])]
            data.append(line_out)
    except:
        print(line)

def gt_date(date1, date2):
    year1 = date1[:4]
    year2 = date2[:4]
    if year1 != year2:
        return year1 < year2

    month1 = date1[4:6]
    month2 = date2[4:6]

    if month1 != month2:
        return month1 < month2

    day1 = date1[6:]
    day2 = date2[6:]

    return day1 < day2

def reformat_date(date):
    year = date[:4]
    month = date[4:6]
    day = date[6:8]

    return f"{month}/{day}/{year}"

start_date = '20200101'
for row in data:
    name = row['name']
    units = row['unitsshort']
    filtered_data = []
    for date, val in row['data']:
        if gt_date(start_date, date):
            filtered_data.append({'date': reformat_date(date), f'val ({units})': val})

    pd.DataFrame(filtered_data).to_csv(f"{row['file']}.csv")
