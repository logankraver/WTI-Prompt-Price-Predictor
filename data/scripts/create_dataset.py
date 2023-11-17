import pandas as pd
from tqdm import tqdm

months = {
    'Jan': '01',
    'Feb': '02',
    'Mar': '03',
    'Apr': '04',
    'May': '05',
    'Jun': '06',
    'Jul': '07',
    'Aug': '08',
    'Sep': '09',
    'Oct': '10',
    'Nov': '11',
    'Dec': '12',
}
def get_date(s):
    month = months[s[:3]]
    year = f"20{s[-2:]}"
    return month, year


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

# combine all EIA datasets
EIA_path = "../EIA/"

EIA_datasets = [
    "US_exports.csv",
    "US_field_prod.csv",
    "US_imports.csv",
    "US_net_imports.csv",
    "US_refiner_input.csv",
    "US_stocks.csv",
]
dataset = []

df = pd.read_csv(EIA_path + EIA_datasets[0])
for _, row in df.iterrows():
    dataset.append({
        'Date': row[1],
        'Exports': row[2]
    })

for file in EIA_datasets[1:]:
    df = pd.read_csv(EIA_path + file)
    col_name = file[file.find('_') + 1 : file.find('.')]
    for i in range(len(dataset)):
        date = dataset[i]['Date']
        dataset[i][col_name] = int(df[df['date'] == date][df.columns[-1]][i]) 

# macro data format into weekly for monthly data for unemployment
unemployment_path = "../macro/unemployment.csv"
df = pd.read_csv(unemployment_path)
for _, row in df.iterrows():
    date = row[0]
    val = row[1]
    if type(date) == str:
        month, year = get_date(date)
        for i in range(len(dataset)):
            row = dataset[i]
            if month == row['Date'][:2] and year == row['Date'][-4:]:
                dataset[i]['Unemployment'] = val


# convert all daily data into weekly
daily_datasets = [
    "../WTI.csv",
    "../macro/VIX.csv",
    "../macro/SnP500.csv",
]

combined = []
wti = pd.read_csv(daily_datasets[0])
vix = pd.read_csv(daily_datasets[1])
snp = pd.read_csv(daily_datasets[2])

for idx, row in tqdm(snp.iterrows(), desc='Get Combined'):
    date = row[0]
    # reformat date
    month, year = get_date(date[date.find("-")+1:])
    day = date[:date.find("-")]
    snp_date = f"{year}-{month}-{day}"
    snp_price = row[1]
    wti_price = 0
    
    for _, row in wti.iterrows():
        wti_date = row[0]
        wti_mon = wti_date[wti_date.find('-')+1:wti_date.find('-', wti_date.find('-')+1)]
        if len(wti_mon) != 2:
            wti_mon = '0' + wti_mon
        wti_day = wti_date[-2:] if wti_date[-2] != '-' else '0' + wti_date[-1]
        if year == wti_date[:4] and month == wti_mon and day == wti_day: 
            wti_price = row[1]

    vix_price = vix[vix['Date'] == snp_date]['Close']
    if list(vix_price) != []:
        vix_price = float(vix_price)
    
    if wti_price != 0 and type(vix_price) == float:
        combined.append({
            'date': snp_date,
            'snp' : snp_price,
            'wti': wti_price,
            'vix' : vix_price,
        })

# convert from daily to weekly
j = len(combined) - 1
for i in range(len(dataset) - 1, -1, -1):
    weekly_date = dataset[i]['Date']   
    reformat_weekly = dataset[i]['Date'][-4:] + dataset[i]['Date'][:2] + dataset[i]['Date'][3:5]
    snp_total = 0
    wti_total = 0
    vix_total = 0
    count = 0
    reformat_daily = combined[j]['date'][:4] + combined[j]['date'][5:7] + combined[j]['date'][-2:]
    while gt_date(reformat_daily, reformat_weekly) and j >= 0:
        snp_total += float(combined[j]['snp'].replace(",", ""))
        wti_total += float(combined[j]['wti']) #.replace(",", ""))
        vix_total += float(combined[j]['vix']) #.replace(",", ""))
        count += 1
        j -= 1
        reformat_daily = combined[j]['date'][:4] + combined[j]['date'][5:7] + combined[j]['date'][-2:]
    if count != 0:
        dataset[i]['snp'] = snp_total / count
        dataset[i]['wti'] = wti_total / count
        dataset[i]['vix'] = vix_total / count


# create dataset
df = pd.DataFrame(dataset)
df.to_csv("dataset.csv")
