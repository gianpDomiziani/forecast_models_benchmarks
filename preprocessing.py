import pandas as pd
import numpy as np

# load data
print('Load Main Data')
raw_dir = "D:\\Downloads\\"
calendar_df = pd.read_csv(raw_dir + "calendar.csv")
train_df = pd.read_csv(raw_dir + "sales_train_evaluation.csv")
prices_df = pd.read_csv(raw_dir + "sell_prices.csv")

# print(calendar_df.head())
# print(train_df.head())
# print(prices_df.head())

# variables
TARGET = 'sales'
TIME_HORIZON = 28
END_TRAIN = 1941-28  # total num of days is 1941, leave the last 28 out for testing purposes
DAY_COLUMN = 'd'

# create grid
print('Create Grid')
# reformat train_df
# instead of days in a horizontal orientation, make it vertical

index_columns = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
grid_df = pd.melt(train_df,
                  id_vars=index_columns,
                  var_name=DAY_COLUMN,
                  value_name='sales')

# print(grid_df.head(10))

# generate holdout validation df
print('Creating holdout validation dataframe')
last_training_day = END_TRAIN
end_val_day = END_TRAIN + TIME_HORIZON

grid_df['d'] = grid_df['d'].apply(lambda x: x[2:]).astype(np.int16)

holdout_df = grid_df[(grid_df['d'] > last_training_day) & (grid_df['d'] <= end_val_day)][index_columns + [DAY_COLUMN, TARGET]]
grid_df = grid_df[grid_df['d'] <= last_training_day]

holdout_df.reset_index(drop=True)
holdout_df['d'] = holdout_df['d'].apply(lambda x: 'd_' + str(x))
grid_df['d'] = grid_df['d'].apply(lambda x: 'd_' + str(x))

# print(grid_df.tail())
# print(holdout_df.head())

# add rows for test
print('Adding test days')
add_grid = pd.DataFrame()
for i in range(1, TIME_HORIZON + 1):
    temp_df = train_df[index_columns]
    temp_df = temp_df.drop_duplicates()
    temp_df['d'] = 'd_' + str(END_TRAIN + i)
    temp_df[TARGET] = np.nan
    add_grid = pd.concat([add_grid, temp_df])

grid_df = pd.concat([grid_df, add_grid])
grid_df = grid_df.reset_index(drop=True)

# print(grid_df.tail())
del temp_df, add_grid, train_df


def merge_by_concat(df1, df2, merge_on):
    merged_gf = df1[merge_on]
    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1


# Release dates
print('Release')
release_df = prices_df.groupby(['store_id', 'item_id'])['wm_yr_wk'].agg(['min']).reset_index()
release_df.columns = ['store_id', 'item_id', 'release']
# print(release_df.head())

grid_df = merge_by_concat(grid_df, release_df, ['store_id', 'item_id'])  # match release date w each product (store_id, item_id)
# print(grid_df.head(20))
idx = calendar_df.index.values.tolist()
d = ['d_' + str(x + 1) for x in idx]
calendar_df['d'] = d
# print(calendar_df.head())
grid_df = merge_by_concat(grid_df, calendar_df[['wm_yr_wk', 'd']], ['d'])  # match day number w wm_yr_wk
# print(grid_df.head(20))
grid_df = grid_df[grid_df['wm_yr_wk'] >= grid_df['release']].reset_index(drop=True)  # for each product only keep if day is after release date -> delete useless rows
# print(grid_df.head(20))

# prices feature normalization (min max)
print('Prices')
# prices_df['price_norm'] = prices_df['sell_price'] / prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('max')
grid_df = grid_df.merge(prices_df, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

# add dates and events
print('Calendar')
# calendar_cols = ['date', 'd', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI']
calendar_cols = ['date', 'd']
grid_df = grid_df.merge(calendar_df[calendar_cols], on="d", how="left")
holdout_df = holdout_df.merge(calendar_df[calendar_cols], on="d", how="left")

holdout_df.to_csv('holdout.csv')
grid_df.to_csv("preprocessed.csv")
