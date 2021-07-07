import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

END_TRAIN = 1913
TIME_HORIZON = 28
DATA_PATH = 'data'

train_df = pd.read_csv(f'{DATA_PATH}/preprocessed.csv')

feature_columns = ['date', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'd', 'sales']
train_df = train_df[feature_columns]

train_df['d'] = train_df['d'].apply(lambda x: x[2:]).astype(np.int16)
# print(train_df.head())
train_df = train_df[train_df['d'] <= END_TRAIN]

# total
total_df = train_df.groupby('date')['sales'].sum().to_frame().reset_index()
# total_df.to_frame(name='sales').reset_index()
print(total_df.tail())
total_df.plot(x='date', y='sales', title='Total sales')
plt.plot(total_df['date'], total_df['sales'])
plt.show()

# state level
state_df = train_df.groupby(['date', 'state_id'])['sales'].sum().to_frame().reset_index()
print(state_df.head(20))
STATE_ID = list(state_df['state_id'].unique())
plt.figure()
states = {}
for state_id in tqdm(STATE_ID):
    states[f'{state_id}'] = state_df[state_df['state_id'] == state_id]
    plt.plot(states.get(f'{state_id}')['date'], states.get(f'{state_id}')['sales'])
plt.legend(states.keys())
plt.show()

# store level
store_df = train_df.groupby(['date', 'store_id'])['sales'].sum().to_frame().reset_index()
print(store_df.head(20))
# print(store_df['sales'].head())
STORE_ID = list(store_df['store_id'].unique())
plt.figure()
stores = {}
for store_id in tqdm(STORE_ID):
    stores[f'{store_id}'] = store_df[store_df['store_id'] == store_id]
    plt.plot(stores.get(f'{store_id}')['date'], stores.get(f'{store_id}')['sales'])
plt.legend(stores.keys())
plt.show()

# store and category level
train_df['store_category_id'] = train_df.apply(lambda x: f"{x['store_id']}_{x['cat_id']}", axis=1)
# store_cat_df2 = df.pivot(index='date', columns='store_category_id', values='sales')
# print(store_df2.head(20))
store_cat_df = train_df.groupby('store_category_id')['sales'].sum().to_frame().reset_index()
print(store_df.head(20))
STORE_CAT_ID = list(store_cat_df['store_category_id'].unique())
plt.figure()
stores_cat = {}
for store_cat_id in tqdm(STORE_CAT_ID):
    stores_cat[f'{store_cat_id}'] = store_cat_df[store_cat_df['store_category_id'] == store_cat_id]
    plt.plot(stores_cat.get(f'{store_cat_id}')['date'], stores_cat.get(f'{store_cat_id}')['sales'])
plt.legend(stores_cat.keys())
plt.show()

