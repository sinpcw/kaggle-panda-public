import pandas as pd

obj = pd.read_csv('ota_o2u_22017707_k01.csv')
ids = [ ]
for i in range(len(obj)):
    if obj.iat[i, 2] == 'radboud':
        ids.append(obj.iat[i, 1])

csv = pd.read_csv('nfold_train.csv')
csv['weight'] = 1.0

csv.loc[csv['image_id'].isin(ids), 'weight'] = 0.02

csv.to_csv('nfold_train_ex1.csv', index=None)