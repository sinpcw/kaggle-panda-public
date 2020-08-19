import pandas as pd

# no.1
csv_t = pd.read_csv('nfold_train.csv')
csv_v = pd.read_csv('nfold_valid.csv')
csv_del = pd.read_csv('ota_diff_over30.csv', index_col=0)

dellist = [ ]
for i in range(len(csv_del)):
    dellist.append(csv_del.iat[i, 0])

csv_td = csv_t[csv_t['image_id'].isin(dellist)]
csv_td.to_csv('nfold_train_eliminate_1.csv', index=None)

csv_tx = csv_t[~csv_t['image_id'].isin(dellist)]
csv_tx.to_csv('nfold_train_sanitized_1.csv', index=None)

csv_vd = csv_v[csv_v['image_id'].isin(dellist)]
csv_vd.to_csv('nfold_valid_eliminate_1.csv', index=None)

csv_vx = csv_v[~csv_v['image_id'].isin(dellist)]
csv_vx.to_csv('nfold_valid_sanitized_1.csv', index=None)

# no.2
csv_t = pd.read_csv('nfold_train.csv')
csv_v = pd.read_csv('nfold_valid.csv')
csv_del = pd.read_csv('ota_diff_over25.csv', index_col=0)

dellist = [ ]
for i in range(len(csv_del)):
    dellist.append(csv_del.iat[i, 0])

csv_td = csv_t[csv_t['image_id'].isin(dellist)]
csv_td.to_csv('nfold_train_eliminate_2.csv', index=None)

csv_tx = csv_t[~csv_t['image_id'].isin(dellist)]
csv_tx.to_csv('nfold_train_sanitized_2.csv', index=None)

csv_vd = csv_v[csv_v['image_id'].isin(dellist)]
csv_vd.to_csv('nfold_valid_eliminate_2.csv', index=None)

csv_vx = csv_v[~csv_v['image_id'].isin(dellist)]
csv_vx.to_csv('nfold_valid_sanitized_2.csv', index=None)

# no.3
csv_t = pd.read_csv('nfold_train.csv')
csv_v = pd.read_csv('nfold_valid.csv')
csv_del = pd.read_csv('ota_o2u_22017707_k01.csv', index_col=0)

dellist = [ ]
for i in range(len(csv_del)):
    dellist.append(csv_del.iat[i, 0])

csv_td = csv_t[csv_t['image_id'].isin(dellist)]
csv_td.to_csv('nfold_train_eliminate_3.csv', index=None)

csv_tx = csv_t[~csv_t['image_id'].isin(dellist)]
csv_tx.to_csv('nfold_train_sanitized_3.csv', index=None)

csv_vd = csv_v[csv_v['image_id'].isin(dellist)]
csv_vd.to_csv('nfold_valid_eliminate_3.csv', index=None)

csv_vx = csv_v[~csv_v['image_id'].isin(dellist)]
csv_vx.to_csv('nfold_valid_sanitized_3.csv', index=None)
