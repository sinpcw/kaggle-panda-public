import os
import pandas as pd

output_dir = './'

def split_dataprovider(df):
    kdf = df[df['data_provider'] == 'karolinska']
    rdf = df[df['data_provider'] == 'radboud']
    return kdf, rdf

df = pd.read_csv('train.csv')
dp = pd.read_csv('duplicate_0940.csv')

dplist = dp.iloc[:, 0].values.tolist()
print(len(dplist))

tdf = df[~df['image_id'].isin(dplist)]

dupl = df[df['image_id'].isin(dplist)]
dupl.to_csv(os.path.join(output_dir, 'dupl.csv'), index=None)

print('non-duplicate {}'.format(len(tdf)))

kdf, rdf = split_dataprovider(df)

# tdf['sort_key'] = 0
# for i in range(len(tdf)):
#     # image_id, data_provider, isup_grade, gleason_score
#     if tdf.iat[i, 3] != 'negative':
#         tdf.iat[i, 4] = int(tdf.iat[i, 3][0]) * 10 + int(tdf.iat[i, 3][2])
kdf, rdf = split_dataprovider(tdf)
kdfs = kdf.sort_values('isup_grade')
rdfs = rdf.sort_values('isup_grade')
# kdfs = kdf.sort_values('sort_key')
# rdfs = rdf.sort_values('sort_key')
# del kdfs['sort_key']
# del rdfs['sort_key']

nfold = 9
for n in range(nfold):
    x = kdfs.iloc[n::nfold, :]
    x.to_csv(os.path.join(output_dir, 'train_kdf_k{}.csv'.format(n)), index=None)
    y = rdfs.iloc[n::nfold, :]
    y.to_csv(os.path.join(output_dir, 'train_rdf_k{}.csv'.format(n)), index=None)
    # z = pd.concat([x, y])
    # z.to_csv(os.path.join(output_dir, 'train_sdf_k{}.csv'.format(n)), index=None)

valid_fold = [ 7, 8 ]

tkdf = None
vkdf = None
trdf = None
vrdf = None
akdf = None
ardf = None
for i in range(nfold):
    kdf = pd.read_csv('train_kdf_k{}.csv'.format(i))
    rdf = pd.read_csv('train_rdf_k{}.csv'.format(i))
    if i in valid_fold:
        vkdf = pd.concat([vkdf, kdf]) if vkdf is not None else kdf
        vrdf = pd.concat([vrdf, rdf]) if vrdf is not None else rdf
    else:
        tkdf = pd.concat([tkdf, kdf]) if tkdf is not None else kdf
        trdf = pd.concat([trdf, rdf]) if trdf is not None else rdf
    akdf = pd.concat([akdf, kdf]) if akdf is not None else kdf
    ardf = pd.concat([ardf, rdf]) if ardf is not None else rdf
tmdf = pd.concat([tkdf, trdf])
vmdf = pd.concat([vkdf, vrdf])
# tkdf.to_csv('nfold_train_karolinska.csv', index=None)
# vkdf.to_csv('nfold_valid_karolinska.csv', index=None)
# trdf.to_csv('nfold_train_radboud.csv', index=None)
# vrdf.to_csv('nfold_valid_radboud.csv', index=None)

# duplicated concat:
tmdf = pd.concat([tmdf, dupl])

# base training
tmdf.to_csv('nfold_train_b.csv', index=None)
vmdf.to_csv('nfold_valid_b.csv', index=None)

# karolinska all and radboud validation
akdf.to_csv('nfold_train_k.csv', index=None)
vrdf.to_csv('nfold_valid_k.csv', index=None)

# radboud all and karolinska validation
vkdf.to_csv('nfold_train_r.csv', index=None)
ardf.to_csv('nfold_valid_r.csv', index=None)
