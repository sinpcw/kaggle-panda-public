import pandas as pd

valid_fold = 9

tkdf = None
vkdf = None
trdf = None
vrdf = None
for i in range(1, 10):
    kdf = pd.read_csv('train_kdf_k{}.csv'.format(i))
    rdf = pd.read_csv('train_rdf_k{}.csv'.format(i))
    if i == valid_fold:
        vkdf = pd.concat([vkdf, kdf]) if vkdf is not None else kdf
        vrdf = pd.concat([vrdf, rdf]) if vrdf is not None else rdf
    else:
        tkdf = pd.concat([tkdf, kdf]) if tkdf is not None else kdf
        trdf = pd.concat([trdf, rdf]) if trdf is not None else rdf
tmdf = pd.concat([tkdf, trdf])
vmdf = pd.concat([vkdf, vrdf])

tkdf.to_csv('nfold_train_karolinska.csv', index=None)
vkdf.to_csv('nfold_valid_karolinska.csv', index=None)
trdf.to_csv('nfold_train_radboud.csv', index=None)
vrdf.to_csv('nfold_valid_radboud.csv', index=None)
tmdf.to_csv('nfold_train.csv', index=None)
vmdf.to_csv('nfold_valid.csv', index=None)
