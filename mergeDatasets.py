import datasetdatabase as dsdb
import pandas as pd
import pathlib as Path
import numpy as np
import os
import re

# Labeled Data
mitoLabeled = dsdb.read_dataset("/allen/aics/modeling/jamies/projects/Data/irenaLabel.dataset")
AssayDset = dsdb.read_dataset("/allen/aics/modeling/jamies/projects/Data/mitoTrainingData/20181030trainingData.dataset")
gDset = dsdb.read_dataset("/allen/aics/modeling/jamies/projects/Data/ipp_18_10_25.dataset")

print(mitoLabeled)
print(AssayDset)
print(gDset)


def nuc_rows(data, key):
    rows_to_remove = []
    for i, v in enumerate(data[key]):
        print(i, v)
        if not os.path.exists(v):
            print("file not found: ", v)
            rows_to_remove.append(i)

    if len(rows_to_remove) > 0:
        print('rows_to_remove:', len(rows_to_remove))
        print(rows_to_remove[-1], rows_to_remove[-2], rows_to_remove[-3])
        data.drop(data.index[rows_to_remove], inplace=True)
    return data


def map_float_to_int(row):
    x = row['Irina_new_Mitosis_label_1']
    if np.isnan(x) is False:
        row['MitosisLabel'] = round(x)
    return row


adData = AssayDset.ds.copy()
adData['CellKey'] = adData['outputThisCellIndex']
#adData = nuc_rows(adData, 'save_flat_proj_reg_path')
#adData.rename(columns={'save_flat_proj_reg_path': 'old_flat_proj_reg_path'}, inplace=True)
adData['FileKey'] = adData['InputFilename']
adData.drop(['MitosisLabel'], axis=1, inplace=True)

mData = mitoLabeled.ds.copy()
mData = mData.apply(lambda row: map_float_to_int(row), axis=1)
mData.rename(columns={'inputFilename': 'FileKey', 'outputThisCellIndex': 'CellKey'}, inplace=True)
mData['MitosisLabel_Keep'] = mData['MitosisLabel']
for k in mData.keys():
    print('key: ', k)
mData.drop(['save_flat_proj_reg_path', 'Irina_new_Mitosis_Meaning_1',
            'Irina_new_Mitosis_label_1', 'MitosisLabel'],
           axis=1, inplace=True
           )


for k in mData.keys():
    print('nkey: ', k)


def d1_apply(row):
    ppath = Path.Path(row['FileKey'])
    row['FileKey'] = str(ppath.name)
    return row


def get_filekey(row):
    global first
    ppath = Path.Path(row['FileKey'])
    if first:
        print('d0path: ', ppath)
        first = False
    row['FileKey'] = str(ppath.name)
    return row


first = True
mData = mData.apply(lambda x: get_filekey(x), axis=1)
first = True
adData = adData.apply(lambda x: get_filekey(x), axis=1)

print('mitoLabeledData.shape => ', mData.shape)
print('assayDevSheets.shape => ', adData.shape)
print('gData.shape => ', gDset.ds.shape)


def get_grgfilename(row):
    global first
    ppath = Path.PurePosixPath(Path.PurePosixPath(row['FileKey']).stem).stem
    if first:
        print('grg: ', ppath, '.czi')
        first = False
    row['FileKey'] = str(ppath + ".czi")
    return row


gData = gDset.ds  # d2set.ds
gData['FileKey'] = gData['SourceFilename'].copy()
gData['CellKey'] = gData['CellIndex']

# d2ata.rename(columns={'InputFilename': 'FileKey', 'outputThisCellIndex': 'CellKey'},
# inplace=True)

first = True
gData = gData.apply(lambda row: get_grgfilename(row), axis=1)

#gData = nuc_rows(gData, 'save_flat_proj_reg_path')
#for k in gData.keys():
#    print('key: ', k)

#ans = pd.merge(d0ata, d2ata, on=['FileKey'], how='inner')

ans2 = pd.merge(mData, adData, on=['FileKey', 'CellKey'], how='inner')
print("0=?=1 : ", mData.shape, " =?= ", ans2.shape)
ans2.to_csv('~/a1.csv')
ans = pd.merge(mData, gData, on=['FileKey', 'CellKey'], how='inner')
ans.to_csv('~/a2.csv')
print(mData.shape, ' =?= ', ans.shape)

print("ans:")
ans = nuc_rows(ans, 'save_flat_proj_reg_path')
print("ans2:")
for k in ans2.keys():
    print(k)
ans2 = nuc_rows(ans2, 'save_flat_proj_reg_path')

ans = pd.concat([ans, ans2], ignore_index=True, sort=False)

print('ans.shape = ', ans.shape, ' out of: ', mData.shape)

ans = ans[['FileKey', 'CellKey', 'MitosisLabel_Keep', 'SourceReadPath', 'save_flat_proj_reg_path']]

ans.rename(columns={'MitosisLabel_Keep': 'MitosisLabel'}, inplace=True)




ans = nuc_rows(ans, 'save_flat_proj_reg_path')
ans.to_csv("~/ans1.csv")

prod = dsdb.DatasetDatabase(config='/allen/aics/modeling/jamies/projects/dbconnect/configs2.json', user='jamies',
                            processing_limit=30)

dataup = dsdb.Dataset(ans, name="New IPP / New Irina Mitosis Labels / Training data",
                      description="This is a meshing of the Irina's new MitosisLabels (datasetID=126) with the new"
                                  " outputs of the IPP consisting of the reprossessed AssayDevData (datasetID=110) and "
                                  " greg's Handoff data (datasetID=117)"
                      )

typemap = {
    'save_flat_proj_reg_path': str,
    'MitosisLabel': int,
    'SourceReadPath': str
}

dataup.validate(type_validation_map=typemap,
                filepath_columns=['save_flat_proj_reg_path'],
                import_as_type_map=True
                )

dataup.upload_to(prod)

dataup.save("/allen/aics/modeling/jamies/projects/Data/MitoTraining20181113")

exit()
