import datasetdatabase as dsdb
import json
import pandas as pd
import os
import pathlib as path
import hashlib

prod = dsdb.DatasetDatabase(config='/allen/aics/modeling/jamies/projects/dbconnect/configs2.json', user='jamies')

print('printing prod:\n', prod)

ds = prod.get_items_from_table('Dataset')
for d in ds:
    print('printing dataset: ', d)
#mngr.add_connections(dsdb.LOCAL)   #dbConnectionInfo)
#prod = mngr.connect(dsdb.LOCAL)

#csv_file = '/allen/aics/modeling/PIPELINE/2018-07-23-17:20:57/data_jobs_out.csv'
csv_file = '/allen/aics/modeling/jamies/projects/Data/mitoHidden/data_jobs_out.csv'
data = pd.read_csv(csv_file)
print('A: ', data.shape)
data.drop(['Version', 'colony_position', 'cellID', 'imageID', 'Index', 'index'], axis=1, inplace=True)
data.fillna("*****", inplace=True)
print('data.shape = ', data.shape)
#data.dropna(axis=0, inplace=True)
img_path = path.Path('/allen/aics/modeling/jamies/projects/Data/mitoHidden/')
print(data['inputFilename'][1])
print('os:', data['outputNucSegWholeFilename'][2])

def js_join(row, ipath):
    row['inputFilename'] = str(path.Path(row['inputFolder']) / path.Path(row['inputFilename']))
    row['outputCellSegWholeFilename'] = str(path.Path(row['outputSegmentationPath']) /path.Path(row['outputCellSegWholeFilename']))
    row['outputNucSegWholeFilename'] = str(path.Path(row['outputSegmentationPath']) / path.Path(row['outputNucSegWholeFilename']))
    row['save_flat_proj_reg_path'] = str(ipath / path.Path(row['save_flat_proj_reg_path']))
    row['save_flat_reg_path'] = str(ipath / path.Path(row['save_flat_reg_path']))
    return(row)

data = data.apply(lambda x: js_join(x, img_path), axis=1)
print(data['inputFilename'][1])
# data['outputCellSegWholeSegScaleFilename'] = data['outputSegmentationPath'] + data['outputCellSegWholeSegScaleFilename']
# data['outputNucSegWholeSegScaleFilename'] = data['outputSegmentationPath'] + data['outputNucSegWholeSegScaleFilename']

print(data.shape)
data.to_csv('~/output.csv')

def nuc_rows(key):
    rows_to_remove = []
    for i, v in enumerate(data[key]):
        if not os.path.exists(v):
            rows_to_remove.append(i)

    if len(rows_to_remove) > 0:
        print('rows_to_remove:', len(rows_to_remove))
        print(rows_to_remove[-1], rows_to_remove[-2], rows_to_remove[-3])
        data.drop(data.index[rows_to_remove], inplace=True)

data.drop(['inputFolder', 'outputSegmentationPath'], axis=1, inplace=True)
print(data['outputNucSegWholeFilename'][2])
nuc_rows('inputFilename')
nuc_rows('outputCellSegWholeFilename')
nuc_rows('outputNucSegWholeFilename')
nuc_rows('save_flat_proj_reg_path')
print('size: ', data.shape)

data.to_csv('~/joutput.csv')

data = dsdb.Dataset(data, name="MitoData: Hidden 20180917",
                    description="Mitosis Hidden Validation Data for Mito-classification",  # not a file},
                    )
data.validate(type_validation_map={  # 'DeliveryDate': str,
                        'FinalScore': int,
                        # 'Index': int,
                        # 'SegmentationXyPixelSize': float,
                        # 'SegmentationZPixelSize': float,
                        # 'VersionNucMemb': str,
                        # 'VersionStructure': str,
                        # 'cell_line_ID': int,
                        # 'colony_position': str,
                        'imageXyPixelSize': float,
                        'imageZPixelSize': float,
                        'inputFilename': str,  # file
                        'lightChannel': int,
                        'memChannel': int,
                        'nucChannel': int,
                        'outputCellSegWholeFilename': str,  # file
                        # 'outputCellSegWholeSegScaleFilename': str,  # file
                        'outputNucSegWholeFilename': str,  # file
                        # 'outputNucSegWholeSegScaleFilename': str,  # file
                        'outputThisCellIndex': int,
                        'plate_ID': int,
                        'position_ID': str,
                        # 'reference_file': str,  # what is this? column 29 in csv
                        'source_data': str,
                        'structureChannel': int,
                        'structureProteinName': str,
                        'timePoint': int,
                        'well_ID': str,
                        # 'imageID': str,
                        'save_dir': str,
                        # 'cellID': str,
                        'uuid': str,
                        'uuid_short': str,
                        'save_feats_path': str,  # file
                        'save_regparam_path': str,  # file
                        'save_imsize_out_path': str,  # file
                        'save_cell_reg_path': str,  # file
                        'save_nuc_reg_path': str,  # file
                        'save_dna_reg_path': str,  # file
                        'save_memb_reg_path': str,  # file
                        'save_struct_reg_path': str,  # file
                        'save_trans_reg_path': str,  # file
                        'save_flat_reg_path': str,  # file
                        'save_flat_proj_reg_path': str,  # file
                        'save_h5_reg_path': str
                    },
                    filepath_columns=['save_flat_proj_reg_path'],
                    import_as_type_map=True
            )
#data._md5 = dsdb.utils.tools.get_object_hash(data.ds)
#data._sha256 = dsdb.utils.tools.get_object_hash(data.ds, hashlib.sha256)
data.upload_to(prod)
