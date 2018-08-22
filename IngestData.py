import datasetdatabase as dsdb
import json
import pandas as pd
import os

dbConnectionInfo = json.load(open('/allen/aics/modeling/jamies/projects/dbconnect/configs.json', 'r'))

mngr = dsdb.ConnectionManager(user="jamies")
mngr.add_connections(dbConnectionInfo)
prod = mngr.connect('prod')
# prod._deep_print()
# prod.schema_version.drop_schema(prod)
# prod.schema.drop_if_exists('File')
# prod = mngr.connect('prod')
# prod._deep_print()
# exit()

# csv_file = '/allen/aics/modeling/PIPELINE/2018-07-23-17:20:57/data_jobs_out.csv'
csv_file = '/allen/aics/modeling/jamies/Data/assay_dev_output/data_jobs_out.csv'
data = pd.read_csv(csv_file)
print(data.shape)
data.drop(['Version', 'colony_position'], axis=1, inplace=True)
#data.fillna(0, inplace=True)
print('data.shape = ', data.shape)
data.to_csv('~/output.csv')
data.dropna(axis=0, inplace=True)
data['inputFilename'] = data['inputFolder'] + data['inputFilename']
data['outputCellSegWholeFilename'] = data['outputSegmentationPath'] + data['outputCellSegWholeFilename']
# data['outputCellSegWholeSegScaleFilename'] = data['outputSegmentationPath'] + data['outputCellSegWholeSegScaleFilename']
data['outputNucSegWholeFilename'] = data['outputSegmentationPath'] + data['outputNucSegWholeFilename']
# data['outputNucSegWholeSegScaleFilename'] = data['outputSegmentationPath'] + data['outputNucSegWholeSegScaleFilename']
data['save_flat_proj_reg_path'] = '/allen/aics/modeling/jamies/Data/assay_dev_output/' + data['save_flat_proj_reg_path']
data['save_flat_reg_path'] = '/allen/aics/modeling/jamies/Data/assay_dev_output/' + data['save_flat_reg_path']
data['save_h5_reg_path'] = '/allen/aics/modeling/jamies/Data/assay_dev_output/' + data['save_h5_reg_path']

print(data.shape)

rows_to_remove = []
for i, v in enumerate(data['save_flat_proj_reg_path']):
    if not os.path.exists(v):
        rows_to_remove.append(i)

data2 = data.head(10)
if len(rows_to_remove) > 0:
    print('rows_to_remove:', len(rows_to_remove))
    print(rows_to_remove[-1], rows_to_remove[-2], rows_to_remove[-3])
    data.drop(data.index[rows_to_remove], inplace=True)

print('size: ', data.shape)

ds_info = prod.upload_dataset(dataset=data,
                              name="MitoEval20180821",
                              description="Mitosis Test Data for Mito-classification",
                              type_map={#'DeliveryDate': str,
                                        'FinalScore': int,
                                        'Index': int,
                                        #'SegmentationXyPixelSize': float,
                                        #'SegmentationZPixelSize': float,
                                        #'VersionNucMemb': str,
                                        #'VersionStructure': str,
                                        #'cell_line_ID': int,
                                        #'colony_position': str,
                                        'imageXyPixelSize': float,
                                        'imageZPixelSize': float,
                                        'inputFilename': str,  # file
                                        'lightChannel': int,
                                        'memChannel': int,
                                        'nucChannel': int,
                                        'outputCellSegWholeFilename': str,  # file
                                        #'outputCellSegWholeSegScaleFilename': str,  # file
                                        'outputNucSegWholeFilename': str,  # file
                                        #'outputNucSegWholeSegScaleFilename': str,  # file
                                        'outputThisCellIndex': int,
                                        'plate_ID': int,
                                        'position_ID': str,
                                        #'reference_file': str,  # what is this? column 29 in csv
                                        'source_data': str,
                                        'structureChannel': int,
                                        'structureProteinName': str,
                                        'timePoint': int,
                                        'well_ID': str,
                                        'imageID': str,
                                        'save_dir': str,
                                        'cellID': str,
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
                                        'save_h5_reg_path': str},  # not a file},
                              filepath_columns=['save_flat_proj_reg_path'],
                              store_files=False,
                              import_as_type_map=True)

print(ds_info)
