import datasetdatabase as dsdb
import pathlib as path
import pandas as pd
import os
import re

prod = dsdb.DatasetDatabase(config='/allen/aics/modeling/jamies/projects/dbconnect/configs2.json', user='jamies',
                            processing_limit=30)

# csv1_file = "/Users/jamies/projects/second_pass/pytorch_mito_feats/second_pass_irina_corrections/csv_files/mitotic_predictions_fcnet_feats_fivemodels_AFTER_Irina_corrections.csv"
# csv1_file = "/Users/jamies/Data/mito_annotations_only_with_pngs.csv"
csv_file = '/allen/aics/modeling/gregj/results/ipp/ipp_18_10_25/data_jobs_out.csv'
data = pd.read_csv(csv_file)

def path_join(row):
    fpath = path.Path(row['inputFolder'], row['inputFilename'])
    row['InputFilename'] = str(fpath)
    return row


print('A1: ', data.shape)
data1 = data[['CellId', 'SourceReadPath', 'save_reg_path_flat_proj']].copy()
data1.rename(index=str, columns={'save_reg_path_flat_proj': 'save_flat_proj_reg_path'}, inplace=True)
data1.dropna(inplace=True)
print('B1: ', data1.shape)
data1.reset_index(drop=True)
print('C1: ', data1.shape)
gflag = True


def nuc_rows(data, key):
    rows_to_remove = []
    for i, v in enumerate(data[key]):
        if not os.path.exists(v):
            print("file not found: ", v)
            rows_to_remove.append(i)

    if len(rows_to_remove) > 0:
        print('rows_to_remove:', len(rows_to_remove))
        print(rows_to_remove[-1], rows_to_remove[-2], rows_to_remove[-3])
        data.drop(data.index[rows_to_remove], inplace=True)
    return data


data1 = nuc_rows(data1, 'save_flat_proj_reg_path')  # save_flat_proj_reg_path
print('size: ', data1.shape)

data1 = dsdb.Dataset(data1, name="Handoff Data for mitoClassifier input",
                     description="IPP images from ipp_18_10_25 containing just the CellId and the path to "
                                 " save_flat_proj_reg_path and the soure readpath "
                     )

typemap = {
    'save_flat_proj_reg_path': str,
    'CellId': int,
    'SourceReadPath': str
}

data1.validate(type_validation_map=typemap,
               filepath_columns=['save_flat_proj_reg_path'],
               import_as_type_map=True
               )

# data = dsdb.Dataset(data, name="Handoff Data from gregj",
#                     description="Ingest of /allen/aics/modeling/gregj/results/ipp/ipp_18_10_12/data_jobs_out.csv "
#                                 "This is the handoff data from gregj for the 20181101 goals.",  # not a file},
#                     )
#
# data.validate(type_validation_map={
#     'save_flat_proj_reg_path': str,
#     },
#     filepath_columns=['save_flat_proj_reg_path'],
#     import_as_type_map=True
# )


data1.upload_to(prod)

data1.save("/allen/aics/modeling/jamies/projects/Data/mitoInput")

print(data1)

# data.validate(type_validation_map={  # 'DeliveryDate': str,
#                         'FinalScore': int,
#                         # 'Index': int,
#                         # 'SegmentationXyPixelSize': float,
#                         # 'SegmentationZPixelSize': float,
#                         # 'VersionNucMemb': str,
#                         # 'VersionStructure': str,
#                         # 'cell_line_ID': int,
#                         # 'colony_position': str,
#                         'imageXyPixelSize': float,
#                         'imageZPixelSize': float,
#                         'inputFilename': str,  # file
#                         'lightChannel': int,
#                         'memChannel': int,
#                         'nucChannel': int,
#                         'outputCellSegWholeFilename': str,  # file
#                         # 'outputCellSegWholeSegScaleFilename': str,  # file
#                         'outputNucSegWholeFilename': str,  # file
#                         # 'outputNucSegWholeSegScaleFilename': str,  # file
#                         'outputThisCellIndex': int,
#                         'plate_ID': int,
#                         'position_ID': str,
#                         # 'reference_file': str,  # what is this? column 29 in csv
#                         'source_data': str,
#                         'structureChannel': int,
#                         'structureProteinName': str,
#                         'timePoint': int,
#                         'well_ID': str,
#                         # 'imageID': str,
#                         'save_dir': str,
#                         # 'cellID': str,
#                         'uuid': str,
#                         'uuid_short': str,
#                         'save_feats_path': str,  # file
#                         'save_regparam_path': str,  # file
#                         'save_imsize_out_path': str,  # file
#                         'save_cell_reg_path': str,  # file
#                         'save_nuc_reg_path': str,  # file
#                         'save_dna_reg_path': str,  # file
#                         'save_memb_reg_path': str,  # file
#                         'save_struct_reg_path': str,  # file
#                         'save_trans_reg_path': str,  # file
#                         'save_flat_reg_path': str,  # file
#                         'save_flat_proj_reg_path': str,  # file
#                         'save_h5_reg_path': str
#                     },
#                     filepath_columns=['save_flat_proj_reg_path'],
#                     import_as_type_map=True
#             )
# data._md5 = dsdb.utils.tools.get_object_hash(data.ds)
# data._sha256 = dsdb.utils.tools.get_object_hash(data.ds, hashlib.sha256)
