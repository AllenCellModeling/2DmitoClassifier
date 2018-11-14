import datasetdatabase as dsdb
import json
import pandas as pd
import os
import pathlib as path
import hashlib

prod = dsdb.DatasetDatabase(config='/allen/aics/modeling/jamies/projects/dbconnect/configs2.json', user='jamies',
                            processing_limit=30)

dsets = prod.get_items_from_table('Dataset')
for d in dsets:
	print(d['DatasetId'], d['Name'])

exit()
dset = prod.get_dataset(id=76)

print(data)

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
#data._md5 = dsdb.utils.tools.get_object_hash(data.ds)
#data._sha256 = dsdb.utils.tools.get_object_hash(data.ds, hashlib.sha256)

