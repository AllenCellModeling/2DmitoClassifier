import datasetdatabase as dsdb
import json
import pandas as pd

dbConnectionInfo = json.load(open('/allen/aics/modeling/jamies/projects/dbconnect/configs.json', 'r'))

mngr = dsdb.ConnectionManager(user="jamies")
mngr.add_connections(dbConnectionInfo)
prod = mngr.connect('prod')

model_files = [
    {'pytorch_model': '/allen/aics/modeling/jamies/projects/three_channel/XYZ3/X/saved_model_10E_01.pt',
     'axis': 'X', 'param_order': 0},
    {'pytorch_model': '/allen/aics/modeling/jamies/projects/three_channel/XYZ3/Y/saved_model_10E_03.pt',
        'axis': 'Y', 'param_order': 1},
    {'pytorch_model': '/allen/aics/modeling/jamies/projects/three_channel/XYZ3/Z/saved_model_10E_09.pt',
     'axis': 'Z', 'param_order': 2}
]

df = pd.DataFrame(model_files)

ds_info = prod.upload_dataset(dataset=df,
                              name="MitoClassifierModels20180810",
                              description="These are the X, Y, Z models for Mito-classification",
                              type_map={'pytorch_model': str,
                                        'axis': str,
                                        'param_order': int},
                              filepath_columns=['pytorch_model'],
                              store_files=False,
                              import_as_type_map=True)

print(ds_info)