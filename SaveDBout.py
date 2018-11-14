import datasetdatabase as dsdb
import json
import pandas as pd
import os
import pathlib as path
import hashlib

os.environ['DSDB_PROCESSING_LIMIT'] = '20'

key = 'QCB_SEC61_feature_old'
jpath = path.Path('/Users/jamies/Data/Jackson/', key)
prod = dsdb.DatasetDatabase(config='/allen/aics/modeling/jamies/projects/dbconnect/configs2.json')
prod.get_dataset(name=key).save(jpath)
prod.recent
