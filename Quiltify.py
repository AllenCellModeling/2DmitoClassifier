import json
import datasetdatabase as dsdb

dbConnectionInfo = json.load(open('/allen/aics/modeling/jamies/projects/dbconnect/configs.json', 'r'))
mngr = dsdb.ConnectionManager(user="jamies")
mngr.add_connections(dbConnectionInfo)
prod = mngr.connect('prod')
#dfio = prod.get_dataset(2)
#for k in dfio.keys():
#    print(k, ": ", dfio[k][1])

quilt_name = prod.export_to_quilt(2, filepath_columns=['pytorch_model'],
                                  quilt_user='jamies')
print(quilt_name)
