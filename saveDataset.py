import datasetdatabase as dsdb

prod = dsdb.DatasetDatabase(config='/allen/aics/modeling/jamies/projects/dbconnect/configs2.json', user='jamies', processing_limit=40)
dset = prod.get_items_from_table('Dataset')
for d in dset:
    print(d['DatasetId'], d['Name'])


#hset = prod.get_dataset(id=90)
#hset.save('/root/projects/three_channel/MitoTrainingSet')

#dset = dsdb.read_dataset('/allen/aics/modeling/jamies/projects/three_channel/gregSet.dataset')

#print(dset.info.md5, ' =?= ', dset.md5)
