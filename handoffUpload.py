import datasetdatabase as dsdb
import featurehandoff as fh
import pandas as pd


def mapToZeroOne(row):
    val = 0
    if row['MitosisLabelPredicted'] > 0:
        val = 1
    row['MitosisLabelPredicted'] = val
    return row


dset = dsdb.read_dataset('/allen/aics/modeling/jamies/projects/Data/uploadData.dataset')

dset.ds.to_csv('~/dset.csv')
data = dset.ds[['CellId', 'MitosisLabelPredicted']]

data = data.apply(lambda x: mapToZeroOne(x), axis=1)

db = fh.FeatureHandoffDatabase("/allen/aics/modeling/jacksonb/prod.json")

algorithm_metadata = {
    "name": "aics-mitosis-classifier",
    "version": "1.0.0",
    "url": "https://github.com/AllenCellModeling/MitoClassifier"
}

column_metadata = {"MitosisLabelPredicted": {"name": "Classified Mitosis Stage",
                                             "units": "unitless"}
                   }

# data = pd.DataFrame('2 column data with CellId, MitosisLabel')

handoff_info = db.upload_feature_set(
    feature_set=data,
    column_metadata=column_metadata,
    algorithm_metadata=algorithm_metadata)

print(handoff_info)
