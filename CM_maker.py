import os, sys
sys.path.append("/allen/aics/modeling/jamies/projects/three_channel/src")

import json
import datasetdatabase as dsdb
from model_analysis import plot_confusion_matrix
import matplotlib.pyplot as plt

dbConnectionInfo = json.load(open('/allen/aics/modeling/jamies/projects/dbconnect/configs.json', 'r'))
mngr = dsdb.ConnectionManager(user="jamies")
mngr.add_connections(dbConnectionInfo)
prod = mngr.connect('prod')

# print(prod._deep_print())
mlist = prod.get_dataset(6)

match_count = 0
mismatch = 0
# for i in range(len(mlist['MitosisLabel'])):
#     if mlist['MitosisLabel'][i] == mlist['MitosisLabelPredicted'][i]:
#         match_count += 1
#     else:
#         mismatch += 1

# print('Matches: ', match_count, '\nMisMatches: ', mismatch, '\nPercent: ', mismatch/(match_count+mismatch))
mlist.drop(mlist.index[mlist['MitosisLabel'] == -1].tolist(), inplace=True)

fig, ax = plot_confusion_matrix(mlist['MitosisLabel'], mlist['MitosisLabelPredicted'])
fig.savefig('/allen/aics/modeling/jamies/Data/assay_dev_output/CM.png', bbox_extra_artists=(ax,), bbox_inches='tight')
plt.close(fig)
mlist.to_csv('/allen/aics/modeling/jamies/Data/assay_dev_output/MitosisOutput.csv')
#print(mlist)
