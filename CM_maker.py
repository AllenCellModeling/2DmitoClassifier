import os, sys
sys.path.append("/allen/aics/modeling/jamies/projects/three_channel/src")

import json
import datasetdatabase as dsdb
from model_analysis import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#dbConnectionInfo = json.load(open('/allen/aics/modeling/jamies/projects/dbconnect/configs2.json', 'r'))

match_count = 0
mismatch = 0
pcounts = dict([(-1, 0), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)])
counts = dict([(-1, 0), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)])

for i in range(len(mlist.ds['MitosisLabel'])):
    if mlist.ds['MitosisLabel'][i] == mlist.ds['MitosisLabelPredicted'][i]:
        match_count += 1
    else:
        mismatch += 1
    counts[mlist.ds['MitosisLabel'][i]] += 1
    pcounts[mlist.ds['MitosisLabelPredicted'][i]] += 1

for i in range(-1, 8):
    print(i, counts[i], pcounts[i])

print('Matches: ', match_count, '\nMisMatches: ', mismatch, '\nPercent: ', mismatch/(match_count+mismatch))
mlist.ds.drop(mlist.ds.index[mlist.ds['MitosisLabel'] <= -1].tolist(), inplace=True)
mlist.ds.drop(mlist.ds.index[mlist.ds['MitosisLabel'] >= 8].tolist(), inplace=True)

fig, ax = plot_confusion_matrix(mlist.ds['MitosisLabel'], mlist.ds['MitosisLabelPredicted'], classes=np.array(["not mitotic",
                                            "M1: prophase 1",
                                            "M2: prophase 2",
                                            "M3: pro metaphase 1",
                                            "M4: pro metaphase 2",
                                            "M5: metaphase",
                                            "M6: anaphase",
                                            "M7: telophase-cytokinesis"]))
fig.savefig('/allen/aics/modeling/jamies/projects/Data/mitoHidden/CM.png', bbox_extra_artists=(ax,), bbox_inches='tight')
plt.close(fig)
mlist.ds.to_csv('/allen/aics/modeling/jamies/projects/Data/mitoHidden/MitosisOutput.csv')
# print(mlist)
