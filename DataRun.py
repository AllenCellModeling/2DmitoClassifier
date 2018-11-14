import os, sys
sys.path.append("./src")

import matplotlib.pyplot as plt
from MLModel import MitosisClassifierZProj
import numpy as np

mito_run = list()
path = '/root/projects/three_channel/ZProj'

mito_run = MitosisClassifierZProj(path)
mito_run.run_me()
print("itteration_{0} complete.".format(str(m_it).zfill(2)))


p_master_table = {k: {'pred_labels': [], 'pred_entropy': [], 'pred_uid': [], 'probability':[]} for k in mito_run.pred_mito_labels.keys()}

for lname in mito_runs[0].pred_mito_labels.keys():
    p_master_table[lname]['pred_labels'] += mito_run.pred_mito_labels[lname]['pred_labels']
    p_master_table[lname]['pred_entropy'] += mito_run.pred_mito_labels[lname]['pred_entropy']
    p_master_table[lname]['pred_uid'] += mito_run.pred_mito_labels[lname]['pred_uid']
    p_master_table[lname]['probability'] +=  mito_run.pred_mito_labels[lname]['probability']

all_preds_labels = np.array(p_master_table['all']['pred_labels'])
all_preds_entropy = np.array(p_master_table['all']['pred_entropy'])
nz_preds_entropy = all_preds_entropy[all_preds_labels != 0]

n, bins, patches = plt.hist([all_preds_entropy, nz_preds_entropy], 25, density=True, alpha=0.75)
cmhist = os.path.join(path, 'Master_hist.png')
plt.savefig(cmhist)
plt.close()

print("all Done.")
