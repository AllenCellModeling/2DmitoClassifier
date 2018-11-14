import os, sys
sys.path.append("./src")

import matplotlib.pyplot as plt
from ThreeChannel import Mitosis2CX, Mitosis2CY, Mitosis2CZ
from model_analysis import plot_confusion_matrix
import numpy as np


def min_entropy(a, b, c):
    ans = None
    if a['pred_entropy'] < b['pred_entropy']:
        if a['pred_entropy'] < c['pred_entropy']:
            ans = a['pred_label']
        else:
            ans = c['pred_label']
    else:
        if b['pred_entropy'] < c['pred_entropy']:
            ans = b['pred_label']
        else:
            ans = c['pred_label']
    return ans



n_of_it = 10
mito_runs = list()
path = '/root/projects/three_channel/XYZ2'

for m_it in range(n_of_it):
    mtmp = Mitosis2CX(path, m_it)
    #mito_runs.append({'X': mtmp, 'Y': mtmp, 'Z': mtmp})
    mito_runs.append( {'X': Mitosis2CX(path, m_it), 'Y': Mitosis2CY(path, m_it), 'Z': Mitosis2CZ(path, m_it)})
    mito_runs[m_it]['X'].run_me()
    mito_runs[m_it]['Y'].run_me()
    mito_runs[m_it]['Z'].run_me()

    print("itteration_{0} complete.".format(str(m_it).zfill(2)))


master_table = {k: {} for k in mito_runs[0]['X'].mito_labels.keys()}

p_master_table = {k: {'pred_labels_X': [], 'pred_entropy_X': [], 'pred_uid_X': [], 'probability_X':[],
                      'pred_labels_Y': [], 'pred_entropy_Y': [], 'pred_uid_Y': [], 'probability_Y':[],
                      'pred_labels_Z': [], 'pred_entropy_Z': [], 'pred_uid_Z': [], 'probability_Z':[]
                      } for k in mito_runs[0]['X'].pred_mito_labels.keys()}




for k in range(n_of_it):
    for lname in mito_runs[0]['X'].mito_labels.keys():
        xkeys = mito_runs[0]['X'].mito_labels[lname].keys()
        ykeys = mito_runs[0]['Y'].mito_labels[lname].keys()
        zkeys = mito_runs[0]['Z'].mito_labels[lname].keys()
        common = list(set(xkeys) & set(ykeys) & set(zkeys))
        for uid in common:
            master_table[lname][uid] = []
            master_table[lname][uid].append(mito_runs[k]['X'].mito_labels[lname][uid])
            master_table[lname][uid].append(mito_runs[k]['Y'].mito_labels[lname][uid])
            master_table[lname][uid].append(mito_runs[k]['Z'].mito_labels[lname][uid])

# Make 2 column table from dictionaries
for k in range(n_of_it):
    cm_input = {ky: {'true_labels': [], 'pred_labels': []} for ky in master_table.keys()}
    for lname in master_table.keys():
        for uid in master_table[lname].keys():
            cm_input[lname]['true_labels'].append(master_table[lname][uid][3*k]['true_label'])
            cm_input[lname]['pred_labels'].append(min_entropy(master_table[lname][uid][3*k], master_table[lname][uid][3*k+1],
                                                              master_table[lname][uid][3*k+2]))




# for k in range(n_of_it):
#     for lname in mito_runs[0].pred_mito_labels.keys():
#         p_master_table[lname]['pred_labels_X'] += mito_runs[k]['X'].pred_mito_labels[lname]['pred_labels']
#         p_master_table[lname]['pred_entropy_X'] += mito_runs[k]['X'].pred_mito_labels[lname]['pred_entropy']
#         p_master_table[lname]['pred_uid_X'] += mito_runs[k]['X'].pred_mito_labels[lname]['pred_uid']
#         p_master_table[lname]['probability_X'] +=  mito_runs[k]['X'].pred_mito_labels[lname]['probability']
#         p_master_table[lname]['pred_labels_Y'] += mito_runs[k]['Y'].pred_mito_labels[lname]['pred_labels']
#         p_master_table[lname]['pred_entropy_Y'] += mito_runs[k]['Y'].pred_mito_labels[lname]['pred_entropy']
#         p_master_table[lname]['pred_uid_Y'] += mito_runs[k]['Y'].pred_mito_labels[lname]['pred_uid']
#         p_master_table[lname]['probability_Y'] +=  mito_runs[k]['Y'].pred_mito_labels[lname]['probability']
#         p_master_table[lname]['pred_labels_Z'] += mito_runs[k]['Z'].pred_mito_labels[lname]['pred_labels']
#         p_master_table[lname]['pred_entropy_Z'] += mito_runs[k]['Z'].pred_mito_labels[lname]['pred_entropy']
#         p_master_table[lname]['pred_uid_Z'] += mito_runs[k]['Z'].pred_mito_labels[lname]['pred_uid']
#         p_master_table[lname]['probability_Z'] +=  mito_runs[k]['Z'].pred_mito_labels[lname]['probability']

print(cm_input['train']['true_labels'])
print("----------------------------------------------------------------------")
print(cm_input['train']['pred_labels'])

fig, ax = plot_confusion_matrix(cm_input['train']['true_labels'], cm_input['train']['pred_labels'])
cmtrain = os.path.join(path, 'CM_master_train.png')
fig.savefig(cmtrain, bbox_extra_artists=(ax,), bbox_inches='tight')
plt.close(fig)

fig, ax = plot_confusion_matrix(cm_input['test']['true_labels'], cm_input['test']['pred_labels'])
cmtest = os.path.join(path, 'CM_master_test.png')
fig.savefig(cmtest, bbox_extra_artists=(ax,), bbox_inches='tight')
plt.close(fig)

# all_preds_labels = np.array(p_master_table['all']['pred_labels'])
# all_preds_entropy = np.array(p_master_table['all']['pred_entropy'])
# nz_preds_entropy = all_preds_entropy[all_preds_labels != 0]
#
# n, bins, patches = plt.hist([all_preds_entropy, nz_preds_entropy], 25, density=True, alpha=0.75)
# cmhist = os.path.join(path, 'Master_hist.png')
# plt.savefig(cmhist)
# plt.close()
#
# precision, recall = mito_runs[0].precision_recall_vec(master_table['test']['true_labels'],
#                                                       master_table['test']['probability'])
# prplot = os.path.join(path, "CM_precision_recall.png")
# mito_runs[0].plot_prec_recall(precision, recall, prplot)


print("all Done.")
