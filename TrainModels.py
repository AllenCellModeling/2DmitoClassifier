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


n_of_it = 1
mito_runs = list()

try:
    os.makedirs('/output/3ch_out/X')
except FileExistsError:
    pass

try:
    os.makedirs('/output/3ch_out/Y')
except FileExistsError:
    pass

try:
    os.makedirs('/output/3ch_out/Z')
except FileExistsError:
    pass

path = '/output/3ch_out'
csv_path = '/input/master2.csv'
splits_path = '/input/splits.pkl'

for m_it in range(n_of_it):
    mtmp = Mitosis2CX(path, m_it)
    #mito_runs.append({'X': mtmp, 'Y': mtmp, 'Z': mtmp})
    mito_runs.append( {'X': Mitosis2CX(path, m_it), 'Y': Mitosis2CY(path, m_it), 'Z': Mitosis2CZ(path, m_it)})
    mito_runs[m_it]['X'].run_me(csv_path, splits_path)
    mito_runs[m_it]['Y'].run_me(csv_path, splits_path)
    mito_runs[m_it]['Z'].run_me(csv_path, splits_path)

    print("itteration_{0} complete.".format(str(m_it).zfill(2)))


master_table = {k: {} for k in mito_runs[0]['X'].mito_labels.keys()}

common = {'test': set(), 'train': set()}
for k in range(n_of_it):
    for lname in mito_runs[0]['X'].mito_labels.keys():
        xkeys = mito_runs[k]['X'].mito_labels[lname].keys()
        ykeys = mito_runs[k]['Y'].mito_labels[lname].keys()
        zkeys = mito_runs[k]['Z'].mito_labels[lname].keys()
        if k == 0:
            common[lname] = set(xkeys) & set(ykeys) & set(zkeys)
        else:
            common[lname] = set(xkeys) & set(ykeys) & set(zkeys) & common[lname]
#for kname in common.keys():
#    common[kname] = list(common[kname])
for k in range(n_of_it):
    for lname in mito_runs[0]['X'].mito_labels.keys():
        for uid in common[lname]:
            master_table[lname][uid] = []

import datetime
lname_cache = None
uid_cache = None
k_cache = None
try:
    for k in range(n_of_it):
        for lname in mito_runs[0]['X'].mito_labels.keys():
            for uid in common[lname]:
                lname_cache = lname
                uid_cache = uid
                k_cache = k
                master_table[lname][uid].append(mito_runs[k]['X'].mito_labels[lname][uid])
                master_table[lname][uid].append(mito_runs[k]['Y'].mito_labels[lname][uid])
                master_table[lname][uid].append(mito_runs[k]['Z'].mito_labels[lname][uid])
except Exception as e:
    print(datetime.datetime.now())
    print(str(e))
    print(f"{k_cache}:{lname_cache}:{uid_cache}")

cm_input = {k:{'true_labels':[], 'pred_labels':[]} for k in master_table.keys() }

for k in range(n_of_it):
    for lname in master_table.keys():
        for uid in master_table[lname].keys():
            cm_input[lname]['true_labels'].append(master_table[lname][uid][3*k]['true_label'])
            cm_input[lname]['pred_labels'].append(  # master_table[lname][uid][3*k+2]['pred_label'])
                    min_entropy(master_table[lname][uid][3*k], master_table[lname][uid][3*k+1],
                                master_table[lname][uid][3*k+2]))

    fig, ax = plot_confusion_matrix(cm_input['train']['true_labels'], cm_input['train']['pred_labels'])
    cmtrain = os.path.join(path, 'CM_master_train.png')
    fig.savefig(cmtrain, bbox_extra_artists=(ax,), bbox_inches='tight')
    plt.close(fig)

    fig, ax = plot_confusion_matrix(cm_input['test']['true_labels'], cm_input['test']['pred_labels'])
    cmtest = os.path.join(path, 'CM_master_test.png')
    fig.savefig(cmtest, bbox_extra_artists=(ax,), bbox_inches='tight')
    plt.close(fig)