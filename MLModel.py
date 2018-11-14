import os, sys, gc, copy, itertools, json
from yaml import load

sys.path.append("src")
from train_model import train_model
from model_analysis import model_analysis, torch_confusion_matrix, plot_confusion_matrix
from plot_images import torch_to_PIL_single_image, ims_labels_to_grid, ims_preds_to_grid, ims_labels_preds_to_grid

import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, utils, models
from torchvision.utils import make_grid

from tensorboardX import SummaryWriter

from sklearn.metrics import precision_recall_curve
from itertools import cycle

import matplotlib
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
from IPython.core.display import display

from pytorch_learning_tools.data_providers.DataframeDataProvider import DataframeDataProvider
from pytorch_learning_tools.data_providers.DataframeDataset import DatasetSingleRGBImageToTarget, \
    DatasetSingleRGBImageToTargetUniqueID
from pytorch_learning_tools.utils.dataframe_utils import filter_dataframe
from pytorch_learning_tools.utils.data_utils import classes_and_weights

import seaborn as sns
import collections
import pickle
from models.irena_classification import IrenaClassification
from sqlalchemy import create_engine


class MitosisClassifier(object):
    """This is a script to automate the running of the mitotic program

    Attributes:


    """

    def __init__(self, base_path, iter_n=0, f_label='mitosis_label'):
        self.GPU_ID = 1
        self.BATCH_SIZE = 64
        self.base_path = base_path
        self.iter_n = iter_n
        self.greg_data = '/root/aics/modeling/gregj/results/ipp/ipp_17_12_03/'
        self.csv_input = 'input_data_files/mito_annotations_all.csv' #
        self.f_type = 'save_flat_proj_reg_path'
        self.f_label = f_label
        # human readable classes
        self.m_class_names = np.array(["not mitotic",
                                     "M1: prophase 1",
                                     "M2: prophase 2",
                                     "M3: pro metaphase 1",
                                     "M4: pro metaphase 2",
                                     "M5: metaphase",
                                     "M6: anaphase",
                                     "M7: telophase-cytokinesis"])
        self.dp = None
        self.dp_no_annots = None
        self.pred_mito_labels = None

    def class_names(self):
        return self.m_class_names


    def prepend_path(self, val):
        return os.path.join('/root/aics/modeling/gregj/results/ipp/ipp_17_12_03/', val)

    def read_and_filter_input(self, f_label='mitosis_label'):

        filt = f_label + ' >= 0'

        conn_string = "postgresql://ro:BR9p66@pg-aics-modeling-01/pg_modeling"
        conn = create_engine(conn_string)
        dfio = pd.read_sql_table('irena_classifications', conn, index_col='id')

        df = dfio.copy(deep=True)

        # add absolute path  '/root/aics/modeling/gregj/results/ipp/ipp_17_12_03/' + df[self.f_type]
        df[self.f_type].apply(self.prepend_path)
        #df[self.f_type] = os.path.join('/root/aics/modeling/gregj/results/ipp/ipp_17_12_03/', str(df[self.f_type]))

        df['targetNumeric'] = -1
        df['targetNumeric'] = df['targetNumeric'].astype(np.int64)

        return df

    def transform(self):
        pass


    def create_data_provider(self, df):
        split_fracs = {'data': 1.0}
        split_seed = 1

        dataset_kwargs = {'data': {'target': 'target_numeric', 'image': self.f_type, 'uniqueID': 'save_h5_reg_path'}}
        dataloader_kwargs = {
            'data': {'batch_size': self.BATCH_SIZE, 'shuffle': False, 'drop_last': False, 'num_workers': 4, 'pin_memory': True}}

        dataset_kwargs['data']['imageTransform'] = self.transform()

        self.dp = DataframeDataProvider(df, datasetClass=DatasetSingleRGBImageToTargetUniqueID,
                                   split_fracs=split_fracs,
                                   split_seed=split_seed,
                                   uniqueID='save_h5_reg_path',
                                   dataset_kwargs=dataset_kwargs,
                                   dataloader_kwargs=dataloader_kwargs)

    def check_images(self, model, dkey='data'):
        i, mb = next(enumerate(self.dp.dataloaders[dkey]))
        print("mb")
        ims_labels_preds = [(im, label, pred) for i, (im, label, pred) in enumerate(
            zip(mb['image'], mb['target'].numpy(),
                model(Variable(mb['image']).cuda(self.GPU_ID)).data.cpu().max(1)[1].numpy())) if i < 16]
        img = ims_labels_preds_to_grid(ims_labels_preds, ncol=4)
        fname = "Inspect_{0}".format(dkey)
        img.save(self.ofname(fname, "png"))


    def ofname(self, b_name, f_ext):
        fname = "{0}_{1}.{2}".format(b_name, str(self.iter_n).zfill(2), f_ext)
        return os.path.join(self.base_path, fname)

    def generate_class_weights(self):
        classes, weights = classes_and_weights(self.dp, split='train', target_col='target_numeric')
        weights = weights.cuda(self.GPU_ID)
        CWP = collections.namedtuple('CWP', ['cls', 'weights'])
        return CWP(classes, weights = weights)

    def phases(self):
        return self.dp.dataloaders.keys();

    def pred_phases(self):
        return self.dp

    def load_model(self):
        model_name = 'resnet18'
        model_class = getattr(models, model_name)
        model = model_class()

        len_classes = 8
        state_dict = torch.load(self.ofname('saved_model_10E', 'pt'))
        model.fc = nn.Linear(model.fc.in_features, len_classes)
        model.load_state_dict(state_dict)

        model = model.cuda(self.GPU_ID)
        return(model)

    def select_n_train_n_run_model(self):

        model = self.load_model()


        # self.check_images(model, 'data')

        print("get predictions.")
        # Get predictions
        model.eval()

        p_mito_labels = {phase: {'pred_labels': [], 'pred_entropy': [], 'pred_uid': [], 'probability': []} for phase in
                       self.dp_no_annots.dataloaders.keys()}

        for phase in self.dp_no_annots.dataloaders.keys():
            for i, mb in tqdm_notebook(enumerate(self.dp_no_annots.dataloaders[phase]),
                                       total=len(self.dp_no_annots.dataloaders[phase]), postfix={'phase': phase}):
                x = mb['image']
                y = mb['target']
                u = mb['uniqueID']

                y_hat_pred = model(Variable(x).cuda(self.GPU_ID))
                _, y_hat = y_hat_pred.max(1)

                probs = F.softmax(y_hat_pred.data.cpu(), dim=1)
                entropy = -torch.sum(probs * torch.log(probs), dim=1)

                p_mito_labels[phase]['pred_labels'] += list(y_hat.data.cpu().numpy())
                p_mito_labels[phase]['pred_entropy'] += list(entropy.data.cpu().numpy())
                p_mito_labels[phase]['probability'] += list(probs.numpy())
                p_mito_labels[phase]['pred_uid'] += u

        self.pred_mito_labels = p_mito_labels

    def save_out(self):
        df_pred = pd.DataFrame({'MitosisLabelPredicted': self.pred_mito_labels['all']['pred_labels'],
                                'MitosisLabelPredictedEntropy': self.pred_mito_labels['all']['pred_entropy'],
                                self.dp_no_annots.opts['uniqueID']: self.pred_mito_labels['all']['pred_uid'],
                                'MitosisLabelProbability': self.pred_mito_labels['all']['probability']})

        df_out = pd.merge(self.dp_no_annots.dfs['all'], df_pred, how='inner', on=self.dp_no_annots.opts['uniqueID'])
        df_out = df_out.drop(columns='target_numeric')
        fname = self.ofname('mitotic_predictions_on_unannotated_cells', 'csv')
        df_out.to_csv(fname)

    def run_me(self):
        df = self.read_and_filter_input()
        self.create_data_provider(df)
        print("ready to run.")
        self.select_n_train_n_run_model()
        self.save_out()
        print("finished.")

    def precision_recall_vec(self, tru_label, probs):
        precision = []
        recall = []
        for idx in range(8):
            y_test = [1 if (x == idx) else 0 for x in tru_label]
            y_score = [x[idx] for x in probs]
            tpre, trec, _ = precision_recall_curve(y_test, y_score)
            precision.append(tpre)
            recall.append(trec)
        return (precision, recall)

    def plot_prec_recall(self, precision, recall, fname):
        labels = ['not mitotic', 'M1: prophase 1', 'M2: prophase 2', 'M3: pro metaphase 1', 'M4: pro metaphase 2', 'M5: metaphase', 'M6: anaphase', 'M7: telophase-cytokinesis']
        # hexcolors = ['#b3e2cd', '#fdcdac', '#cbd5e8', '#f4cae4', '#e6f5c9', '#fff2ae', '#f1e2cc', '#cccccc']
        colors = sns.color_palette("hls", 8)
        plt.figure()  # figsize=(7, 8))
        lines = []

        for i, color in zip(range(8), colors):
            l, = plt.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
        #    labels.append('Precision-recall for non-mitotic')

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Extension of Precision-Recall curve to multi-class')
        lg = plt.legend(lines, labels, loc=(1.2, .5), prop=dict(size=14))
        plt.savefig(fname, bbox_extra_artists=(lg,), bbox_inches='tight')



class MitosisClassifierThreeChannel(MitosisClassifier):
    def __init__(self, base_path, iter_n=0, f_label='MitosisLabel'):
        MitosisClassifier.__init__(self, base_path, iter_n, f_label)


    def transform(self):
        return self.three_channel_transform()


    def three_channel_transform(self):
        w = 224
        s = 2.5

        # this is the xyz geometry imbeded in the image
        gx = 96
        gy = 128
        gz = 64

        # xyz once the image has been scaled by s
        nx = int(round(s * gx))
        ny = int(round(s * gy))
        nz = int(round(s * gz))

        nwidth = int(round(170 * s))
        nheight = int(round(230 * s))

        # offsets for each pain within the image
        xo = 0
        yo = 90
        x2o = 40
        y2o = 50
        x3o = 40

        # start coordinates for each fram
        x1s = xo + 0
        y1s = yo + 0
        x2s = x2o + nz
        y2s = y2o + (nheight - w - 50)
        x3s = x3o + nz

        return transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize(nwidth), transforms.CenterCrop((nheight, nwidth)),
             transforms.ToTensor(), transforms.Lambda(lambda x: torch.stack(
                [x[2, y1s:(w + y1s), x1s:(w + x1s)], x[2, y1s:(w + y1s), x2s:(w + x2s)],
                 x[2, y2s:(w + y2s), x3s:(w + x3s)]]))])


class MitosisClassifierOneChannel(MitosisClassifier):
    def __init__(self, base_path, iter_n=0, f_label='MitosisLabel'):
        MitosisClassifier.__init__(self, base_path, iter_n, f_label)


    def transform(self):
        return self.one_channel_transform()


    def one_channel_transform(self):
        w = 224

        mask = torch.ones([3, w, w])
        mask[1, :, :] = 0

        return transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize((w, w)),
             transforms.ToTensor(), transforms.Lambda(lambda x: mask*x)])

class MitosisClassifierZProj(MitosisClassifier):
    def __init__(self, base_path, iter_n=0, f_label='MitosisLabel'):
        MitosisClassifier.__init__(self, base_path, iter_n, f_label)
        self.f_type = 'save_flat_reg_path'


    def transform(self):
        return self.z_channel_transform()


    def z_channel_transform(self):
        w = 224

        mask = torch.ones([3, w, w])
        mask[1, :, :] = 0


        return transforms.Compose([transforms.ToPILImage(), transforms.Resize(256), transforms.CenterCrop(224),
                                   transforms.ToTensor(), transforms.Lambda(lambda x: mask*x)])

