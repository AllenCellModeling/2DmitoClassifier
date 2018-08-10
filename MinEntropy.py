import os, sys

sys.path.append("src")

import numpy as np
import pandas as pd
import json
import datasetdatabase as dsdb
import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, utils, models

from pytorch_learning_tools.data_providers.DataframeDataProvider import DataframeDataProvider
from pytorch_learning_tools.data_providers.DataframeDataset import DatasetSingleRGBImageToTarget, \
    DatasetSingleRGBImageToTargetUniqueID
from pytorch_learning_tools.utils.data_utils import classes_and_weights

import collections



class MitosisClassifier(object):
    """This is a script to automate the running of the mitotic program

    Attributes:


    """

    def __init__(self,df, f_label='mitosis_label'):
        self.GPU_ID = 1
        self.BATCH_SIZE = 64  # 64
        self.base_path = None
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

        self.df = df

        self.modelX = None
        self.modelY = None
        self.modelZ = None

        self.dpX = None
        self.dpY = None
        self.dpZ = None

        self.mito_labels = None

    def set_basepath(self, bp):
        self.base_path = bp

    def class_names(self):
        return self.m_class_names

    def get_dpX(self):
        return self.dpX

    def read_and_filter_input(self, f_label='mitosis_label'):
        dbConnectionInfo = json.load(open('/allen/aics/modeling/jamies/projects/dbconnect/configs.json', 'r'))
        mngr = dsdb.ConnectionManager(user="jamies")
        mngr.add_connections(dbConnectionInfo)
        prod = mngr.connect('prod')
        dfio = prod.get_dataset(1)
        #dfio['save_flat_proj_reg_path'] = '/allen/aics/modeling/PIPELINE/2018-07-23-17:20:57/' + dfio['save_flat_proj_reg_path']
        dfio['target_numeric'] = -1
        df = dfio.copy(deep=True)
        return df

    def create_data_providers_xyz(self, df):
        self.dpX = self.create_data_provider(df, self.transform_x())
        self.dpY = self.create_data_provider(df, self.transform_y())
        self.dpZ = self.create_data_provider(df, self.transform_z())

    def create_data_provider(self, df, transform):
        split_fracs = {'all': 1.0}
        split_seed = 1

        dataset_kwargs = {split: {'target': 'target_numeric', 'image': self.f_type,
                                  'uniqueID': self.f_type} for split in split_fracs.keys()}
        dataloader_kwargs = {
            split: {'batch_size': self.BATCH_SIZE, 'shuffle': False, 'drop_last': False, 'num_workers': 4,
                    'pin_memory': True} for split in split_fracs.keys()}

        dataset_kwargs['all']['imageTransform'] = transform
        dp = DataframeDataProvider(df, datasetClass=DatasetSingleRGBImageToTargetUniqueID,
                                   split_fracs=split_fracs,
                                   split_seed=split_seed,
                                   uniqueID=self.f_type,
                                   dataset_kwargs=dataset_kwargs,
                                   dataloader_kwargs=dataloader_kwargs)
        return dp

    def ofname(self, b_name, f_ext):
        fname = "{0}_{1}.{2}".format(datetime.datetime.now().isoformat(timespec='minutes'), b_name, f_ext)
        return os.path.join(self.base_path, fname)

    def generate_class_weights(self):
        classes, weights = classes_and_weights(self.dp, split='train', target_col='target_numeric')
        weights = weights.cuda(self.GPU_ID)
        CWP = collections.namedtuple('CWP', ['cls', 'weights'])
        return CWP(classes, weights=weights)

    def phases(self):
        return self.dp.dataloaders.keys()

    def pred_phases(self):
        return self.dp

    def load_model(self, model_path):
        # cwp = self.generate_class_weights()
        state_dict = torch.load(model_path)
        model_name = 'resnet18'
        model_class = getattr(models, model_name)  # models is from the imports at the top
        model = model_class()
        model.fc = nn.Linear(model.fc.in_features, 8)  # len(cwp.cls))
        model.load_state_dict(state_dict)
        model = model.cuda(self.GPU_ID)
        model.eval()
        return model
    #
    # def load_models_xyz(self, mPathX, mPathY, mPathZ):
    #     self.modelX = self.load_model(mPathX)
    #     self.modelY = self.load_model(mPathY)
    #     self.modelZ = self.load_model(mPathZ)

    def apply_models(self, xpath, ypath, zpath):
        mito_labels = {k: {'x_pred_labels': [], 'x_pred_entropy': [], 'x_pred_uid': [], 'x_probability': [],
                           'y_pred_labels': [], 'y_pred_entropy': [], 'y_pred_uid': [], 'y_probability': [],
                           'z_pred_labels': [], 'z_pred_entropy': [], 'z_pred_uid': [], 'z_probability': [],
                           'pred_labels': []
                           } for k in self.dpX.dataloaders.keys()}

        self.apply_single_model(self.dpX, mito_labels,
                                'x_pred_labels', 'x_pred_entropy', 'x_pred_uid', 'x_probability', xpath)

        self.apply_single_model(self.dpY, mito_labels,
                                'y_pred_labels', 'y_pred_entropy', 'y_pred_uid', 'y_probability', ypath)

        self.apply_single_model(self.dpZ, mito_labels,
                                'z_pred_labels', 'z_pred_entropy', 'z_pred_uid', 'z_probability', zpath)

        self.apply_min_entropy(mito_labels)
        self.mito_labels = mito_labels

    def apply_single_model(self, dp_h, mlabels, h_pred_labels, h_pred_entropy, h_pred_uid, h_probability, mPath):
        model = self.load_model(mPath)
        for phase in dp_h.dataloaders.keys():
            for i, mb in enumerate(dp_h.dataloaders[phase]):
                x = mb['image']
                u = mb['uniqueID']

                y_hat_pred = model(Variable(x).cuda(self.GPU_ID))
                _, y_hat = y_hat_pred.max(1)

                probs = F.softmax(y_hat_pred, dim=1)
                entropy = -torch.sum(probs * torch.log(probs), dim=1)

                pred_labels = list(y_hat.data.cpu().numpy())
                pred_entropy = list(entropy.data.cpu().numpy())

                mlabels[phase][h_pred_labels] += pred_labels
                mlabels[phase][h_pred_entropy] += pred_entropy
                mlabels[phase][h_pred_uid] += u
                mlabels[phase][h_probability] += list(probs.data.cpu())

    def apply_min_entropy(self, mlabels):
        for phase in self.dpX.dataloaders.keys():
            for i in range(len(mlabels[phase]['x_pred_labels'])):
                mlabels[phase]['pred_labels'].append(self.min_entropy(mlabels[phase]['x_pred_labels'][i],
                                                                      mlabels[phase]['x_pred_entropy'][i],
                                                                      mlabels[phase]['y_pred_labels'][i],
                                                                      mlabels[phase]['y_pred_entropy'][i],
                                                                      mlabels[phase]['z_pred_labels'][i],
                                                                      mlabels[phase]['z_pred_entropy'][i])
                                                     )

    def save_out(self, create_csv=False):
        uidKey = self.dpX.opts['uniqueID']
        df_pred = pd.DataFrame({'MitosisLabelPredicted': self.mito_labels['all']['pred_labels'],
                                'X_MitoLabel': self.mito_labels['all']['x_pred_labels'],
                                'Y_MitoLabel': self.mito_labels['all']['y_pred_labels'],
                                'Z_MitoLabel': self.mito_labels['all']['z_pred_labels'],
                                'X_MitoEntropy': self.mito_labels['all']['x_pred_entropy'],
                                'Y_MitoEntropy': self.mito_labels['all']['y_pred_entropy'],
                                'Z_MitoEntropy': self.mito_labels['all']['z_pred_entropy'],
                                'X_MitoProb': self.mito_labels['all']['x_probability'],
                                'Y_MitoProb': self.mito_labels['all']['y_probability'],
                                'Z_MitoProb': self.mito_labels['all']['z_probability'],
                                uidKey: self.mito_labels['all']['x_pred_uid']})
        df_out = pd.merge(self.dpX.dfs['all'], df_pred, how='inner', on=uidKey)
        df_out = df_out.drop(columns='target_numeric')
        if create_csv:
            fname = self.ofname('mitotic_predictions_on_unannotated_cells', 'csv')
            df_out.to_csv(fname)
        return df_out

    def run_me(self, modellist):

        xpath = modellist[0]
        ypath = modellist[1]
        zpath = modellist[2]
        #df = self.read_and_filter_input()
        self.create_data_providers_xyz(self.df)
        #self.load_models_xyz(xpath, ypath, zpath)
        self.apply_models()
        df_out = self.save_out()
        return df_out

    @staticmethod
    def min_entropy(xl, xe, yl, ye, zl, ze):  # xl => x_label, xe => x_entropy
        ans = zl
        if ze < xe and ze < ye:
            return ans
        if xe < ye:
            ans = xl
        else:
            ans = yl
        return ans

    @staticmethod
    def transform_x():
        w = 224
        s = 2.5

        # xyz once the image has been scaled by s
        nwidth = int(round(170 * s))
        nheight = int(round(230 * s))

        # offsets for each pain within the image
        xo = 0
        yo = 90

        # start coordinates for each fram
        x1s = xo + 0
        y1s = yo + 0

        blank = torch.zeros([w, w])

        return transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize(nwidth), transforms.CenterCrop((nheight, nwidth)),
             transforms.ToTensor(), transforms.Lambda(lambda x: torch.stack(
                [x[0, y1s:(w + y1s), x1s:(w + x1s)],
                 blank,
                 x[2, y1s:(w + y1s), x1s:(w + x1s)]
                 ]))])

    @staticmethod
    def transform_y():
        w = 224
        s = 2.5

        # this is the xyz geometry imbeded in the image
        gz = 64

        # xyz once the image has been scaled by s
        nz = int(round(s * gz))

        nwidth = int(round(170 * s))
        nheight = int(round(230 * s))

        # offsets for each pain within the image
        y2o = 50
        x3o = 40

        # start coordinates for each fram
        y2s = y2o + (nheight - w - 50)
        x3s = x3o + nz

        blank = torch.zeros([w, w])

        return transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize(nwidth), transforms.CenterCrop((nheight, nwidth)),
             transforms.ToTensor(), transforms.Lambda(lambda x: torch.stack(
                [
                    x[0, y2s:(w + y2s), x3s:(w + x3s)],
                    blank,
                    x[2, y2s:(w + y2s), x3s:(w + x3s)]
                ]))])

    @staticmethod
    def transform_z():
        w = 224
        s = 2.5

        # this is the xyz geometry imbeded in the image
        gz = 64

        # xyz once the image has been scaled by s
        nz = int(round(s * gz))

        nwidth = int(round(170 * s))
        nheight = int(round(230 * s))

        # offsets for each pain within the image
        yo = 90
        x2o = 40

        # start coordinates for each fram
        y1s = yo + 0
        x2s = x2o + nz

        blank = torch.zeros([w, w])

        return transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize(nwidth), transforms.CenterCrop((nheight, nwidth)),
             transforms.ToTensor(), transforms.Lambda(lambda x: torch.stack(
                [x[0, y1s:(w + y1s), x2s:(w + x2s)],
                 blank,
                 x[2, y1s:(w + y1s), x2s:(w + x2s)]
                 ]))])


# This is a hack to bootstrap models trained on aics001 into newer version of pytorch
#
# import torch._utils
# try:
#     torch._utils._rebuild_tensor_v2
# except AttributeError:
#     def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
#         tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
#         tensor.requires_grad = requires_grad
#         tensor._backward_hooks = backward_hooks
#         return tensor
#     torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

def mito_runner(dataset, params):
    m = params["mito_classifier"](df=dataset, **params["mito_init"])
    df = m.run_me()
    return df

if __name__ ==  '__main__':



    prod.process_run(mito_runner
    1,
    alg_parameters = {"mito_classifier": MitosisClassifer,
                      "mito_init", {"parameter": '/allen/aics/modeling/jamies/projects/dbconnect/configs.json'}},
    dataset_parameters = {"name": "mito predictions",
                          "description": "this is the first test of mitotic predictions",
                          "filepath_columns": ["new", "filepath", "columns"]})
