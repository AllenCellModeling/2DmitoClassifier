import json
import datasetdatabase as dsdb
from PIL import Image, ImageDraw
import numpy as np
from torchvision.utils import make_grid
from plot_images import torch_to_PIL_single_image, ims_labels_preds_to_grid
from torchvision import transforms

class MakeFigures(object):
    def __init__(self, data_set_id=6):
        manager = dsdb.ConnectionManager('/allen/aics/modeling/jamies/projects/dbconnect/configs.json', user='jamies')
        prod = manager.connect("prod")
        #prod._deep_print()
        self.df = prod.get_dataset(data_set_id, get_info_items=True)
        self.df.drop(self.df.index[self.df['MitosisLabel'] == -1].tolist(), inplace=True)
        #for k in self.df.keys():
        #    print(k)

    def make_figures(self, im_label='save_flat_proj_reg_path'):
        wc = (255,0,0)
        for s_idx in range(1,7): # range(1,6):
            true_list = self.df.index[self.df['MitosisLabel'] == s_idx].tolist()
            for idx in true_list:
                if self.df['MitosisLabel'][idx] == self.df['MitosisLabelPredicted'][idx] and self.df['MitosisLabel'][idx] != 0:
                    self.blue_channel(idx, s_idx, '')
                else:
                    self.blue_channel(idx, s_idx, 'X',channel='save_flat_reg_path', wrong_color=wc)

    def blue_channel(self, idx, state_idx, pre_tag, channel='save_flat_proj_reg_path', wrong_color=(255,255,255)):
        fname = self.df[channel][idx]
        ikey = self.df['MitosisLabelPredicted(IotaId)'][idx]
        true_label = self.df['MitosisLabel'][idx]
        pred_label = self.df['MitosisLabelPredicted'][idx]
        im = Image.open(fname)
        arr = np.array(im)
        arr[:, :, 0] = arr[:, :, 2]
        arr[:, :, 1] = arr[:, :, 2]
        im = Image.fromarray(arr)
        d = ImageDraw.Draw(im)
        d.text((10, 10), "true: {}".format(true_label))  #  , fill=wrong_color)
        d.text((10, 20), 'pred: {}'.format(pred_label))  #  , fill=wrong_color)
        d.text((10, 30), 'key: {}'.format(ikey))
        im.save('/allen/aics/modeling/jamies/Data/iRevHR/state_{0}/{2}rev_{1}.png'.format(state_idx, ikey, pre_tag))


if __name__ == '__main__':
    mf = MakeFigures()
    mf.make_figures()
