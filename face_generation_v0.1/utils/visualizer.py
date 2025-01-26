import os
from . import util


class Visualizer:
    def __init__(self, opt):
        self.opt = opt
        self.img_dir = opt.results_dir

    def display_current_results(self, visuals, total_iters, dataset='train', save_results=True, count=0,
                                name=None, add_image=True):
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        for label, image in visuals.items():
            for i in range(image.shape[0]):
                image_numpy = util.tensor2im(image[i])
                if add_image:
                    self.writer.add_image(label + '%s_%02d' % (dataset, i + count),
                                          image_numpy, total_iters, dataformats='HWC')
                if save_results:
                    # save_path = os.path.join(self.img_dir, dataset, 'epoch_%s_%06d' % (epoch, total_iters))
                    save_path = self.img_dir
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path)

                    if name is not None:
                        img_path = os.path.join(save_path, '%s.png' % name)
                    else:
                        img_path = os.path.join(save_path, '%s_%03d.png' % (label, i + count))
                    util.save_image(image_numpy, img_path)
