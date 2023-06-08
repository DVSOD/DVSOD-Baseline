import torch
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
from skimage import io
from tqdm import tqdm
from model.VSOD_model import RGBDVSODModel
from data import get_loader


def LOG(logfile, output):
    with open(logfile, 'a') as f:
        f.write(output)

def eval_data(opt, test_loader, ckpt_name, is_ResNet=True, ckpt_path=None):
    model = RGBDVSODModel(opt, isTrain=False, is_ResNet=is_ResNet)
    if is_ResNet:
        model.load_state_dict(torch.load(ckpt_path + 'model.pth.' + str(ckpt_name)))
    else:
        model.load_state_dict(torch.load(ckpt_path + 'model.pth.' + str(ckpt_name)))

    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()
    model.eval()

    if is_ResNet:
        save_path = os.path.join('./results/Resnet/')
    else:
        save_path = os.path.join('./results/VGG/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logfile = os.path.join('./', 'result.txt')

    print('Evaluating dataset:')
    avg_mae, img_num = 0.0, 0.0
    for i, pack in enumerate(tqdm(test_loader), start=1):
        images, gts, depths, name = pack
        images = Variable(images)
        gts = Variable(gts)
        depths = Variable(depths)
        if cuda:
            images = images.cuda()
            gts = gts.cuda()
            depths = depths.cuda()

        _, _, res = model(images, depths)

        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        os.makedirs(save_path+name[1][0],exist_ok=True)
        io.imsave(os.path.join(save_path+name[1][0], name[0][0]), np.uint8(res * 255))


        '''Evaluate MAE'''
        gt = gts.data.cpu().numpy().squeeze()
        mea = np.abs(res - gt).mean()
        if mea == mea:  # for Nan
            avg_mae += mea
            img_num += 1.0
    avg_mae /= img_num
    print('[MAE] The current evaluation metric is: {:.4f} mae.'.format(avg_mae))
    LOG(logfile, 'model.pth.{} CKPT with {:.4f} mae.\n'.format(str(ckpt_name), avg_mae))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=320, help='testing size')
    parser.add_argument('--baseline_mode', type=bool, default=False, help='whether apply baseline mode or not')
    parser.add_argument('--sample_rate', type=int, default=3, help='sample rate')
    parser.add_argument('--stm_queue_size', type=int, default=3, help='stm queue size')
    parser.add_argument('--is_ResNet', type=bool, default=True, help='VGG or ResNet backbone')
    parser.add_argument('--win_size', type=int, default=-1, help='if not given, mem size + 1')
    cfg = parser.parse_args()


    if cfg.win_size == -1:
        cfg.win_size = cfg.stm_queue_size + 1
    interval = [cfg.win_size - 1, 0]
    assert (len(interval) == 2 and interval[0] >= 0 and interval[1] >= 0)

    videos_ROOT = "Your_DViSal_dataset_path/"
    ckpt_path = 'Your_ckpt_save_path/'

    ckpt_name = 'best'       # specify ckpt name, 'best'+.pth
    test_name = 'test_all'   # specify testset, e.g., test_DET, test_track3D

    test_loader = get_loader(videos_ROOT, batchsize=1, trainsize=cfg.testsize, shuffle=False, pin_memory=False,
                             subset=test_name, augmentation=False, interval=interval, sample_rate=cfg.sample_rate)
    eval_data(cfg, test_loader, ckpt_name, is_ResNet=cfg.is_ResNet, ckpt_path= ckpt_path)
