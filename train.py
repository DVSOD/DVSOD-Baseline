import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
from tqdm import tqdm
from datetime import datetime
from data import get_loader
from utils import clip_gradient, adjust_lr
from test import eval_data
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
from model.VSOD_model import RGBDVSODModel

cudnn.benchmark = True

#Train_setup = [0, False, True]    # without temporal cues
Train_setup = [0, False, False]    # the DVSOD baseline

videos_ROOT = 'Your_DViSal_dataset_path/'
ckpt_path = 'Your_ckpt_save_path/'
val_setName= 'test_PTB'

writer = SummaryWriter()
parser = argparse.ArgumentParser()
parser.add_argument('--is_ResNet', type=bool,   default=True,   help='VGG or ResNet backbone')
parser.add_argument('--snapshot',  type=int,    default=Train_setup[0], help='load checkpoint number')
parser.add_argument('--ckpt_load', type=bool,   default=Train_setup[1], help='whether load checkpoint or not')
parser.add_argument('--baseline_mode',type=bool,default=Train_setup[2], help='whether apply baseline mode or not')
parser.add_argument('--sample_rate',type=int,   default=3,      help='sample rate')
parser.add_argument('--stm_queue_size',type=int,default=3,      help='stm queue size')
parser.add_argument('--win_size', type=int,     default=-1,     help='if not given, mem size + 1')
parser.add_argument('--save_interval',type=int, default=2,      help='saving ckpt per N epochs')
parser.add_argument('--epoch', type=int,        default=200,    help='epoch number')
parser.add_argument('--lr', type=float,         default=1e-4,   help='learning rate')
parser.add_argument('--batchsize', type=int,    default=2,     help='training batch size')
parser.add_argument('--trainsize', type=int,    default=320,    help='training dataset size')
parser.add_argument('--clip', type=float,       default=0.5,    help='gradient clipping margin')
parser.add_argument('--decay_rate',type=float,  default=0.1,    help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int,  default=150,    help='every n epochs decay learning rate')
opt = parser.parse_args()
print('Learning Rate: {} ResNet: {}'.format(opt.lr, opt.is_ResNet))

if opt.win_size == -1:
    opt.win_size = opt.stm_queue_size + 1
interval = [opt.win_size - 1, 0]
assert(len(interval) == 2 and interval[0] >= 0 and interval[1] >= 0)

train_loader = get_loader(videos_ROOT, batchsize=opt.batchsize, trainsize=opt.trainsize,
                          subset='train', augmentation=True, interval=interval, sample_rate=opt.sample_rate,
                          baseMode=opt.baseline_mode)
val_loader = get_loader(videos_ROOT, batchsize=1, trainsize=opt.trainsize, shuffle=False, pin_memory=False,
                          subset=val_setName, augmentation=False, interval=interval, sample_rate=opt.sample_rate,
                          baseMode=opt.baseline_mode)

# build models
model = RGBDVSODModel(opt, isTrain=True, is_ResNet=opt.is_ResNet)

if opt.ckpt_load:
    if opt.is_ResNet:
        model.load_state_dict(torch.load(ckpt_path + 'model.pth.' + str(opt.snapshot)))
    else:
        model.load_state_dict(torch.load(ckpt_path + 'model.pth.' + str(opt.snapshot)))

cuda = torch.cuda.is_available()
if cuda:
    model.cuda()

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

total_step = len(train_loader)
CEloss = torch.nn.BCEWithLogitsLoss()


def train(opt, train_loader, model, optimizer, epoch):
    model.train()
    for i, pack in enumerate(tqdm(train_loader), start=1):
        iteration = i + epoch*len(train_loader)
        optimizer.zero_grad()

        '''~~~Model Inputs~~~'''
        images, gts, depths, _ = pack
        images = Variable(images)
        gts = Variable(gts)
        depths = Variable(depths)
        if cuda:
            images = images.cuda()
            gts = gts.cuda()
            depths = depths.cuda()
        (Att_r, Out_r, Att_d, Out_d), (Att, Pred), Output = model(images, depths)
        loss11_rgb = CEloss(Att_r, gts)
        loss12_rgb = CEloss(Out_r, gts)
        loss11_dep = CEloss(Att_d, gts)
        loss12_dep = CEloss(Out_d, gts)
        loss1 = (loss11_rgb + loss12_rgb + loss11_dep + loss12_dep) / 4.0
        loss2_rgb = CEloss(Att, gts)
        loss2_dep = CEloss(Pred, gts)
        loss2 = (loss2_rgb + loss2_dep) / 2.0
        loss3 = CEloss(Output, gts)
        loss_seg = (loss1 + loss2 + loss3) / 3.0
        loss = loss_seg

        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        '''~~~END~~~'''

        if i % 400 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))
        writer.add_scalar('Loss/rgb', loss.item(), iteration)
        writer.add_images('Results/Pred', Output.sigmoid(), iteration)

    if opt.is_ResNet:
        save_path = ckpt_path
    else:
        save_path = ckpt_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) % opt.save_interval == 0:
        torch.save(model.state_dict(), save_path + 'model.pth' + '.%d' % (epoch + 1))


print("Let's go!")
for epoch in range(1, opt.epoch):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(opt, train_loader, model, optimizer, epoch)
    if (epoch+1) % opt.save_interval == 0:
        ckpt_name = str(epoch+1)
        eval_data(opt, val_loader, ckpt_name, is_ResNet=opt.is_ResNet, ckpt_path=ckpt_path)
    if epoch >= opt.epoch -1:
        writer.close()
