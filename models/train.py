import sys
sys.path.append('../')
import os
import re
import time
import argparse
import torch as t
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchnet import meter
from models import configs
from models import network
from utils.dataset import CatDogDataset
from utils.visualize import Visualizer
from torchvision.models.densenet import load_state_dict_from_url


def train(args):
    vis = Visualizer()

    config = getattr(configs, args.model + 'Config')()
    model = getattr(network, args.model)(config).eval()

    if args.pretrain and args.model == 'DenseNet121':
        pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/densenet121-a639ec97.pth', progress=False)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    train_set = CatDogDataset(root_path=config.train_path, config=config, mode='train')
    valid_set = CatDogDataset(root_path=config.train_path, config=config, mode='valid')

    train_dataloader = DataLoader(train_set, config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers)
    valid_dataloader = DataLoader(valid_set, config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers)

    if args.load_model_path:
        model.load(args.load_model_path)
    if args.use_gpu:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)

    train_loss_meter, valid_loss_meter = meter.AverageValueMeter(), meter.AverageValueMeter()
    train_confusion_matrix, valid_confusion_matrix = meter.ConfusionMeter(10), meter.ConfusionMeter(10)

    best_valid_loss = 1e5
    best_epoch = 0
    dist_to_best = 0

    time_begin = time.clock()

    for epoch in range(config.epoch):

        # train
        model.train()
        train_loss_meter.reset()
        train_confusion_matrix.reset()

        for _iter, (train_data, train_target) in enumerate(train_dataloader):

            if args.use_gpu:
                train_data = train_data.cuda()
                train_target = train_target.cuda()

            optimizer.zero_grad()
            train_logits, train_output = model(train_data)
            train_loss = criterion(train_logits, train_target)
            train_loss.backward()
            optimizer.step()

            train_loss_meter.add(train_loss.item())
            train_confusion_matrix.add(train_logits.data, train_target.data)

            if _iter % config.print_freq == 0:
                vis.plot('train_loss', train_loss_meter.value()[0])
        model.save(path=os.path.join(args.ckpts_dir, 'model_{0}.pth'.format(str(epoch))))

        # valid
        model.eval()
        valid_loss_meter.reset()
        valid_confusion_matrix.reset()

        for _iter, (valid_data, valid_target) in enumerate(valid_dataloader):

            if args.use_gpu:
                valid_data = valid_data.cuda()
                valid_target = valid_target.cuda()

            valid_logits, valid_output = model(valid_data)
            valid_loss = criterion(valid_logits, valid_target)

            valid_loss_meter.add(valid_loss.item())
            valid_confusion_matrix.add(valid_logits.detach().squeeze(), valid_target.type(t.LongTensor))

        valid_cm = valid_confusion_matrix.value()
        valid_accuracy = 100. * (valid_cm.diagonal().sum()) / (valid_cm.sum())

        vis.plot('valid_accuracy', valid_accuracy)

        vis.log("epoch:{epoch}, train_loss:{train_loss}, train_cm:{train_cm}, valid_loss:{valid_loss}, valid_cm:{valid_cm}, valid_accuracy:{valid_accuracy}".format(
            epoch=epoch,
            train_loss=train_loss_meter.value()[0],
            train_cm=str(train_confusion_matrix.value()),
            valid_loss=valid_loss_meter.value()[0],
            valid_cm=str(valid_cm),
            valid_accuracy=valid_accuracy
        ))
        print("epoch:{epoch}, train_loss:{train_loss}, valid_loss:{valid_loss}, valid_accuracy:{valid_accuracy}".format(
            epoch=epoch,
            train_loss=train_loss_meter.value()[0],
            valid_loss=valid_loss_meter.value()[0],
            valid_accuracy=valid_accuracy
        ))
        print("train_cm:\n{train_cm}\n\nvalid_cm:\n{valid_cm}".format(
            train_cm=str(train_confusion_matrix.value()),
            valid_cm=str(valid_cm),
        ))

        # early stop
        if valid_loss_meter.value()[0] < best_valid_loss:
            best_epoch = epoch
            best_valid_loss = valid_loss_meter.value()[0]
            dist_to_best = 0

        dist_to_best += 1
        if dist_to_best > 4:
            break

    model.save(path=os.path.join(args.ckpts_dir, 'model.pth'))
    vis.save()
    print("save model successfully")
    print("best epoch: ", best_epoch)
    print("best valid loss: ", best_valid_loss)
    time_end = time.clock()
    print('time cost: %.2f' % (time_end - time_begin))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='ResNet', help="model to be used")
    parser.add_argument('--pretrain', action='store_true', help="whether use pretrained model")
    parser.add_argument('--use_gpu', action='store_true', help="whether use gpu")
    parser.add_argument('--load_model_path', type=str, default=None, help="Path of pre-trained model")
    parser.add_argument('--ckpts_dir', type=str, default=None, help="Dir to store checkpoints")

    args = parser.parse_args()

    if not os.path.exists(args.ckpts_dir):
        os.makedirs(args.ckpts_dir)

    train(args)





