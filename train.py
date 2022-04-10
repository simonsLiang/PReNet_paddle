import os
import argparse
import numpy as np
import paddle
from paddle.io import DataLoader
from DerainDataset import *
from utils import *
from paddle.optimizer.lr import MultiStepDecay
from SSIM import SSIM
from NetWorks import *
import time
import sys
parser = argparse.ArgumentParser(description="PReNet_train")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batch-size", type=int, default=18)
# parser.add_argument("--batch_size", type=int, default=18, help="Training batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[30,50,80], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--save_path", type=str, default="./", help='path to save models and logfiles')
parser.add_argument("--save_freq",type=int,default=1,help='save intermediate model')
parser.add_argument("--data_path",type=str, default=r'./RainTrainH',help='path to training data')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
parser.add_argument("--output-dir", type=str, default='')

opt = parser.parse_args()


if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

if not os.path.exists(os.path.join(opt.save_path,'trainlog.txt')):
    f = open(os.path.join(opt.save_path,'trainlog.txt'),'w',encoding='utf8')
    f.close()


def train_one_epoch_paddle(model, data_loader,criterion,optimizer,scheduler,epoch,print_freq):
    model.train()
    scheduler.step()
    train_reader_cost = 0.0
    train_run_cost = 0.0
    total_samples = 0
    reader_start = time.time()
    batch_past = 0

    for batch_idx, (input_train, target_train) in enumerate(data_loader, 0):
        train_reader_cost += time.time() - reader_start
        train_start = time.time()

        output, _ = model(input_train)
        pixel_metric = criterion(output, target_train)
        loss = -pixel_metric
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        train_run_cost += time.time() - train_start
        total_samples += input_train.shape[0]
        batch_past += 1

        if batch_idx > 0 and batch_idx % print_freq == 0:
            msg = "[Epoch {}, iter: {}] lr: {:.5f}, loss: {:.5f}, avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {}, avg_ips: {:.5f} images/sec.".format(
                epoch, batch_idx,
                optimizer.get_lr(),
                loss.item(), train_reader_cost / batch_past,
                                  (train_reader_cost + train_run_cost) / batch_past,
                                  total_samples / batch_past,
                                  total_samples / (train_reader_cost + train_run_cost))
            # just log on 1st device
            if paddle.distributed.get_rank() <= 0:
                print(msg)
                f = open(os.path.join(opt.save_path,'trainlog.txt'), 'a', encoding='utf8')
                f.write(msg+'\n')
                f.close()
            sys.stdout.flush()
            train_reader_cost = 0.0
            train_run_cost = 0.0
            total_samples = 0
            batch_past = 0

        reader_start = time.time()

def main():

    print('Loading dataset ...\n')
    dataset_train = Dataset(data_path=opt.data_path)

    train_batch_sampler = paddle.io.DistributedBatchSampler(dataset=dataset_train,batch_size=opt.batch_size,shuffle=True,drop_last=False)

    loader_train = DataLoader(dataset=dataset_train, num_workers=4,batch_sampler=train_batch_sampler)

    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    model = PReNet(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)

    print_network(model)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    # loss function
    # criterion = nn.MSELoss(size_average=False)
    criterion = SSIM()
    paddle.set_device('gpu')
    # Optimizer
    scheduler = MultiStepDecay(opt.lr, milestones=opt.milestone, gamma=0.2)  # learning rates
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=scheduler)

    initial_epoch = 0
    for epoch in range(initial_epoch, opt.epochs):
        train_one_epoch_paddle(model,loader_train,criterion,optimizer,scheduler,epoch,print_freq=100)
        if epoch % opt.save_freq == 0 and paddle.distributed.get_rank() <= 0:
            paddle.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pdparams' % (epoch+1)))

if __name__ == "__main__":
    #prepare_data_RainTrainH(data_path=opt.data_path, patch_size=100, stride=80)
    main()