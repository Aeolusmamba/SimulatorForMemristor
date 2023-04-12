import argparse
from datetime import datetime
import torch.cuda
import torch.optim as optim
from Model import MyModel
from Dataset.dataloader import get_mnist
import time
import torch.nn as nn
from torch.autograd import Variable
import os
import numpy as np


parser = argparse.ArgumentParser(description="PyTorch for VGG-8 on MNIST example")
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, help='the number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=1, help='learning rate')
parser.add_argument('--optim', default='SGD', help='optimization method')
parser.add_argument('--desc_lr_epoch', default='200,250', help='how many epochs to descend lr')
parser.add_argument('--desc_lr_scale', type=int, default=8, help='descending lr scale')
parser.add_argument('--log_batch_interval', type=int, default=100, help='how many batches to wait before logging training status')
parser.add_argument('--test_epoch_interval', type=int, default=1, help='how many epochs to wait before another test')
parser.add_argument('--log_dir', default='log', help='which directory to save the log information')
parser.add_argument('--weight_precision', type=int, default=5, help='weight precision')
parser.add_argument('--activation_precision', type=int, default=8, help='activation precision')
parser.add_argument('--error_precision', type=int, default=8, help='error precision')
parser.add_argument('--grad_precision', type=int, default=5, help='gradient precision')
parser.add_argument('--cellBit', default=5, help='cell precision')
parser.add_argument('--nonlinearityLTP', default=0.01, help='nonlinearity in LTP')
parser.add_argument('--nonlinearityLTD', default=-0.01, help='nonlinearity in LDP')
parser.add_argument('--max_level', default=32, help='Maximum number of conductance states during weight update (floor(log2(max_level))=cellBit)')
parser.add_argument('--d2dVari', default=0.0, help='device-to-device variation')
parser.add_argument('--c2cVari', default=0.003, help='cycle-to-cycle variation')
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
beta = 0.9  # for momentum


if __name__ == '__main__':
    out_record = open("PythonWrapper_Output.csv", 'ab')
    out_firstline = np.array([["epoch", "average loss", "accuracy"]])
    np.savetxt(out_record, out_firstline, delimiter=",", fmt='%s')

    delta_distribution = open("delta_dist.csv", 'ab')
    delta_firstline = np.array([["1_mean", "2_mean", "3_mean", "4_mean", "5_mean", "6_mean", "7_mean", "8_mean",
                                 "1_std", "2_std", "3_std", "4_std", "5_std", "6_std", "7_std", "8_std"]])
    np.savetxt(delta_distribution, delta_firstline, delimiter=",", fmt='%s')

    weight_distribution = open("weight_dist.csv", 'ab')
    weight_firstline = np.array([["1_mean", "2_mean", "3_mean", "4_mean", "5_mean", "6_mean", "7_mean", "8_mean",
                                  "1_std", "2_std", "3_std", "4_std", "5_std", "6_std", "7_std", "8_std"]])
    np.savetxt(weight_distribution, weight_firstline, delimiter=",", fmt='%s')


    # load model
    model = MyModel(file_path="NetWork.csv", args=args, num_classes=10)
    # load dataset
    train_loader, test_loader = get_mnist(args.batch_size)
    if args.cuda:
        model.cuda()

    optimizer = None
    assert args.optim in ['SGD', 'Adam', 'RMSProp'], args.optim
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

    desc_lr_epoch = list(map(int, args.desc_lr_epoch.split(',')))
    print("descending lr epoch: " + str(desc_lr_epoch))
    best_acc, old_file = 0, None
    lr = args.lr
    begin_t = time.time()

    try:
        for epoch in range(args.epochs):
            print("epoch: " + str(epoch))
            model.train()  # turn on the training 'mode'
            velocity = {}
            i = 0
            for layer in list(model.parameters())[::-1]:
                velocity[i] = torch.zeros_like(layer)
                i = i + 1

            if epoch in desc_lr_epoch:
                lr = lr / args.desc_lr_scale

            print("training phase begin")
            for batch_idx, (data, target) in enumerate(train_loader):
                target_clone = target.clone()
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = model(data)
                # print("type", type(output))
                criterion = nn.MSELoss()
                label = torch.zeros_like(output)
                label[torch.arange(label.size(0)).long(), target] = 1
                loss = criterion(output, label)

                loss.backward()

                j=0
                # introduce non-ideal property
                for name, param in list(model.named_parameters())[::-1]:
                    velocity[j] = beta * velocity[j] + (1 - beta) * velocity[j]  # momentum
                    param.grad.data = velocity[j]
                    j = j + 1
                optimizer.step()

                if batch_idx % args.log_batch_interval == 0 and batch_idx > 0:
                    pred = output.data.max(1)[1]  # get the index of the max log-probability
                    # print("output.data.max(1): ", output.data.max(1))
                    correct = pred.cpu().eq(target_clone).sum()
                    acc = float(correct) * 1.0 / len(data)
                    print('Train epoch: {} ({}/{}); Loss: {:.6f}; Acc: {:.4f}; lr: {:.2e}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        loss.data, acc, optimizer.param_groups[0]['lr']))
            elapse_time = time.time() - begin_t
            speed_epoch = elapse_time / (epoch + 1)
            speed_batch = speed_epoch / len(train_loader)
            eta = speed_epoch * args.epochs - elapse_time  # estimated time of arrival
            print("Elapsed {:.2f} s, {:.2f} s/epoch, {:.2f} s/batch, eta {:.2f} s".format(
                elapse_time, speed_epoch, speed_batch, eta
            ))

            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'latest.pth'))

            delta_std = np.array([])
            delta_mean = np.array([])
            w_std = np.array([])
            w_mean = np.array([])
            oldWeight = {}
            k = 0

            for name, param in list(model.named_parameters()):
                oldWeight[k] = param.data + param.grad.data
                k = k + 1
                delta_std = np.append(delta_std, (torch.std(param.grad.data)).cpu().data.numpy())
                delta_mean = np.append(delta_mean, (torch.mean(param.grad.data)).cpu().data.numpy())
                w_std = np.append(w_std, (torch.std(param.data)).cpu().data.numpy())
                w_mean = np.append(w_mean, (torch.mean(param.data)).cpu().data.numpy())

            delta_mean = np.append(delta_mean, delta_std, axis=0)
            np.savetxt(delta_distribution, [delta_mean], delimiter=",", fmt='%f')
            w_mean = np.append(w_mean, w_std, axis=0)
            np.savetxt(weight_distribution, [w_mean], delimiter=",", fmt='%f')

            print("weight distribution")
            print(w_mean)
            print("delta distribution")
            print(delta_mean)

            if epoch % args.test_epoch_interval == 0:
                model.eval()  # turn on the evaluation mode
                test_loss = 0
                correct = 0
                print("testing phase begin")
                for i, (data, target) in enumerate(test_loader):
                    target_clone = target.clone()
                    if args.cuda:
                        data, target = data.cuda(), target.cuda()
                    with torch.no_grad():
                        data, target = Variable(data), Variable(target)
                        output = model(data)
                        test_criterion = nn.MSELoss()
                        label = torch.zeros_like(output)
                        label[torch.arange(label.size(0)).long(), target] = 1
                        test_loss_i = test_criterion(output, label)
                        test_loss += test_loss_i.cpu().data
                        pred = output.data.max(1)[1]
                        correct += pred.cpu().eq(target_clone).sum()
                test_loss = test_loss / len(test_loader)  # average loss over the number of batch
                acc = 100. * correct / len(test_loader.dataset)
                print('Epoch {}; test set: average loss: {:.4f}, accuracy: {:.0f}% ({}/{})'.format(
                    epoch, test_loss, acc, correct, len(test_loader.dataset)
                ))

                np.savetxt(out_record, [[epoch, test_loss, acc]], delimiter=',', fmt='%f')

                if acc > best_acc:
                    inference_file = os.path.join(os.path.dirname(__file__), 'best-{}.pth'.format(epoch))
                    if old_file and os.path.exists(old_file):
                        os.remove(old_file)
                    torch.save(model.state_dict(), inference_file)
                    best_acc = acc
                    old_file = inference_file


    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        print("Total elapse: {:.2f}, best result: {:.3f}%".format(time.time() - begin_t, best_acc))
