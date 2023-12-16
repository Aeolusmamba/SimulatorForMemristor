import argparse
from datetime import datetime
import torch.cuda
from Model import Model
from Dataset.dataloader import get_mnist
import time
# import torch.nn as nn
from C_Graph.variable import Variable
import os
import numpy as np
from Loss.mse import MSE
from Loss.cross_entropy import CrossEntropy
from util import wage_quantizer
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="NumPy for VGG-8 on MNIST example")
parser.add_argument('--batch_size', type=int, default=250, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, help='the number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate (won\'t change overtime)')
parser.add_argument('--grad_scale', type=float, default=1, help='learning rate (will change overtime) for wage delta calculation')
parser.add_argument('--dr', type=float, default=0.001, help='decay rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam')
parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon for adam')
parser.add_argument('--desc_lr_epoch', default='2,5', help='how many epochs to descend lr')
parser.add_argument('--desc_lr_scale', type=float, default=8.0, help='descending lr scale')
parser.add_argument('--record_batch_interval', type=int, default=100, help='how many batches to wait before recording training status')
parser.add_argument('--test_epoch_interval', type=int, default=1, help='how many epochs to wait before another test')
parser.add_argument('--record_dir', default='record', help='which directory to save the record information')
parser.add_argument('--weight_precision', type=int, default=5, help='weight precision')
parser.add_argument('--activation_precision', type=int, default=8, help='activation precision')
parser.add_argument('--error_precision', type=int, default=8, help='error precision')
parser.add_argument('--grad_precision', type=int, default=5, help='gradient precision')
parser.add_argument('--cellBit', default=5, help='cell precision (we only support one-cell-per-synapse, i.e. cellBit==wl_weight==wl_grad)')
parser.add_argument('--nonlinearityLTP', default=50, help='nonlinearity in LTP')
parser.add_argument('--nonlinearityLTD', default=-50, help='nonlinearity in LDP')
parser.add_argument('--max_level', default=32, help='Maximum number of conductance states during weight update (floor(log2(max_level))=cellBit)')
parser.add_argument('--d2dVari', default=0.0, help='device-to-device variation')
parser.add_argument('--c2cVari', default=0.003, help='cycle-to-cycle variation')
parser.add_argument('--loss', default='CrossEntropy', help='choose the loss function (MSE or CrossEntropy')
parser.add_argument('--wl_weight', type = int, default=5, help='weight precision')
parser.add_argument('--wl_grad', type = int, default=5, help='gradient precision')
parser.add_argument('--wl_activate', type = int, default=8)
parser.add_argument('--wl_error', type = int, default=8)
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
# or use the entire args
hyper_p = {'lr': args.lr, 'grad_scale': args.grad_scale, 'dr': args.dr, 'momentum': args.momentum,
           'beta1': args.beta1, 'beta2': args.beta2, 'epsilon': args.epsilon,
           'nonlinearityLTP': args.nonlinearityLTP, 'nonlinearityLTD': args.nonlinearityLTD, 'c2cVari': args.c2cVari}


if __name__ == '__main__':
    try:
        # load dataset
        train_loader, test_loader = get_mnist(args.batch_size)

        # load model
        model = Model(file_path="NetWork4-LeNet.csv", batch_size=args.batch_size, hyper_p=hyper_p)
        # if args.cuda:
        #     model.cuda()

        desc_lr_epoch = list(map(int, args.desc_lr_epoch.split(',')))
        print("descending lr epoch: " + str(desc_lr_epoch))
        best_acc, old_file = 0, None
        lr = hyper_p['lr']
        grad_scale = hyper_p['grad_scale']
        begin_t = time.time()

        train_loss = []
        test_acc = []
        test_loss = []

        for epoch in range(args.epochs):
            print("======================epoch No." + str(epoch) + "======================")

            if epoch in desc_lr_epoch:
                grad_scale = grad_scale / args.desc_lr_scale

            print("training phase begins")
            for batch_idx, (data, target) in enumerate(train_loader):
                print("----------------------batch No." + str(batch_idx) + "----------------------")
                data, target = data.numpy(), target.numpy()
                # np.set_printoptions(threshold=np.inf)
                # for parameter in model.parameters:
                #     if "conv_0" in parameter.name:
                #         print(f"{parameter.name}: ", parameter.data)
                phase = 'train'
                output = model.forward_propagation(data, phase)
                label = np.zeros(output.shape)
                label[np.arange(label.shape[0]), target] = 1  # convert to [N, M]
                # print("label[np.arange(label.shape[0]): ", label[np.arange(label.shape[0]), target])
                # print(label)
                label_var = Variable(list(label.shape), name="label", scope="train_epoch"+str(epoch)+"_batch"+str(batch_idx), grad=False, learnable=False, init='None')
                label_var.data = label
                assert args.loss in ['MSE', 'CrossEntropy'], args.loss
                if args.loss == 'MSE':
                    loss = MSE([output, label_var], name='MSE_epoch'+str(epoch)+"_batch"+str(batch_idx))
                else:
                    loss = CrossEntropy([output, label_var], name='CrossEntropy_epoch'+str(epoch)+"_batch"+str(batch_idx))

                loss.forward(phase)
                train_loss.append(loss.output_variable.data)
                print("loss: ", loss.output_variable.data)
                print("Back propagation begins")
                loss.backward()
                model.back_propagation()

                print("back propagated")

                # for parameter in model.parameters:
                #     if "conv_0" in parameter.name and "bias" not in parameter.name:
                #         print(f"{parameter.name} grad: ", parameter.diff)
                #     elif "Linear_1" in parameter.name and "bias" not in parameter.name:
                #         print(f"{parameter.name} grad: ", parameter.diff)

                model.update()
                print("updated")

                # for parameter in model.parameters:
                #     if "conv_0" in parameter.name and "bias" not in parameter.name:
                #         print(f"{parameter.name}: ", parameter.data)
                #     if "Linear_1" in parameter.name and "bias" not in parameter.name:
                #         print(f"{parameter.name}: ", parameter.data)

                # j=0
                # for param in model.parameters:
                #     print(f"{j}: param.data: ", param.data)
                #     print(f"{j}: param.diff: ", param.diff)
                #     j = j + 1


                if batch_idx % args.record_batch_interval == 0 and batch_idx > 0:
                    pred = np.argmax(output.data, axis=1)  # get the index of the max log-probability
                    correct = np.sum(pred == target)
                    acc = 100. * float(correct) / len(data)
                    print('Train epoch: {} ({}/{}); Loss: {:.6f}; Acc: {:.4f}%; lr: {:.2e}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        loss.output_variable.data, acc, lr))
            elapse_time = time.time() - begin_t
            speed_epoch = elapse_time / (epoch + 1)
            speed_batch = speed_epoch / len(train_loader)
            eta = speed_epoch * args.epochs - elapse_time  # estimated time of arrival
            print("Elapsed {:.2f} s, {:.2f} s/epoch, {:.2f} s/batch, eta {:.2f} s".format(
                elapse_time, speed_epoch, speed_batch, eta
            ))



            if epoch % args.test_epoch_interval == 0:
                test_loss_epoch = 0
                correct = 0
                print("testing phase begin")
                for batch_idx, (data, target) in enumerate(test_loader):
                    data, target = data.numpy(), target.numpy()
                    phase = 'test'
                    output = model.forward_propagation(data, phase)
                    label = np.zeros(output.shape)
                    label[np.arange(label.shape[0]), target] = 1  # convert to [N, M]
                    label_var = Variable(list(label.shape), name="label",
                                         scope="test_epoch" + str(epoch) + "_batch" + str(batch_idx), grad=False,
                                         learnable=False, init='None')
                    label_var.data = label
                    assert args.loss in ['MSE', 'CrossEntropy'], args.loss
                    if args.loss == 'MSE':
                        loss = MSE([output, label_var], name='test_MSE_epoch'+str(epoch)+"_batch"+str(batch_idx))
                    else:
                        loss = CrossEntropy([output, label_var], name='test_CrossEntropy_epoch'+str(epoch)+"_batch"+str(batch_idx))
                    loss.forward(phase)
                    test_loss_i = loss.output_variable.data
                    test_loss_epoch += test_loss_i
                    pred = np.argmax(output.data, axis=1)
                    correct += np.sum(pred == target)
                test_loss_epoch = test_loss_epoch / len(test_loader)  # average loss over the number of batches
                acc = 100. * correct / len(test_loader.dataset)  # note that len(test_loader) == num of batches; len(test_loader.dataset) == num of test data
                test_acc.append(acc)
                test_loss.append(test_loss_epoch)

                print('Epoch {}; test set: average loss: {:.4f}, accuracy: {:.0f}% ({}/{})'.format(
                    epoch, test_loss_epoch, acc, correct, len(test_loader.dataset)
                ))

                if acc > best_acc:
                    best_acc = acc

        # record data
        with open("./Experiment/LeNet-5_MNIST.txt", 'w') as f:
            f.write("training loss\n")
            for item in train_loss:
                f.write("%s " % item)
            f.write("\ntest loss\n")
            for item in test_loss:
                f.write("%s " % item)
            f.write("\ntest accuracy\n")
            for item in test_acc:
                f.write("%s " % item)

        # draw some graphs
        plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.xlabel('No. of iterations')
        plt.ylabel('Loss')
        plt.plot(range(len(train_loss)), train_loss)
        # plt.legend()
        plt.title('Loss curve')

        plt.subplot(2, 1, 2)
        plt.xlabel('No. of epochs')
        plt.ylabel('Accuracy (%)')
        plt.plot(range(len(test_acc)), test_acc)
        # plt.legend()
        plt.title("Test Accuracy curve")
        plt.tight_layout()
        plt.show()


    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        print("Total elapse: {:.2f}, best result: {:.3f}%".format(time.time() - begin_t, best_acc))
