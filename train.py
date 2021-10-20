import argparse
import image_process as ip
import modelStuff
import torch
from torchvision import models
from torch import nn, optim
from collections import OrderedDict
import torch.nn.functional as F

def msg(name=None):
    return '''
    Basic usage: python train.py data_directory


'''
def arg_parse():
    parser = argparse.ArgumentParser(description='Train a model using a dataset', usage=msg())
    parser.add_argument('data_directory', help='Path of the Dataset directory')
    parser.add_argument('--save_dir', action='store', help='Specify where to save the model checkpoint after training. default="/checkpoints"',default='checkpoints/checkpoint.pth')
    parser.add_argument('--arch', action='store', help='Choose architecture (vgg19, default=densenet161)', default='densenet161')
    parser.add_argument('--learning_rate', action='store', help='Set learning rate. default=0.003', type=float, default=0.003)
    parser.add_argument('--hidden_units', action='store', help='Set hidden units. default=1024', type=int, default=1024)
    parser.add_argument('--epochs', action='store', help='Set epochs. default=20', type=int, default=20)
    parser.add_argument('--gpu', action='store', help='Use GPU for training? default=false', default='false')
    return parser.parse_args()

def main():
    
    args = arg_parse()
    
    data_dir = args.data_directory
    save_dir = args.save_dir
    structure = args.arch
    lr = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    device = 'cuda' if args.gpu =='true' else 'cpu'
    
    # loading the data
    train_dataset, trainloader, testloader, validloader = ip.load_data(data_dir)
    
    model, optimizer, criterion = modelStuff.setup_model(structure, hidden_units, lr, device)
    
    modelStuff.train(model, optimizer, criterion, trainloader, validloader,train_dataset, device, save_dir, epochs)
    
if __name__ == '__main__':
    main()
    
    


