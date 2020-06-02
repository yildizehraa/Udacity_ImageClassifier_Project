import argparse
import torch
from collections import OrderedDict
from os.path import isdir
import os.path 
from PIL import Image
import numpy as np
import json
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def arg_parser():
    parser = argparse.ArgumentParser(description="Predict.py")
    parser.add_argument('--input_img', default='/home/workspace/ImageClassifier/flowers/test/1/image_06752.jpg', action="store", type = str)
    parser.add_argument('--checkpoint', default='/home/workspace/ImageClassifier/checkpoint.pth', nargs='*', action="store",type = str)
    parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")
    args = parser.parse_args()
    
  
    return args


def loadData():
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
    'train' : transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.454, 0.406],
                            [0.229, 0.224, 0.225])
    ]),
    'valid' : transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.454, 0.406],
                            [0.229, 0.224, 0.225])
    ]),
    'test' : transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.454, 0.406],
                             [0.229, 0.224, 0.225])
    ])
                }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid' : datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test' : datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }


    trainLoader = torch.utils.data.DataLoader(image_datasets['train'], batch_size = 64, shuffle = True)
    validationLoader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size = 32, shuffle = True)
    testLoader = torch.utils.data.DataLoader(image_datasets['test'], batch_size = 32, shuffle = True)        
        
    return trainLoader, validationLoader, testLoader, image_datasets


def loadCheckpoint(path, device):
     
    path='checkpoint.pth'
   # model.cuda()
    
    checkpoint = torch.load('checkpoint.pth')
    structure = checkpoint['structure']
    hiddenLayer_1 = checkpoint['hiddenLayer_1']
    
    
    model,_,_ = setModel(structure)
    model.class_to_idx = checkpoint['class_to_idx']
    model.state_dict(checkpoint['state_dict'])
    print("Loading checkpoint is successfull !")
    
    return model
    
                       
def predict(image_path, model, topK, device, cat_to_Name):  
 #   if torch.cuda.is_available() and device == 'gpu':
 #       model.to('cuda:0')
        
        model.to("cpu")
        image = process_image(image_path)
        image = torch.from_numpy(np.asarray(image).astype('float'))
        image.unsqueeze_(0)
        image = image.float()
  
        outputs = model(image)
        probs, classes = torch.exp(outputs).topk(topK)
              
        return probs[0].tolist(), classes[0].add(1).tolist()

    
def print_probability(flowers, probabilities, index):  
    i=0
    while i < index:
        print("{} with a probability of {}".format(flowers[i], probabilities[i]))
        i += 1    
        
def setModel(architecture):
    architecture="vgg16"
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
        
    for param in model.parameters():
        param.requires_grad = False 
        
          
    classifier = nn.Sequential(OrderedDict([
            ('inputs', nn.Linear(25088, 120)),
            ('rl1', nn.ReLU()),
            ('dropout', nn.Dropout(0.5)),
            ('hiddenLayer_1', nn.Linear(120, 90)),
            ('rl2', nn.ReLU()),
            ('hiddenLayer_2', nn.Linear(90,70)),
            ('rl3', nn.ReLU()),
            ('hiddenLayer_3', nn.Linear(70, 102)),
            ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    model.cuda()

    return model, criterion, optimizer


def process_image(image):
    proccessedImg = Image.open(image)
   
    prepoceess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = prepoceess(proccessedImg)
    
    return img
        
    
def main():
    
    args = arg_parser()

    trainLoad, validLoad, testLoad, image_datasets = loadData()
    
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
   
    model = loadCheckpoint(args.checkpoint, args.gpu)
    
    
    probs, classes = predict(args.input_img, model, args.top_k, args.gpu, cat_to_name)
    
    flowers = [cat_to_name[str(x)]  for x in classes]
    
    
    print_probability(flowers, classes, args.top_k)
                        

if __name__ == '__main__': main()