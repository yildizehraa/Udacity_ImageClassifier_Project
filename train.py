import argparse
import torch
from collections import OrderedDict
from os.path import isdir
import os.path 
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def arg_parser():
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    parser.add_argument('--data_dir', action="store", nargs='*', default="./flowers/")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    parser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
    args = parser.parse_args()
    
  
    return args



def loadData(data = "./flowers"):
    
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
    train_data_size = len(image_datasets['train'])
    validation_data_size = len(image_datasets['valid'])
    test_data_size =  len(image_datasets['test'])

    return trainLoader, validationLoader, testLoader, image_datasets, train_data_size, validation_data_size



def setModel(lr, hiddenUnits, epochs, architecture):
    if architecture == 'vgg16':
        architecture="vgg16"
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        
        for param in model.parameters():
            param.requires_grad = False 
        
          
        classifier = nn.Sequential(OrderedDict([
            ('inputs', nn.Linear(25088, 120)),
            ('rl1', nn.ReLU()),
            ('dropout', nn.Dropout(0.5)),
            ('hiddenLayer_1', nn.Linear(hiddenUnits, 90)),
            ('rl2', nn.ReLU()),
            ('hiddenLayer_2', nn.Linear(90,70)),
            ('rl3', nn.ReLU()),
            ('hiddenLayer_3', nn.Linear(70, 102)),
            ('output', nn.LogSoftmax(dim=1))]))

        model.classifier = classifier
        
    elif architecture == 'Densenet':
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, 512)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout()),
            ('fc2', nn.Linear(512, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier
        
    else:
        raise Exception("Architecture not accepted")
        
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    model.cuda()

    return model, criterion, optimizer


def trainModel(model, criterion, optimizer, epochs, trainLoader, power, validationLoader, train_data_size, validation_data_size):
    model.cuda()
    for epoch in range(epochs):
        model.train()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        trainLoss  = 0.0
        trainAcc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        for i, (inputs, labels) in enumerate(trainLoader):
            model.eval()
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
        
            optimizer.zero_grad()
        
            outputs = model.forward(inputs)
    
            loss = criterion(outputs, labels)
            loss.backward()
        
            optimizer.step()
        
            trainLoss += loss.item() * inputs.size(0)
                
            #accuracy
            ret, ps = torch.max(outputs.data, 1)
            total = ps.eq(labels.data.view_as(ps))
        
            # compute mean
            accuracy = torch.mean(total.type(torch.FloatTensor))
        
            trainAcc += accuracy.item() * inputs.size(0)
               
            print("Iteration : {:03d}, Loss on trainig: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), accuracy.item()))
    
        with torch.no_grad():
            model.eval()
            for j, (inputs, labels) in enumerate(validationLoader):
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_prediction_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_prediction_counts.type(torch.FloatTensor))
                valid_acc +=acc.item() * inputs.size(0)

            avg_train_loss = trainLoss/train_data_size
            avg_train_acc = trainAcc/train_data_size

            avg_valid_loss = valid_loss/validation_data_size
            avg_valid_acc = valid_acc/validation_data_size

            print(f'Epoch : {epoch}, Training: Loss: f{avg_train_loss}, Accuracy: {avg_train_acc*100}%, \n\t\tValidation : Loss : {avg_valid_loss}, Accuracy: { avg_valid_acc*100}%')
    
    correct,total = 0,0
    model.eval()
  #  model.to('cuda')
    with torch.no_grad():        
        for data in trainLoader:    
            img, lbl = data
            img, lbl = img.to('cuda'), lbl.to('cuda')
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)
            total += lbl.size(0)
            correct += (predicted == lbl).sum().item()        
     
    print('Accuracy is: %d%%' % (100 * correct / total))
    
    return model
    

def saveCheckpoint(tModel, path, structure , hiddenLayer_1, dropout, lr, epochs, image_datasets):
    hiddenLayer_1=120
    structure ='vgg16'
    path='checkpoint.pth'
    epochs=1
    lr=0.001
    dropout=0.5
    tModel.class_to_idx = image_datasets['train'].class_to_idx
    tModel.cpu
    torch.save ({'structure' : structure,
              'hiddenLayer_1': hiddenLayer_1,
              'state_dict': tModel.state_dict(),
              'class_to_idx': tModel.class_to_idx},
               path)
     
            
            
def main():
     
        
    args = arg_parser()
    
    trainLoad, validLoad, testLoad, image_datasets, trainSize, validSize = loadData(args.data_dir)
   
    
    model, criterion, optimizer = setModel(args.learning_rate, args.hidden_units, args.epochs, architecture=args.arch)
       
    
    print(' *** Training Started *** ')
    
    trainedModel = trainModel(model, criterion, optimizer, args.epochs, trainLoad, args.gpu, validLoad, trainSize, validSize)
    
    print(' *** Training Ended *** ')
    
   
    saveCheckpoint(trainedModel, args.save_dir, args.arch, args.hidden_units, args.dropout, args.learning_rate, args.epochs, image_datasets)

    print(' Dataset trained and saved to the checkpoint ! ') 
    
if __name__ == '__main__': main()