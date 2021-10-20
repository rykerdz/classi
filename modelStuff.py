import torch
from torchvision import models
from torch import nn, optim
from collections import OrderedDict
import torch.nn.functional as F
import image_process as ip
import json

def setup_model(structure='densenet161', hidden_units=1024, lr=0.003, device='cpu'):
    if structure=='densenet161':
        model = models.densenet161(pretrained=True)
        model.name = 'densenet161'
        model.hidden_units = hidden_units
    elif structure=='vgg19':
        model = models.vgg19(pretrained=True)
        model.name = 'vgg19'
        model.hidden_units = hidden_units
    else:
        print('This model isn\'t supported Please choose either densenet161 or vgg19')
        return None
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(2208, hidden_units)),
                                       ('relu1', nn.ReLU()),
                                       ('dropout1', nn.Dropout(p=0.2)),
                                       ('fc2', nn.Linear(hidden_units, 512)),
                                       ('relu2', nn.ReLU()),
                                       ('dropout2', nn.Dropout(p=0.5)),
                                       ('fc3', nn.Linear(512, 102)),
                                       ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    if torch.cuda.is_available() and device == 'cuda':
        model.to(device)
    
    return model, optimizer, criterion


def train(model, optimizer, criterion, trainloader, validloader,train_dataset, device='cpu', save_dir='checkpoint.pth', epochs=20):
    steps = 0
    running_loss = 0
    print_every = 20
    print('Training has Started!.....')
    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1
            if device=='cuda':
                images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validloader:
                        if device=='cuda':
                            images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    model.class_to_idx = train_dataset.class_to_idx
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    torch.save({'structure': model.name,
                'hidden_units': model.hidden_units,
                'lr': lr,
                'model_sd': model.state_dict(),
                'class2idx': model.class_to_idx}, save_dir)
  
    print(f"\n\n Finished Training. checkpoint saved to: {save_dir}.")
    
    

    

def load_checkpoint(path):
    
    checkpoint = torch.load(path)
    structure = checkpoint['structure'].lower()
    hidden_units = checkpoint['hidden_units']
    lr=checkpoint['lr']

    model,_,_ = setup_model(structure, hidden_units, lr)

    model.class_to_idx = checkpoint['class2idx']
    model.load_state_dict(checkpoint['model_sd'])
    
    return model

def predict(img_path, model_path, cat_to_names, topk=3, device='cpu'):
    
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = ip.process_img(img_path)
    model = load_checkpoint(model_path)
    with open(cat_to_names, 'r') as json_file:
        cat_to_name = json.load(json_file)
    
    
    with torch.no_grad():
        image = image.to(dtype=torch.float32)
        image.unsqueeze_(0)
        logps = model.forward(image)
        ps = torch.exp(logps)

        # getting top_p and top classes
        top_p, top_class = ps.topk(topk, dim=1)
        # converting to list
        top_p = top_p.detach().numpy().tolist()[0] 
        top_class = top_class.detach().numpy().tolist()[0]

        # indice to class
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_classes = [cat_to_name[idx_to_class[x]] for x in top_class]
    

    
    return top_p, top_classes

def validate_model(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    model.eval()
    model.to(device)
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            logps = model.forward(images)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        print(f"Test accuracy: {accuracy/len(testloader):.3f}")

    
                          
    