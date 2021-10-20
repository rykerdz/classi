import argparse
import modelStuff
import json
from torch import nn
import image_process as ip




def msg(name=None):
    return '''
    Basic usage: python predict.py img checkpoint


'''
def arg_parse():
    parser = argparse.ArgumentParser(description='predict using a model', usage=msg())
    parser.add_argument('img_path', help='Path of the image')
    parser.add_argument('checkpoint_path', help='Path of the model chechpoint')
    parser.add_argument('--topk', action='store', help='return tok k most likely classes .default=3', type=int, default=3)
    parser.add_argument('--category_names', action='store', help='Categories to real names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', help='Use GPU for training? default=false', default='false')

    return parser.parse_args()

def main():
    args = arg_parse()
    
    img_path = args.img_path
    checkpoint = args.checkpoint_path
    topk = args.topk
    cat_to_names = args.category_names
    device = 'cpu' if args.gpu == 'false' else 'cuda'
    
    
    criterion = nn.NLLLoss()
    train_dataset, trainloader, testloader, validloader = ip.load_data('flowers')
    #modelStuff.validate_model(model, testloader, criterion, device)
    
    
    
    top_p, top_class = modelStuff.predict(img_path, checkpoint, cat_to_names, topk, device)
    
    i=0
    print("======== PREDECTIONS RESULTS ========")
    while i < topk:
        print(f'{i+1}. {top_class[i]} with propability of {top_p[i]}')
        i+=1
        
    print("END OF PREDICTIONS!")
        
    
    
    
    
    





if __name__ == '__main__':
    main()