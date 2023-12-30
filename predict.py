import argparse
import ic_model as ic

import json
import numpy as np
import torch
from PIL import Image
from config import *

def predicts(args):
    print("Loading the checkpoint model")
    model = ic.load_model(args.checkpoint_path, 'alexnet', args.hidden_units)
    print("Loading the checkpoint model is finished")
    model.to(device)

    probs, classes = predict(args.input, model, args.top_k)
    print("Most "+str(args.top_k)+" Classes with their probabilities" )

    if(args.category_names):
        flower_to_name=[]
        with open(args.category_names, 'r') as f:
            flower_to_name = json.load(f)

        classes_label=[]
        print(probs.shape)
        for i in classes:
            classes_label.append(flower_to_name.get(str(i)))
        for i in range(args.top_k):
            print ("Class name: ",classes_label[i],", Probability :" ,probs[0][i].detach().numpy()*100)
    else:
        for i in range(args.top_k):
            print("Class #", classes[i]," Probability: ",probs[0][i].detach().numpy()*100)

   

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    mean = MEAN
    std = STD
    size = SIZE
    
    image.thumbnail(size, Image.LANCZOS)
    image = image.crop((16,16,224+16,224+16))
    np_image = np.array(image)/255
    
    for i in [0,1,2]:
        np_image[:,:,i]=(np_image[:,:,i]-mean[i])/std[i]
    final_image = np.transpose(np_image, (2,0,1))

    return final_image

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    indexClass={}
    for i,value in model.class_to_idx.items():
        indexClass[value]=i

    image=process_image(Image.open(image_path))
    image=torch.FloatTensor([image])
    model.eval()
    output=model(image.to(device))
    prob=torch.exp(output.cpu())

    top_p,top_c = prob.topk(topk,dim=1)
    top_class = [indexClass.get(x) for x in top_c.numpy()[0]]
    
    return top_p,top_class


def main():
    parser = argparse.ArgumentParser(description='Flowers Classifcation Predictor')
    parser.add_argument('--input', type=str, default='flower_data/test/10/image_07090.jpg', help='path for image to predict')
    parser.add_argument('--checkpoint_path', type=str, default='assets/' + MODEL_FILE, help='path to en existance  checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='top k classes for the input')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='json path file of categories names of flowers')
    parser.add_argument('--gpu' , type=bool, default=False, help='checkpoint directory path')
    parser.add_argument('--hidden_units', type=int, default=512, help='hidden units for  fc layer')

    args = parser.parse_args()
    print(args)
    
    predicts(args)
    print("\nPrediction is finished\n")
if __name__ == "__main__":
    main()