from tensorflow.keras.models import load_model
from data import *
import metricsandloss as metricsandloss
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def conf_img(gr_t,pred):
    tp = (pred*gr_t)*2
    x = pred-gr_t
    x = x+tp
    return x
    
def load_models_ev(path,custom={}):
    models = []
    for i in os.listdir(path):
        path = os.path.join(path,i)
        model = load_model(path,custom_objects=custom)
        models.append(model)
    return models

def evaluate(models,gen,steps):
    ev = []
    for model in models:
        ev.append(model.evaluate(gen,steps))
    return ev

def ev_metrics_img(gr_t,pred):
    return [metricsandloss.dice_coef(gr_t,pred),metricsandloss.jacard_coef(gr_t,pred),metricsandloss.precision(gr_t,pred),metricsandloss.recall(gr_t,pred)]

def make_evaluation_table(arch_names,metrics,metrics_names,csv_save_dir):
    df = pd.DataFrame(data=metrics,index=arch_names,columns=metrics_names)
    df.to_csv(csv_save_dir)

def make_img_to_pap(path_models,path_gen,save_dir='',custom={}):

    models = load_models_ev(path_models,custom=custom)
    test_img,test_masks = read_train_flow(path_gen)
    test_gen = make_generator_flow(val_img,val_masks,batch_size=1)
    columns = 2+len(models)
    rows = len(os.listdir(path_gen+'/images'))
    fig = plt.figure(figsize=(rows*6, columns*6))
    l = 1
    for i in range(rows):
        img,msk = test_gen.__next__()
        pred_img = []
        for j in range(0,1):
            for model in models:
                image = img[j]
                mask = np.argmax(msk[j], axis=2)
                pred = model.predict(img) 
                pred = np.argmax(pred[j],axis=2)
                im = conf_img(mask,pred)
                pred_img.append(im)
        fig.add_subplot(rows,columns,l)
        plt.axis(False)
        plt.imshow(image)
        fig.add_subplot(rows,columns,l+1)
        plt.axis(False)
        plt.imshow(mask, cmap='gray')
        for j in range(0,len(models)):
            fig.add_subplot(rows,columns,l+2+j)
            plt.axis(False)
            plt.imshow(pred_img[j],cmap='RdYlGn')
        l = l+columns
        fig.tight_layout()

    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    plt.savefig(save_dir,dpi=1000)
    plt.show()



    