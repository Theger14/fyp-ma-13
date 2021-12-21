#!/usr/bin/env python
# coding: utf-8

# ### This script contains:
# - gen_metrics() : Returns classification report and confusion matrix (sklearn.metrics)
# - gen_save_cr_cm() : Generates, saves and returns classification reports and confusion matrix
# - make_pred() : Returns predicted class and confidence for a single image

# In[1]:


from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import imp
import logging

from subprocess import call         # To call mask filter function
logger = logging.getLogger(__name__)

# In[2]:


import os.path
from os import path


# In[3]:

print(os.getcwd())
os.chdir(os.getcwd()+'/preprocessing')
print("changed:", os.getcwd())

BATCH_SIZE = 32
EPOCHS = 500
IMG_SIZE = (224,224)
LABELS = ["female", "male"]


# In[4]:


def gen_metrics(model_type, all_models, original_fp, test_pert, gender=None):
    """
    Returns classification report and confusion matrix (sklearn.metrics)
    
    model_type : str
        Either 'mobile' (MobileNet), 'dense' (DenseNet) or 'res' (ResNet50)
    all_models : list
        List of models i.e. [mobilenet, densenet, resnet]
    original_fp : str
        Original image file path
    test_pert: str
        Perturbation type. Either 'ori', 'masked', 'glasses', 'make_up' or 'all'
    gender : str
        Gender. Either None, 'male' or 'female' to specify the gender. If None it make predictions on both.
    """
    # FEMALE => 0
    # MALE => 1
    
    # Set model
    if (model_type == "mobile"):
        model = all_models[0]
    elif (model_type == "dense"):
        model = all_models[1]
    elif (model_type == "res"):
        model = all_models[2]
    else:
        raise Exception("Sorry, model_type allowed are 'mobile' (MobileNet), 'dense' (DenseNet)         or 'res' (ResNet50)")
    assert gender in [None, 'male', 'female'], "gender needs to be None, 'male' or 'female'"
#     print(f"Model:{model.summary()}")
    
    datasets = ["test", "test_masked", "test_glasses", "test_makeup"]
    # If we only want ont type of perturbation
    if test_pert != 'all':
        assert test_pert in ['ori', 'masked', 'glasses', 'makeup']
        if test_pert == 'ori':
            datasets = ["test"]
        else:
            datasets = ["test_"+test_pert]
    print(f"Testing on {test_pert}")
    for i in tqdm(range(len(datasets)), 'Testing...'):
        data_name = datasets[i]
        y_true = []
        y_pred = []
        male_dir = os.listdir(original_fp + data_name + "/male")
        female_dir = os.listdir(original_fp + data_name + "/female")
        print(f"male test dataset: {original_fp + data_name}/male")
        print(f"female test dataset: {original_fp + data_name}/female")
        preprocessing_fp = f"/home/monash/Desktop/fyp-work/fyp-ma-13/fyp-models/{original_fp}/"
        if gender is None:
            male_correct, male_tot = 0, 0
            female_correct, female_tot = 0, 0
            for j in range(len(male_dir)):
                fn = male_dir[j]
                p = original_fp + data_name + "/male/" + fn
                img = Image.open(p)
                img = img.resize((224, 224))
                img = np.array(img)
                img = np.expand_dims(img, 0)
                y_true.append(1)
                pred = 1 if model.predict(img) > 0.5 else 0
                y_pred.append(pred)
                male_correct+=pred
                male_tot += 1
                print(f"Male accuracy:{male_correct/male_tot}", end='\r')
#                 print(f"gen_metrics - p:{pred}")
            print("")
            for k in range(len(female_dir)):
                fn = female_dir[k]
                p = original_fp + data_name + "/female/" + fn
                img = Image.open(p)
                img = img.resize((224, 224))
                img = np.array(img)
                img = np.expand_dims(img, 0)
                y_true.append(0)
                pred = 1 if model.predict(img) > 0.5 else 0
                y_pred.append(pred)
                if pred == 0:
                    female_correct+=1
                female_tot += 1
                print(f"Female accuracy:{female_correct/female_tot}", end='\r')
            print()
#                 print(f"gen_metrics - female p:{pred}")
        elif gender == 'male':
            for j in range(len(male_dir)):
                fn = male_dir[j]
                p = original_fp + data_name + "/male/" + fn
                img = Image.open(p)
                img = img.resize((224, 224))
                img = np.array(img)
                img = np.expand_dims(img, 0)

                y_true.append(1)
                pred = 1 if model.predict(img) > 0.5 else 0
                y_pred.append(pred)
#                 print(f"gen_metrics - p:{pred}")
        elif gender == 'female':
            for k in range(len(female_dir)):
                fn = female_dir[k]
                p = original_fp + data_name + "/female/" + fn
                img = Image.open(p)
                img = img.resize((224, 224))
                img = np.array(img)
                img = np.expand_dims(img, 0)
                y_true.append(0)
                pred = 1 if model.predict(img) > 0.5 else 0
                y_pred.append(pred)
#                 print(f"gen_metrics - female p:{pred}")

        cr = classification_report(y_true, y_pred, zero_division = 1)
        cm = confusion_matrix(y_true, y_pred)
    return cr, cm

def gen_save_cr_cm(model_type, all_models, original_fp, target_fp, test_pert='ori', gender=None):
    """
    Generates, saves and returns classification reports and confusion matrix
    
    model_type : str
        Either 'mobile' (MobileNet), 'dense' (DenseNet) or 'res' (ResNet50)
    all_models : list
        List of models i.e. [mobilenet, densenet, resnet]
    original_fp : str
        Original image file path
    target_fp : str
        Target file path to save results
    test_pert: str
        Either 'ori', 'masked', 'glasses', 'make_up' or 'all'
    gender : str
        Either None, 'male' or female to specify the gender. If None it make predictions on both.
    """
    assert model_type in ['mobile', 'dense', 'res'], 'Incorrect model_type param value'
    assert gender in [None, 'male', 'female'], 'Incorrect gender param value'
    
    # Assign to appropriate folder
    if test_pert != 'all':
        assert test_pert in ['ori', 'masked', 'glasses', 'makeup']
    
    temp = gender
    if temp is None: # Checks if it is for all genders
        temp = 'bothg'
    x = f'{target_fp}cr_cm_{model_type}_{test_pert}_{temp}'
    if path.exists(x):    # if it already exists
        print(x + " already exists, pass")
        return None, None
    else:
        print("Creating " + x +"...")
    cr, cm = gen_metrics(model_type, all_models, original_fp, test_pert, gender=gender)
    
    # If we only want one type of perturbation
    if gender == None:
        gender = 'bothg'
    # Dumps metrics into a JSON object
    res = {"cr_{}_{}".format(model_type, gender): cr, 
           "cm_{}_{}".format(model_type, gender): cm.tolist()}
    j = json.dumps(res, indent = 4)
    
    # Save as JSON object
    fn = Path(f'{target_fp}/cr_cm_{model_type}_{test_pert}_{gender}')
    if not fn.is_dir():
        with open(f'{target_fp}/cr_cm_{model_type}_{test_pert}_{gender}', 'w') as outfile:
            json.dump(j, outfile)
    return cr, cm


# # Make individual predictions

# In[21]:


# Load in glasses filter module
glasses_mod = imp.load_source('apply_glasses', '/home/monash/Desktop/fyp-work/fyp-ma-13/fyp-models/preprocessing/perturb_filters/glasses/put_glasses.py')
makeup_mod = imp.load_source('apply_makeup', '/home/monash/Desktop/fyp-work/fyp-ma-13/fyp-models/preprocessing/gen_perturbed_test_sets.py')


# In[22]:


# Perturbation functions
def apply_filter(original_fp, target_fp, filter_type):
    """
    Applies a desired filter to specific image
    
    original_fp : str
        File path containing image
    target_fp : str
        Target path to folder to store image
    filter_type : str
        Type of filter to apply on the image
    """
    spl = original_fp.split("/")
    
    try:
        if filter_type == "glasses":
            # Call apply_glasses from the glasses module
            glasses_mod.apply_glasses('/'.join(spl[:-1]), spl[-1], target_fp)
        elif filter_type == "makeup":
            makeup_mod.apply_makeup('/'.join(spl[:-1]), spl[-1], target_fp)
        elif filter_type == "mask":
            print("Original fp:", original_fp)
            status = call("python mask_the_face.py --path {} --mask_type 'N95' --verbose".format(original_fp),
                    cwd="/home/monash/Desktop/fyp-work/fyp-ma-13/fyp-models/preprocessing/perturb_filters/mask", 
                    shell=True)
            
            # Move saved image to target_fp
            mask_image_fp = '_N95.'.join(spl[-1].split("."))
            os.rename('/'.join(spl[:-1]) + "/" + mask_image_fp, target_fp + spl[-1])
            print("Moved image to temp folder", status)
        else: raise Exception("filter_type is not accepted")
        print("Success!")
    except Exception as e:
        print("Please try Again.")
        print(e)


# In[3]:


# mobilenet = tf.keras.models.load_model('model_tl_best_weights_mobile.h5')
# densenet = tf.keras.models.load_model('model_tl_best_weights_dense.h5')
# resnet = tf.keras.models.load_model('model_tl_best_weights_res.h5')
# all_models = [mobilenet, densenet, resnet]


# In[13]:


def make_pred(image_fn, model_type, debiased=False, pt=None):
    """
    Returns predicted class and confidence for a single image
    
    image_fn : str
        Path to image
    model_type : str
        Either 'mobile' (MobileNet), 'dense' (DenseNet) or 'res' (ResNet50)
    debiased : boolean
        Determines whether the debiased model should be used.
    pt : str
        Perturbation type (default = None)
            'g' - glasses filter, 'mu' - makeup filter, 'msk' - mask filter
    """
    model_path = '/home/monash/Desktop/fyp-work/fyp-ma-13/fyp-models/timeline/{}/best_weights/set10/model_tl_best_weights_{}_set10.h5'
    # Set model
    if (model_type == "mobile") or (model_type == "dense") or (model_type == "res"):
        if not debiased:
            model = tf.keras.models.load_model(model_path.format("(8)_debiased_25", model_type))
        else:
            model = tf.keras.models.load_model(model_path.format("(5)_early_stopping_20", model_type))
    else:
        raise Exception("Sorry, model_type allowed are 'mobile' (MobileNet), 'dense' (DenseNet)         or 'res' (ResNet50)")
        
    print(model_type, "loaded")
    
    if pt is None:
        # For unperturbed
        img = Image.open(image_fn)
    elif pt == 'g':
        # For glasses
        apply_glasses(image_fn)
        img = Image.open(image_fn+{}).format(pt)
    elif pt == 'mu':
        # For makeup
        apply_makeup(image_fn)
        img = Image.open(image_fn+{}).format(pt)
    elif pt == 'msk':
        # For masked
        apply_mask(image_fn)
        img = Image.open(image_fn+{}).format(pt)
    
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    
    confidence = model.predict(img)
    res = [1 if confidence > 0.5 else 0][0]
    
    if pt is not None:
        pass
    
    if res == 1:
        return ("Male", confidence)
    elif res == 0:
        return ("Female", 1 - confidence)
    else:
        raise Exception("Issue during prediction occured")

