#!/usr/bin/env python
# coding: utf-8

# ### This script contains:
# - gen_metrics() : Returns classification report and confusion matrix (sklearn.metrics)
# - gen_save_cr_cm() : Generates, saves and returns classification reports and confusion matrix
# - make_pred() : Returns predicted class and confidence for a single image

# In[8]:


from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from pathlib import Path


# In[9]:


import os.path
from os import path


# In[2]:


BATCH_SIZE = 32
EPOCHS = 500
IMG_SIZE = (224,224)
LABELS = ["female", "male"]


# In[4]:


def gen_metrics(model_type, all_models, original_fp, perturbation='all', gender=None):
    """
    Returns classification report and confusion matrix (sklearn.metrics)
    
    model_type : str
        Either 'mobile' (MobileNet), 'dense' (DenseNet) or 'res' (ResNet50)
    all_models : list
        List of models i.e. [mobilenet, densenet, resnet]
    original_fp : str
        Original image file path
    perturbation: str
        Perturbation type. Either 'ori', 'masked', 'glasses', 'make_up' or 'all'
    gender : str
        Gender. Either None, 'male' or female to specify the gender. If None it make predictions on both.
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
    
    datasets = ["test", "test_masked", "test_glasses", "test_makeup"]
    # If we only want ont type of perturbation
    if perturbation != 'all':
        assert perturbation in ['ori', 'masked', 'glasses', 'makeup']
        if perturbation == 'ori':
            datasets = ["test"]
        else:
            datasets = ["test_"+perturbation]
    
    for i in tqdm(range(len(datasets)), 'Testing...'):
        data_name = datasets[i]
        y_true = []
        y_pred = []
        male_dir = os.listdir(original_fp + data_name + "/male")
        female_dir = os.listdir(original_fp + data_name + "/female")
        
        if gender is None:
            for j in range(len(male_dir)):
                fn = male_dir[j]
                img = Image.open(original_fp + data_name + "/male/" + fn)
                img = img.resize((224, 224))
                img = np.array(img)
                img = np.expand_dims(img, 0)

                y_true.append(1)
                y_pred.append(1 if model.predict(img) > 0.5 else 0)

            for k in range(len(female_dir)):
                fn = female_dir[k]
                img = Image.open(preprocessing_fp + data_name + "/female/" + fn)
                img = img.resize((224, 224))
                img = np.array(img)
                img = np.expand_dims(img, 0)

                y_true.append(0)
                y_pred.append(1 if model.predict(img) > 0.5 else 0)
        elif gender == 'male':
            for j in range(len(male_dir)):
                fn = male_dir[j]
                img = Image.open(preprocessing_fp + data_name + "/male/" + fn)
                img = img.resize((224, 224))
                img = np.array(img)
                img = np.expand_dims(img, 0)

                y_true.append(1)
                y_pred.append(1 if model.predict(img) > 0.5 else 0)
        elif gender == 'female':
            for k in range(len(female_dir)):
                fn = female_dir[k]
                img = Image.open(preprocessing_fp + data_name + "/female/" + fn)
                img = img.resize((224, 224))
                img = np.array(img)
                img = np.expand_dims(img, 0)

                y_true.append(0)
                y_pred.append(1 if model.predict(img) > 0.5 else 0)

        cr = classification_report(y_true, y_pred, zero_division = 1)
        cm = confusion_matrix(y_true, y_pred)
    return cr, cm

def gen_save_cr_cm(model_type, all_models, original_fp, target_fp, perturbation='all', gender=None):
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
    perturbation: str
        Either 'ori', 'masked', 'glasses', 'make_up' or 'all'
    gender : str
        Either None, 'male' or female to specify the gender. If None it make predictions on both.
    """
    assert model_type in ['mobile', 'dense', 'res'], 'Incorrect model_type param value'
    assert gender in [None, 'male', 'female'], 'Incorrect gender param value'
    
    # Assign to appropriate folder
    if perturbation != 'all':
        assert perturbation in ['ori', 'masked', 'glasses', 'makeup']
    
    # @TODO: Solve this inefficient checking of gender
    temp = gender
    if temp is None: # Checks if it is for all genders
        temp = 'bothg'
    x = target_fp+'cr_cm_{}_{}_{}'.format(model_type, perturbation, temp)
    if path.exists(x):    # if it already exists
        print(x + " already exists, pass")
        return None, None
    else:
        print("Creating " + x +"...")

    cr, cm = gen_metrics(model_type, all_models, original_fp, perturbation, gender=gender)
    
    # If we only want one type of perturbation
    if gender == None:
        gender = 'bothg'
    # Dumps metrics into a JSON object
    res = {"cr_{}_{}".format(model_type, gender): cr, 
           "cm_{}_{}".format(model_type, gender): cm.tolist()}
    j = json.dumps(res, indent = 4)
    
    # Save as JSON object
    fn = Path(target_fp+'cr_cm_{}_{}_{}'.format(model_type, perturbation, gender))
    if not fn.is_dir():
        with open(target_fp+'cr_cm_{}_{}_{}'.format(model_type, perturbation, gender), 'w') as outfile:
            json.dump(j, outfile)
    return cr, cm


# # Make individual predictions

# In[1]:


import imp


# In[2]:


# Load in glasses filter module
glasses_mod = imp.load_source('apply_glasses', '/home/monash/Desktop/fyp-work/fyp-ma-13/fyp-models/preprocessing/perturb_filters/glasses/put_glasses.py')
makeup_mod = imp.load_source('apply_makeup', '/home/monash/Desktop/fyp-work/fyp-ma-13/fyp-models/preprocessing/gen_perturbed_test_sets.py')


# In[3]:


# Perturbation functions
# @TODO: Implement perturbation functions
# Each function temporarily saves the perturbed images
def apply_mask():
    return
    
def apply_filter(original_fp, target_fp, filter_type):
    spl = original_fp.split("/")
    try:
        if filter_type == "glasses":
            # Call apply_glasses from the glasses module
            glasses_mod.apply_glasses('/'.join(spl[:-1]), spl[-1], target_fp)
        elif filter_type == "makeup":
            makeup_mod.apply_makeup('/'.join(spl[:-1]), spl[-1], target_fp)
        print("Success!")
    except Exception as e:
        print("Please try Again.")
        print(e)

# In[4]:


# apply_glasses("tom.jpg", '')


# In[3]:


# mobilenet = tf.keras.models.load_model('model_tl_best_weights_mobile.h5')
# densenet = tf.keras.models.load_model('model_tl_best_weights_dense.h5')
# resnet = tf.keras.models.load_model('model_tl_best_weights_res.h5')
# all_models = [mobilenet, densenet, resnet]


# In[13]:


def make_pred(image_fn, model_type, pt=None):
    """
    Returns predicted class and confidence for a single image
    image_fn : str
        Path to image
    model_type : str
        Either 'mobile' (MobileNet), 'dense' (DenseNet) or 'res' (ResNet50)
    pt : str
        Perturbation type (default = None)
    """
    # Set model
    if (model_type == "mobile"):
        model = tf.keras.models.load_model('/home/monash/Desktop/fyp-work/fyp-ma-13/fyp-models/timeline/(8)_debiased_25/best_weights/set10/model_tl_best_weights_mobile_set10.h5')
    elif (model_type == "dense"):
        model = tf.keras.models.load_model('/home/monash/Desktop/fyp-work/fyp-ma-13/fyp-models/timeline/(8)_debiased_25/best_weights/set10/model_tl_best_weights_dense_set10.h5')
    elif (model_type == "res"):
        model = tf.keras.models.load_model('/home/monash/Desktop/fyp-work/fyp-ma-13/fyp-models/timeline/(8)_debiased_25/best_weights/set10/model_tl_best_weights_res_set10.h5')
    else:
        raise Exception("Sorry, model_type allowed are 'mobile' (MobileNet), 'dense' (DenseNet)         or 'res' (ResNet50)")
        
    print("Model Loaded")
    
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


# In[16]:


# make_pred('gal.jpeg', 'mobile')


# In[ ]:



