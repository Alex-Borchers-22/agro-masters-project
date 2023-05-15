# -*- coding:utf-8 -*-
"""@package web
This method is responsible for the inner workings of the different web pages in this application.

Modified on Mon May 15 2023
@author Alex Borchers

"""
from flask import Flask
from flask import render_template, flash, redirect, url_for, session, request, jsonify
from app import app
from app.DataPreprocessing import DataPreprocessing
from app.ML_Class import Active_ML_Model, AL_Encoder, ML_Model, ML_PreTrained
from app.Image_Transform_Class import Image_MR
from app.SamplingMethods import lowestPercentage
from app.forms import LabelForm
from flask_bootstrap import Bootstrap
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
import os
import numpy as np
import boto3
from io import StringIO
import csv
from PIL import Image #to open images
from app.gpt_api import generateChatResponse
from app.gpt_api import mutateHyperParameters
import time
import random
import datetime

bootstrap = Bootstrap(app)

# Globals in use
ml_model = None
ml_model_pretrain = None
ml_model_modified = None
data_standard = None
data_modified = None
train_img_names = None
test_set = None
full_classification = {}
class_list = {}
confidence_list = {}
h_u_count = {}
mutated_list = {}
accuracy = {}

def getData(pretrain = False):
    """
    Gets bitmap of all images and returns as a DataFrame.

    Returns
    -------
    data_standard : Pandas DataFrame
        The data that contains the bitmap for each image (resized to 120x80)
    data_modified : Pandas DataFrame
        The data that contains the modified bitmap for each image (resized to 120x80).
    """

    # Loads in full csvOut data (name & classification)
    with open('csvOut_names.csv', newline='') as csvfile:
        cls_full = list(csv.reader(csvfile))

    # Path to images in local directory
    path = "images_handheld_resized"

    # Initialize array for classification name & data
    bitmap = {}
    bitmap_modified = {}

    # loop through files and get bit map for each (save as object where filename => bitmap for r,g,b)
    for index, file in enumerate(cls_full):
        
        # Get actual classification
        full_classification[file[0]] = file[1]

        # method found https://stackoverflow.com/questions/46385999/transform-an-image-to-a-bitmap
        #img = Image.open(path + "\\" + file[0]).resize((120, 80))
        #img = Image.open(path + "\\" + file[0]).resize((90, 60))
        img = Image.open(path + "\\" + file[0]).resize((3, 2))
        
        # set dictionary reference for standard
        bitmap[file[0]] = np.array(img).reshape(-1)

        # set dictionary reference for modified
        mr_img = Image_MR(img)
        mr_img.modifyRGB(session['rgb_channel'])
        mr_img.modifyByTransform(session['transform'])
        mr_img.modifyByInverting(session['invert_data'])
        mr_img.modifyByPermutation(session['permute_data'])
        
        # Must reshape before the rest of the operations (possible upgrade later on)
        mr_img.reshapeBitmap()
        mr_img.modifyByConstant(session['multiply_by_constant'])
        mr_img.modifyByNormalizing(session['normalize_data'])
       
        # set dictionary reference
        bitmap_modified[file[0]] = mr_img.image_data

    # convert dictionary to pandas data# Create a pandas DataFrame with image names as index and their bitmaps as values
    df = pd.DataFrame.from_dict(bitmap, orient='index')
    df_modified = pd.DataFrame.from_dict(bitmap_modified, orient='index')

    # Set the column and index names
    df.index.name = 'Image Name'
    df.columns.name = 'Bitmap'
    df_modified.index.name = 'Image Name'
    df_modified.columns.name = 'Bitmap'

    #return df, df_modified
    return df, df_modified

def custom_log(message, file='ml_tracking.txt'):
    """
    Prepares the training set and creates a machine learning model using the training set.

    Parameters
    ----------
    message : String
        The message to be logged
    file : String
        The file to write log to

    """
    # Configure logging, write to file, close logger
    now = datetime.datetime.now()
    full_msg = now.strftime("%Y-%m-%d %H:%M:%S") + ": " + message + "\n"
    with open(file, 'a') as f:
        f.write(full_msg)

def createMLModel(data):
    """
    Prepares the training set and creates a machine learning model using the training set.

    Parameters
    ----------
    data : Pandas DataFrame
        The data that contains the bitmap for each image

    Returns
    -------
    ml_model : ML_Model class object
        ml_model created from the training set.
    train_img_names : String
        The names of the images.
    """
    # session is a pair that holds image label and classification (user classified)
    train_img_names, train_img_label = list(zip(*session['train']))
    train_set = data.loc[train_img_names, :]
    train_set['y_value'] = train_img_label
    ml_model = ML_Model(train_set, SVC(kernel=session['kernel'], C=float(session['c']), gamma=session['gamma'], probability=True), DataPreprocessing(False))

    return ml_model, train_img_names

def createPreTrainedModel(data):
    """
    Loads in best pre-trained model trained offline.

    Parameters
    ----------
    data : Pandas DataFrame
        The data that contains the bitmap for each image

    Returns
    -------
    ml_model : ML_Model class object (loaded from Pickle, trained with GridSearchCV)
        ml_model pre-trained offline.
    train_img_names : String
        The names of the images.
    """

    # session is a pair that holds image label and classification (user classified)
    train_img_names, train_img_label = list(zip(*session['train']))
    train_set = data.loc[train_img_names, :]
    train_set['y_value'] = train_img_label

    #can replace RandomForestClassifier with some SVM
    ml_model = ML_PreTrained(train_set, DataPreprocessing(False))

    return ml_model, train_img_names

def renderLabel(form):
    """
    prepairs a render_template to show the label.html web page.

    Parameters
    ----------
    form : LabelForm class object
        form to be used when displaying label.html

    Returns
    -------
    render_template : flask function
        renders the label.html webpage.
    """
    from collections import deque

    queue = deque(session['queue'])
    img = queue.popleft()
    session['queue'] = list(queue)

    return render_template(url_for('label'), form = form, picture = img, confidence = session['confidence'], rgb_channel = session['rgb_channel'], multiply_by_constant = session['multiply_by_constant'], transform = session['transform'], normalize_data = session['normalize_data'], invert_data = session['invert_data'], permute_data = session['permute_data'],remaining = len(queue))

def initializeAL(form, confidence_break):
    """
    Initializes the active learning model and sets up the webpage with everything needed to run the application.

    Parameters
    ----------
    form : LabelForm class object
        form to be used when displaying label.html    
    confidence_break : number
        How confident the model is.
    user_defaults : Dictionary
        The MR relations selected on menu.html. (rgb, multiply_by_constant, transform, normalize_data)

    Returns
    -------
    render_template : flask function
        renders the label.html webpage.
    """

    # identify globals to use
    global data_standard, data_modified, accuracy

    # Log a session start
    custom_log("[SESSION START]")

    # set metamorphic relation user selections
    session['rgb_channel'] = format(request.form.get('rgb_channel'))
    session['multiply_by_constant'] = format(request.form.get('multiply_by_constant'))
    session['transform'] = format(request.form.get('transform'))
    session['normalize_data'] = format(request.form.get('normalize_data'))
    session['invert_data'] = format(request.form.get('invert_data'))
    session['permute_data'] = format(request.form.get('permute_data'))
    
    # Log MR user choices
    custom_log("rgb_channel=" + session['rgb_channel'] + ",mult_const=" + session['multiply_by_constant'] + ",transform=" + session['transform'] + ",normalize_data=" + session['normalize_data'] + ",invert_data=" + session['invert_data'] + ",permute_data=" + session['permute_data'])

    # Get standard & modified data
    data_standard, data_modified = getData() 

    # set hyperparameter user selections
    session['kernel'] = format(request.form.get('kernel'))
    session['c'] = format(request.form.get('c'))
    session['gamma'] = format(request.form.get('gamma'))

    # Log HP user choices
    custom_log("kernel=" + session['kernel'] + ",c=" + session['c'] + ",gamma=" + session['gamma'])

    # convert gamma to float if not auto or scale
    if session['gamma'] != "auto" and session['gamma'] != "scale":
        session['gamma'] = float(session['gamma'])

    ml_classifier = SVC(kernel=session['kernel'], C=float(session['c']), gamma=session['gamma'], probability=True)
    #data_standard, data_modified = getData()
    al_model = Active_ML_Model(data_standard, ml_classifier, DataPreprocessing(False), full_classification) 

    session['confidence'] = 0
    session['confidence_preTrain'] = 0
    session['confidence_modified'] = 0
    session['confidence_break'] = confidence_break
    session['number_break'] = 20
    session['labels'] = []
    session['sample_idx'] = list(al_model.sample.index.values)
    session['test'] = list(al_model.test.index.values)
    session['train'] = al_model.train
    session['model'] = True
    session['queue'] = list(al_model.sample.index.values)

    # TESTING (use correct labels, skip to prepair-results)
    #session['labels'] = getCorrectLabels(session['queue'])
    #session['queue'] = []
    #session['confidence_break'] = 0.1
    #return prepairResults(form) 
    
    # PRODUCTION (begin labelling process)
    return renderLabel(form)

def getUserAccuracy():
    """
    Gets accuracy of users selections thus far

    Returns
    -------
    accuracy : float
        accuracy of the users selections
    """

    from sklearn.metrics import accuracy_score
    users, actuals = getLabelClassifications(session['train'])
    accuracy = accuracy_score(users, actuals)
    return accuracy

def getLabelClassifications(user_selection):
        """
        This function returns the predicted accuracy of the model

        Parameters
        ----------
        labels : list
            list of image names in the training set.

        Returns
        -------
        users : list
            The user selected classifications
        actual_list : list
            The actual classification [B/U] for the test set of images
        """
        # identify globals to use
        global full_classification

        # Loop through labels, and push actuals to a new list, that is returned
        users = []
        actuals = []
        for choice in user_selection:
            users.append(choice[1])
            actuals.append(full_classification[choice[0]])
        return users, actuals

def getCorrectLabels(queue):
        """
        This function returns the correct labels for the upcoming queue (used for testing)

        Parameters
        ----------
        labels : list
            list of image names in the training set.

        Returns
        -------
        correct_label : list
            The actual classification [B/U] for the queue set of images
        """
        # identify globals to use
        global full_classification

        # Loop through labels, and push actuals to a new list, that is returned
        correct_label = []
        for pic in queue:
            correct_label.append(full_classification[pic])
        return correct_label

def getNextSetOfImages(form, sampling_method):
    """
    Uses a sampling method to get the next set of images needed to be labeled.

    Parameters
    ----------
    form : LabelForm class object
        form to be used when displaying label.html
    sampling_method : SamplingMethods Function
        function that returns the queue and the new test set that does not contain the queue.

    Returns
    -------
    render_template : flask function
        renders the label.html webpage.
    """
    global data_standard
    ml_model, train_img_names = createMLModel(data_standard)
    test_set = data_standard[data_standard.index.isin(train_img_names) == False]

    from sklearn.utils import shuffle
    test_set = shuffle(test_set)
    session['sample_idx'], session['test'] = sampling_method(ml_model, test_set, 5)
    session['queue'] = session['sample_idx'].copy()

    return renderLabel(form)

def prepairResults(form):
    """
    Creates the new machine learning model and gets the confidence of the machine learning model.

    Parameters
    ----------
    form : LabelForm class object
        form to be used when displaying label.html

    Returns
    -------
    render_template : flask function
        renders the appropriate webpage based on new confidence score.
    """

    # Include globals needed in this function
    global ml_model, ml_model_pretrain, ml_model_modified, train_img_names, test_set, data_standard, data_modified, class_list, confidence_list, h_u_count, mutated_list, accuracy, full_classification

    session['labels'].append(form.choice.data)
    session['sample'] = tuple(zip(session['sample_idx'], session['labels']))

    if session['train'] != None:
        session['train'] = session['train'] + session['sample']
    else:
        session['train'] = session['sample']

    # Get regular set of data, & create model
    ml_model, train_img_names = createMLModel(data_standard)
    ml_model_modified, train_img_names_modified = createMLModel(data_modified)

    session['confidence'] = np.mean(ml_model.K_fold())
    session['labels'] = []

    test_set = data_standard.loc[session['test'], :]
    user_accuracy = getUserAccuracy()
    result_opts = ['10', '25', '50', 'All']

    if session['confidence'] < session['confidence_break'] and len(session['train']) < session['number_break']:
        health_pic, blight_pic = ml_model.infoForProgress(train_img_names)
        return render_template('intermediate.html', form = form, confidence = "{:.2%}".format(round(session['confidence'],4)), health_user = health_pic, blight_user = blight_pic, healthNum_user = len(health_pic), blightNum_user = len(blight_pic), user_accuracy = user_accuracy)
    else:

        # Reset globals
        for type in ['regular', 'modified', 'pretrain']:
            class_list[type] = []
            confidence_list[type] = []
            h_u_count[type] = []
            mutated_list[type] = []
            accuracy[type] = []

        # get test set of images
        test_set = data_standard.loc[session['test'], :]
        
        # Get pre-trained model
        ml_model_pretrain, train_img_names_pretrain = createPreTrainedModel(data_standard)
        session['confidence_preTrain'] = np.mean(ml_model_pretrain.K_fold())

        # Get modified set of data & create model
        ml_model_modified, train_img_names_modified = createMLModel(data_modified)
        session['confidence_modified'] = np.mean(ml_model_modified.K_fold())

        # Prepare final results for user
        health_pic_user, blight_pic_user, health_pic, blight_pic, health_pic_prob, blight_pic_prob = ml_model.infoForResults(train_img_names, test_set)
        PT_health_pic_user, PT_blight_pic_user, PT_health_pic, PT_blight_pic, PT_health_pic_prob, PT_blight_pic_prob = ml_model_pretrain.infoForResults(train_img_names, test_set)
        MD_health_pic_user, MD_blight_pic_user, MD_health_pic, MD_blight_pic, MD_health_pic_prob, MD_blight_pic_prob = ml_model_modified.infoForResults(train_img_names, test_set)

        # Push new values to class list and confidence list
        class_list['regular'].append(str(ml_model.ml_model.get_params()))
        confidence_list['regular'].append(session['confidence'])
        h_u_count['regular'].append(str(len(health_pic)) + " - " + str(len(blight_pic)))
        mutated_list['regular'].append("")
        accuracy['regular'].append(ml_model.getAccuracy(test_set, full_classification))

        class_list['modified'].append(str(ml_model_modified.ml_model.get_params()))
        confidence_list['modified'].append(session['confidence_modified'])
        h_u_count['modified'].append(str(len(MD_health_pic)) + " - " + str(len(MD_blight_pic)))
        mutated_list['modified'].append("")
        accuracy['modified'].append(ml_model_modified.getAccuracy(test_set, full_classification))

        class_list['pretrain'].append(str(ml_model_pretrain.ml_model.get_params()))
        confidence_list['pretrain'].append(session['confidence_preTrain'])
        h_u_count['pretrain'].append(str(len(PT_health_pic)) + " - " + str(len(PT_blight_pic)))  
        mutated_list['pretrain'].append("") 
        accuracy['pretrain'].append(ml_model_pretrain.getAccuracy(test_set, full_classification))

        # Log current results
        train_img_len = len(health_pic_user) + len(blight_pic_user)
        custom_log("[R] train_img_len=" +  str(train_img_len) + ",Confidence=" + str(session['confidence']) + ",Accuracy=" + str(accuracy['regular'][-1]))
        custom_log("[M] train_img_len=" +  str(train_img_len) + ",Confidence=" + str(session['confidence_modified']) + ",Accuracy=" + str(accuracy['modified'][-1]))
        custom_log("[P] train_img_len=" +  str(train_img_len) + ",Confidence=" + str(session['confidence_preTrain']) + ",Accuracy=" + str(accuracy['pretrain'][-1]))

        #health_pic_user, blight_pic_user, health_pic, blight_pic, health_pic_prob, blight_pic_prob = ml_model_pretrain.infoForResults(train_img_names_pretrain, test_set)
        return render_template('final.html', form = form, user_accuracy = user_accuracy, result_opts = result_opts, confidence = "{:.2%}".format(round(session['confidence'],4)), confidence_modified = "{:.2%}".format(round(session['confidence_modified'],4)), 
                                            confidence_preTrain = "{:.2%}".format(round(session['confidence_preTrain'],4)), health_user = health_pic_user, 
                                            blight_user = blight_pic_user, healthNum_user = len(health_pic_user), blightNum_user = len(blight_pic_user), 
                                            health_test = health_pic, unhealth_test = blight_pic, healthyNum = len(health_pic), h_loop = min(len(health_pic), 10),
                                            unhealthyNum = len(blight_pic), u_loop = min(len(blight_pic), 10),
                                            healthyPct = "{:.2%}".format(len(health_pic)/(177-(len(health_pic_user)+len(blight_pic_user)))), 
                                            unhealthyPct = "{:.2%}".format(len(blight_pic)/(177-(len(health_pic_user)+len(blight_pic_user)))), 
                                            h_prob = health_pic_prob, b_prob = blight_pic_prob, ml_model_list = class_list['regular'], ml_pretrain_list = class_list['pretrain'], 
                                            ml_modified_list = class_list['modified'], c_list = confidence_list['regular'], c_pretrain_list = confidence_list['pretrain'], 
                                            c_modified_list = confidence_list['modified'], reg_h_b_count = h_u_count['regular'], PT_h_b_count = h_u_count['pretrain'],
                                            MD_h_b_count = h_u_count['modified'], m_list = mutated_list['regular'], m_list_modified = mutated_list['modified'], 
                                            m_list_pretrain = mutated_list['pretrain'], accuracy = accuracy['regular'], accuracy_modified = accuracy['modified'], 
                                            accuracy_pretrain = accuracy['pretrain'], result_options = ["Original", "Modified", "Pretrain"], model_type = "Original",
                                            rgb = session['rgb_channel'], transform = session['transform'], invert_data = session['invert_data'], 
                                            multiply_by_constant = session['multiply_by_constant'], normalize_data = session['normalize_data'])

@app.route("/", methods=['GET'])
@app.route("/index.html",methods=['GET'])
def home():
    """
    Operates the root (/) and index(index.html) web pages.
    """
    session.pop('model', None)
    return render_template('index.html')

@app.route("/gpt",methods=['POST'])
def test():
    """
    Operates the GPT requests
    """    
    time.sleep(2)
    prompt = request.form['prompt']
    res = {}
    res['answer'] = generateChatResponse(prompt)
    return jsonify(res), 200

@app.route("/menu.html",methods=['GET', 'POST'])
def menu():
    """
    Operates the menu(menu.html) web page.
    """
    form = LabelForm()
    return render_template('menu.html', form = form)

@app.route("/label.html",methods=['GET', 'POST'])
def label():
    """
    Operates the label(label.html) web page.
    """

    global data_standard, data_modified

    form = LabelForm()
    if 'model' not in session:#Start       
        return initializeAL(form, .7)

    elif session['queue'] == [] and session['labels'] == []: # Need more pictures
        return getNextSetOfImages(form, lowestPercentage)

    elif form.is_submitted() and session['queue'] == []:# Finished Labeling
        return prepairResults(form)

    elif form.is_submitted() and session['queue'] != []: #Still gathering labels

        # Log user label, add to label list, render label.html
        custom_log(session['sample_idx'][len(session['labels'])] + "=" + form.choice.data)
        session['labels'].append(form.choice.data)
        return renderLabel(form)

    return render_template('label.html', form = form)

@app.route("/intermediate.html",methods=['GET'])
def intermediate():
    """
    Operates the intermediate(intermediate.html) web page.
    """
    return render_template('intermediate.html')

@app.route("/final.html",methods=['POST'])
def final():
    """
    Operates the final(final.html) web page.
    """
    form = LabelForm()
    global ml_model, ml_model_pretrain, ml_model_modified, train_img_names, test_set, data_standard, data_modified, class_list, confidence_list, h_u_count, mutated_list, accuracy, full_classification

    # If we are requesting more results, set limit_reference
    if 'result_limit' in request.form:
        limit_reference = request.form['result_limit']
    else:
        limit_reference = 10

    # Depending on user's decision, render new final.html template
    if 'model_type' in request.form:

        # Prepare final results for user
        health_pic_user, blight_pic_user, health_pic, blight_pic, health_pic_prob, blight_pic_prob = ml_model.infoForResults(train_img_names, test_set)
        MD_health_pic_user, MD_blight_pic_user, MD_health_pic, MD_blight_pic, MD_health_pic_prob, MD_blight_pic_prob = ml_model_modified.infoForResults(train_img_names, test_set)
        PT_health_pic_user, PT_blight_pic_user, PT_health_pic, PT_blight_pic, PT_health_pic_prob, PT_blight_pic_prob = ml_model_pretrain.infoForResults(train_img_names, test_set)

        # Update values to use for healthy/unhealthy breakout based on type
        if 'model_type' in request.form: 
            model_type = request.form['model_type']
        else: 
            model_type = "Original"

        if model_type == "Original":
            use_h_pic_user = health_pic_user
            use_b_pic_user = blight_pic_user
            use_h_pic = health_pic
            use_b_pic = blight_pic
            use_h_prob = health_pic_prob
            use_b_prob = blight_pic_prob
        elif model_type == "Modified": 
            use_h_pic_user = MD_health_pic_user
            use_b_pic_user = MD_blight_pic_user
            use_h_pic = MD_health_pic
            use_b_pic = MD_blight_pic
            use_h_prob = MD_health_pic_prob
            use_b_prob = MD_blight_pic_prob
        else:
            use_h_pic_user = PT_health_pic_user
            use_b_pic_user = PT_blight_pic_user
            use_h_pic = PT_health_pic
            use_b_pic = PT_blight_pic
            use_h_prob = PT_health_pic_prob
            use_b_prob = PT_blight_pic_prob           

    # Otherwise, process mutation request
    else:

        # for testing, make loops
        #for i in range(5):
        model_type = "Original"

        if request.form['mutation_type'] == "hp":
            # Make call to chat GPT to create mutation of hyperparameters
            ml_model, mutated_list['regular'], class_list['regular'] = mutateHyperParameters(ml_model, mutated_list['regular'], class_list['regular'])
            ml_model_modified, mutated_list['modified'], class_list['modified'] = mutateHyperParameters(ml_model_modified, mutated_list['modified'], class_list['modified'])
            ml_model_pretrain, mutated_list['pretrain'], class_list['pretrain'] = mutateHyperParameters(ml_model_pretrain, mutated_list['pretrain'], class_list['pretrain'])

        elif request.form['mutation_type'] == "sv":
            # Randomly change 1 support vector
            for model_type in ['regular', 'modified', 'pretrain']:

                # Pick model based on type
                if model_type == "regular":
                    model = ml_model
                elif model_type == "modified":
                    model = ml_model_modified
                else:
                    model = ml_model_pretrain

                rand_vector = random.randint(0, len(model.ml_model.support_vectors_) - 1)
                rand_multiplier = random.uniform(0.9, 1.1)
                model.ml_model.support_vectors_[rand_vector] = model.ml_model.support_vectors_[rand_vector] * rand_multiplier 
                mutated_list[model_type].append("sv," + str(rand_vector) + "," + str(rand_multiplier))
                class_list[model_type].append(str(model.ml_model.get_params()))            
    
        # Re-run confidence levels & final results
        session['confidence'] = np.mean(ml_model.K_fold())
        session['confidence_modified'] = np.mean(ml_model_modified.K_fold())
        session['confidence_preTrain'] = np.mean(ml_model_pretrain.K_fold()) 

        # Prepare final results for user
        health_pic_user, blight_pic_user, health_pic, blight_pic, health_pic_prob, blight_pic_prob = ml_model.infoForResults(train_img_names, test_set)
        MD_health_pic_user, MD_blight_pic_user, MD_health_pic, MD_blight_pic, MD_health_pic_prob, MD_blight_pic_prob = ml_model_modified.infoForResults(train_img_names, test_set)
        PT_health_pic_user, PT_blight_pic_user, PT_health_pic, PT_blight_pic, PT_health_pic_prob, PT_blight_pic_prob = ml_model_pretrain.infoForResults(train_img_names, test_set)

        # Set defaults for 'use' for all defaults
        use_h_pic_user = health_pic_user
        use_b_pic_user = blight_pic_user
        use_h_pic = health_pic
        use_b_pic = blight_pic
        use_h_prob = health_pic_prob
        use_b_prob = blight_pic_prob

        # Push new values to class list and confidence list
        #class_list['regular'].append(str(ml_model.ml_model))
        confidence_list['regular'].append(session['confidence'])
        h_u_count['regular'].append(str(len(health_pic)) + " - " + str(len(blight_pic)))
        accuracy['regular'].append(ml_model.getAccuracy(test_set, full_classification))

        #class_list['modified'].append(str(ml_model_modified.ml_model))
        confidence_list['modified'].append(session['confidence_modified'])
        h_u_count['modified'].append(str(len(MD_health_pic)) + " - " + str(len(MD_blight_pic)))
        accuracy['modified'].append(ml_model_modified.getAccuracy(test_set, full_classification))

        #class_list['pretrain'].append(str(ml_model_pretrain.ml_model))
        confidence_list['pretrain'].append(session['confidence_preTrain'])
        h_u_count['pretrain'].append(str(len(PT_health_pic)) + " - " + str(len(PT_blight_pic)))  
        accuracy['pretrain'].append(ml_model_pretrain.getAccuracy(test_set, full_classification))

        # Log current results
        train_img_len = len(use_h_pic_user) + len(use_h_pic_user)
        custom_log("[R] train_img_len=" +  str(train_img_len) + ",Confidence=" + str(session['confidence']) + ",Accuracy=" + str(accuracy['regular'][-1]))
        custom_log("[M] train_img_len=" +  str(train_img_len) + ",Confidence=" + str(session['confidence_modified']) + ",Accuracy=" + str(accuracy['modified'][-1]))
        custom_log("[P] train_img_len=" +  str(train_img_len) + ",Confidence=" + str(session['confidence_preTrain']) + ",Accuracy=" + str(accuracy['pretrain'][-1]))

    if limit_reference == "All":
        h_limit = len(health_pic)
        b_limit = len(blight_pic)
    else:
        h_limit = min(len(health_pic), int(limit_reference))
        b_limit = min(len(blight_pic), int(limit_reference))

    # Get user accuracy
    user_accuracy = getUserAccuracy()
    result_opts = ['10', '25', '50', 'All']

    return render_template('final.html', form = form, user_accuracy = user_accuracy, result_opts = result_opts,
                                        confidence = "{:.2%}".format(round(session['confidence'],4)), confidence_modified = "{:.2%}".format(round(session['confidence_modified'],4)), confidence_preTrain = "{:.2%}".format(round(session['confidence_preTrain'],4)), 
                                        health_user = use_h_pic_user, blight_user = use_b_pic_user, healthNum_user = len(use_h_pic_user), blightNum_user = len(use_b_pic_user), 
                                        health_test = use_h_pic, unhealth_test = use_b_pic, healthyNum = len(health_pic), h_loop = h_limit,
                                        unhealthyNum = len(blight_pic), u_loop = b_limit,
                                        healthyPct = "{:.2%}".format(len(use_h_pic)/(177-(len(use_h_pic_user)+len(use_b_pic_user)))), 
                                        unhealthyPct = "{:.2%}".format(len(blight_pic)/(177-(len(use_h_pic_user)+len(use_b_pic_user)))), 
                                        h_prob = use_h_prob, b_prob = use_b_prob, 
                                        ml_model_list = class_list['regular'], ml_pretrain_list = class_list['pretrain'], 
                                        ml_modified_list = class_list['modified'], c_list = confidence_list['regular'], c_pretrain_list = confidence_list['pretrain'], 
                                        c_modified_list = confidence_list['modified'], reg_h_b_count = h_u_count['regular'], PT_h_b_count = h_u_count['pretrain'],
                                        MD_h_b_count = h_u_count['modified'], m_list = mutated_list['regular'], m_list_modified = mutated_list['modified'], 
                                        m_list_pretrain = mutated_list['pretrain'], accuracy = accuracy['regular'], accuracy_modified = accuracy['modified'], 
                                        accuracy_pretrain = accuracy['pretrain'], result_options = ["Original", "Modified", "Pretrain"], model_type = model_type,
                                        rgb = session['rgb_channel'], transform = session['transform'], invert_data = session['invert_data'], 
                                        multiply_by_constant = session['multiply_by_constant'], normalize_data = session['normalize_data'])

@app.route("/feedback/<h_list>/<u_list>/<h_conf_list>/<u_conf_list>",methods=['GET'])
def feedback(h_list,u_list,h_conf_list,u_conf_list):
    """
    Operates the feedback(feedback.html) web page.
    """
    h_feedback_result = list(h_list.split(","))
    u_feedback_result = list(u_list.split(","))
    h_conf_result = list(h_conf_list.split(","))
    u_conf_result = list(u_conf_list.split(","))
    h_length = len(h_feedback_result)
    u_length = len(u_feedback_result)
    
    return render_template('feedback.html', healthy_list = h_feedback_result, unhealthy_list = u_feedback_result, healthy_conf_list = h_conf_result, unhealthy_conf_list = u_conf_result, h_list_length = h_length, u_list_length = u_length)

#app.run( host='127.0.0.1', port=5000, debug='True', use_reloader = False)