# -*- coding:utf-8 -*-
"""@package web
This method is responsible for the inner workings of the different web pages in this application.
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

bootstrap = Bootstrap(app)

def getData():
    """
    Gets bitmap of all images and returns as a DataFrame.

    Returns
    -------
    data : Pandas DataFrame
        The data that contains the features for each image.
    """

    # Loads in full csvOut data (name & classification)
    with open('csvOut_names.csv', newline='') as csvfile:
        cls_full = list(csv.reader(csvfile))

    # Path to images in local directory
    path = "images_handheld_resized"

    # Initialize array for classification name & data
    bitmap = {}

    # loop through files and get bit map for each (save as object where filename => bitmap for r,g,b)
    for index, file in enumerate(cls_full):
        
        # method found https://stackoverflow.com/questions/46385999/transform-an-image-to-a-bitmap
        #img = Image.open(path + "\\" + file[0]).resize((120, 80))
        #img = Image.open(path + "\\" + file[0]).resize((48, 36))
        img = Image.open(path + "\\" + file[0]).resize((3, 2))
        
        # set dictionary reference
        bitmap[file[0]] = np.array(img).reshape(-1)

    # convert dictionary to pandas data# Create a pandas DataFrame with image names as index and their bitmaps as values
    df = pd.DataFrame.from_dict(bitmap, orient='index')

    # Set the column and index names
    df.index.name = 'Image Name'
    df.columns.name = 'Bitmap'

    return df

def getDataModified():
    """
    Gets bitmap of all images, performs user defined motifications and returns as a DataFrame.

    Returns
    -------
    data : Pandas DataFrame
        The data that contains the features for each image.
    """

    # Loads in full csvOut data (name & classification)
    with open('csvOut_names.csv', newline='') as csvfile:
        cls_full = list(csv.reader(csvfile))

    # Path to images in local directory
    path = "images_handheld_resized"

    # Initialize array for classification name & data
    bitmap = {}

    # loop through files and get bit map for each (save as object where filename => bitmap for r,g,b)
    for index, file in enumerate(cls_full):
        
        # Method found https://stackoverflow.com/questions/46385999/transform-an-image-to-a-bitmap
        #img = Image.open(path + "\\" + file[0]).resize((120, 80))
        #img = Image.open(path + "\\" + file[0]).resize((48, 36))
        img = Image.open(path + "\\" + file[0]).resize((3, 2))

        mr_img = Image_MR(img)
        mr_img.modifyRGB(session['rgb_channel'])
        mr_img.modifyByTransform(session['transform'])
        #mr_img.modifyByInverting(session['invert_data'])
        
        # Must reshape before the rest of the operations (possible upgrade later on)
        mr_img.reshapeBitmap()
        mr_img.modifyByConstant(session['multiply_by_constant'])
        mr_img.modifyByNormalizing(session['normalize_data'])
       
        # set dictionary reference
        bitmap[file[0]] = mr_img.image_data

    # convert dictionary to pandas data# Create a pandas DataFrame with image names as index and their bitmaps as values
    df = pd.DataFrame.from_dict(bitmap, orient='index')

    # Set the column and index names
    df.index.name = 'Image Name'
    df.columns.name = 'Bitmap'

    return df

def createMLModel(data):
    """
    Prepares the training set and creates a machine learning model using the training set.

    Parameters
    ----------
    data : Pandas DataFrame
        The data that contains the features for each image

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

    #can replace RandomForestClassifier with some SVM
    #ml_model = ML_Model(train_set, SVC(kernel="poly", C=0.1, degree=2, coef0=0, gamma="scale", probability=True), DataPreprocessing(True))
    ml_model = ML_Model(train_set, SVC(kernel=session['kernel'], C=float(session['c']), gamma=session['gamma'], probability=True), DataPreprocessing(True))

    return ml_model, train_img_names

def createPreTrainedModel(train_set):
    """
    Loads in best pre-trained model trained offline.

    Parameters
    ----------
    

    Returns
    -------
    ml_model : ML_Model class object (loaded from Pickle, trained with GridSearchCV)
        ml_model pre-trained offline.
    """

    #can replace RandomForestClassifier with some SVM
    ml_model = ML_PreTrained(train_set, DataPreprocessing(True))

    return ml_model

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
    queue = session['queue']
    img = queue.pop()
    session['queue'] = queue
    return render_template(url_for('label'), form = form, picture = img, confidence = session['confidence'], rgb_channel = session['rgb_channel'], multiply_by_constant = session['multiply_by_constant'], transform = session['transform'], normalize_data = session['normalize_data'])

def initializeAL(form, confidence_break, user_defaults):
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

    # set metamorphic relation user selections
    session['rgb_channel'] = format(user_defaults['rgb_channel'])
    session['multiply_by_constant'] = format(user_defaults['multiply_by_constant'])
    session['transform'] = format(user_defaults['transform'])
    session['normalize_data'] = format(user_defaults['normalize_data'])
    
    # set hyperparameter user selections
    session['kernel'] = format(user_defaults['kernel'])
    session['c'] = format(user_defaults['c'])
    session['gamma'] = format(user_defaults['gamma'])

    ml_classifier = SVC(kernel=session['kernel'], C=float(session['c']), gamma=session['gamma'], probability=True)
    data = getData()
    al_model = Active_ML_Model(data, ml_classifier, DataPreprocessing(True)) 

    session['confidence'] = 0
    session['confidence_break'] = confidence_break
    session['labels'] = []
    session['sample_idx'] = list(al_model.sample.index.values)
    session['test'] = list(al_model.test.index.values)
    session['train'] = al_model.train
    session['model'] = True
    session['queue'] = list(al_model.sample.index.values)

    return renderLabel(form)

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
    data = getData()
    ml_model, train_img_names = createMLModel(data)
    test_set = data[data.index.isin(train_img_names) == False]

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
    session['labels'].append(form.choice.data)
    session['sample'] = tuple(zip(session['sample_idx'], session['labels']))

    if session['train'] != None:
        session['train'] = session['train'] + session['sample']
    else:
        session['train'] = session['sample']

    # Get regular set of data, & create model
    data = getData()
    ml_model, train_img_names = createMLModel(data)

    session['confidence'] = np.mean(ml_model.K_fold())
    session['labels'] = []

    if session['confidence'] < session['confidence_break']:
        health_pic, blight_pic = ml_model.infoForProgress(train_img_names)
        return render_template('intermediate.html', form = form, confidence = "{:.2%}".format(round(session['confidence'],4)), health_user = health_pic, blight_user = blight_pic, healthNum_user = len(health_pic), blightNum_user = len(blight_pic))
    else:

        # Get pre-trained model
        ml_model_pretrained = createPreTrainedModel(data)
        session['confidence_preTrain'] = np.mean(ml_model_pretrained.K_fold())

        # Get modified set of data & create model
        data_modified = getDataModified()
        ml_model_modified, train_img_names_modified = createMLModel(data_modified)
        session['confidence_modified'] = np.mean(ml_model_modified.K_fold())

        # Prepare final results for user
        test_set = data.loc[session['test'], :]
        health_pic_user, blight_pic_user, health_pic, blight_pic, health_pic_prob, blight_pic_prob = ml_model.infoForResults(train_img_names, test_set)
        return render_template('final.html', form = form, confidence = "{:.2%}".format(round(session['confidence'],4)), confidence_modified = "{:.2%}".format(round(session['confidence_modified'],4)), 
                                            confidence_preTrain = "{:.2%}".format(round(session['confidence_preTrain'],4)), health_user = health_pic_user, 
                                            blight_user = blight_pic_user, healthNum_user = len(health_pic_user), blightNum_user = len(blight_pic_user), 
                                            health_test = health_pic, unhealth_test = blight_pic, healthyNum = len(health_pic), unhealthyNum = len(blight_pic), 
                                            healthyPct = "{:.2%}".format(len(health_pic)/(200-(len(health_pic_user)+len(blight_pic_user)))), 
                                            unhealthyPct = "{:.2%}".format(len(blight_pic)/(200-(len(health_pic_user)+len(blight_pic_user)))), 
                                            h_prob = health_pic_prob, b_prob = blight_pic_prob)

@app.route("/", methods=['GET'])
@app.route("/index.html",methods=['GET'])
def home():
    """
    Operates the root (/) and index(index.html) web pages.
    """
    session.pop('model', None)
    return render_template('index.html')

@app.route("/menu.html",methods=['GET', 'POST'])
def menu():
    """
    Operates the menu(menu.html) web page.
    """
    form = LabelForm()
    
    """
    if 'model' not in session:#Start
        return initializeAL(form, .7)

    elif session['queue'] == [] and session['labels'] == []: # Need more pictures
        return getNextSetOfImages(form, lowestPercentage)

    elif form.is_submitted() and session['queue'] == []:# Finished Labeling
        return prepairResults(form)

    elif form.is_submitted() and session['queue'] != []: #Still gathering labels
        session['labels'].append(form.choice.data)
        return renderLabel(form)
    """

    return render_template('menu.html', form = form)

@app.route("/label.html",methods=['GET', 'POST'])
def label():
    """
    Operates the label(label.html) web page.
    """
    form = LabelForm()
    if 'model' not in session:#Start
        user_defaults = {}
        user_defaults['rgb_channel'] = request.form.get('rgb_channel')
        user_defaults['multiply_by_constant'] = request.form.get('multiply_by_constant')
        user_defaults['transform'] = request.form.get('transform')
        user_defaults['normalize_data'] = request.form.get('normalize_data') #on/None
        user_defaults['kernel'] = request.form.get('kernel') #on/None
        user_defaults['c'] = request.form.get('c') #on/None
        user_defaults['gamma'] = request.form.get('gamma') #on/None
        return initializeAL(form, .1, user_defaults)

    elif session['queue'] == [] and session['labels'] == []: # Need more pictures
        return getNextSetOfImages(form, lowestPercentage)

    elif form.is_submitted() and session['queue'] == []:# Finished Labeling
        return prepairResults(form)

    elif form.is_submitted() and session['queue'] != []: #Still gathering labels
        session['labels'].append(form.choice.data)
        return renderLabel(form)

    return render_template('label.html', form = form, rgb_channel = session['rgb_channel'])

@app.route("/intermediate.html",methods=['GET'])
def intermediate():
    """
    Operates the intermediate(intermediate.html) web page.
    """
    return render_template('intermediate.html')

@app.route("/final.html",methods=['GET'])
def final():
    """
    Operates the final(final.html) web page.
    """
    return render_template('final.html')

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