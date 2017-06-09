import os, sys, glob
import re
import json
import pickle

import numpy as np

### Collect data about features ###

OUTCOMES = ["gpa","grit","materialHardship","eviction","layoff","jobTraining"]

def parse_codebook_data(fname):
    fNames = dict()
    with open(fname, 'r') as fl:
        lstate = False
        for line in fl:
            if line.startswith('-----'):
                lstate = True
            else:
                if lstate:
                    #print line.replace('  ', '|')
                    while '  ' in line:
                        line = line.replace('  ', ' ')
                    #print line
                    temp = line.strip().split(' ')
                    fNames[temp[0]] = ' '.join(temp[1:])
                lstate = False
    return fNames


def get_feature_name_definitions():
    featureNames = dict()
    # Looking for a path that has all feature description files
    for fname in glob.glob('data/codebooks/ff*.txt'):
        featureNames.update(parse_codebook_data(fname))
    return featureNames


def get_feature_list(fname='data/raw_data/background.csv'):
    with open(fname, 'r') as fl:
        header = set(fl.readline().strip().replace('"','').split(','))
        return list(header - {'challengeID'})

# Return set of features common to all users
def get_core_feature_list():
    coreFeatures = set(get_feature_list())
    for uid, feat in iterate_user_background_features():
        coreFeatures = coreFeatures & set(feat)
    return sorted(coreFeatures)

# Get all relevant features by given keywords. For instance 'education', 'mother', etc.
def get_feature_by_keyword(keyword):
    flist = list()
    fDefinition = get_feature_name_definitions()
    for f in get_core_feature_list():
        if keyword in fDefinition.get(f,''):
            flist.append(f)
    return flist

def parse_feature_name(fstr):
    temp = re.split('(\d+)',fstr)
    return {'prefix':temp[0].split('_')[0], 'wave':int(temp[1]), 'question':''.join(temp[2:])}

# Select groups of features that represent different prefix and wave
def select_feature_by_type(features=None, prefix=None, wave=None):
    if features == None:
        features = get_feature_list()

    selectedFeatures = list()
    for f in features:
        try:
            ftype = parse_feature_name(f)
        except:
            continue

        toAddP = True
        if prefix <> None:
            toAddP = True if ftype['prefix'] in prefix else False

        toAddW = True
        if wave <> None:
            toAddW = True if ftype['wave'] in wave else False
            
        if toAddP & toAddW:
            selectedFeatures.append(f)

    return selectedFeatures


def iterate_user_missing_features(fname='data/raw_data/background.csv'):
    with open(fname, 'r') as fl:
        header = fl.readline().strip().replace('"','').split(',')
        #print header
        for line in fl:
            temp = line.strip().split(',')
            
            ufeat = dict()
            missingFeatures = list()
            for i in range(1,len(header)):
                try:
                    ufeat[header[i]] = float(temp[i])
                except:
                    missingFeatures.append(header[i])

            mfeat = dict()
            for f in missingFeatures:
                try:
                    ftype = parse_feature_name(f)
                except:
                    continue
                mf = '{}{}'.format(ftype['prefix'], ftype['wave'])
                if mf not in mfeat:
                    mfeat[mf] = 0
                mfeat[mf] += 1

            yield int(temp[0]), mfeat

### LOAD DATASETS ###

def iterate_user_background_features(fname='data/raw_data/background.csv'):
    with open(fname, 'r') as fl:
        header = fl.readline().strip().replace('"','').split(',')
        #print header
        for line in fl:
            temp = line.strip().split(',')
            ufeat = dict()
            for i in range(1,len(header)):
                try:
                    ufeat[header[i]] = float(temp[i])
                except:
                    pass
            yield int(temp[0]), ufeat

def iterate_training_data(fname='data/raw_data/train.csv'):
    with open(fname, 'r') as fl:
        header = fl.readline().strip().replace('"','').split(',')
        #print header
        for line in fl:
            temp = line.strip().split(',')
            ulabels = dict()
            for i in range(1,len(header)):
                try:
                    ulabels[header[i]] = float(temp[i])
                except:
                    pass
            yield int(temp[0]), ulabels


def combine_user_feature_iterators(externalFeatures):
    for uid, feat in iterate_user_background_features():
        for ext in externalFeatures:
            if uid in ext:
                feat.update(ext[uid])
        yield uid, feat


# Creates dataset for ML models
def create_data_matrix(featureList=None, outcome='gpa', userFeatureIterator=None):
    if featureList == None:
        featureList = sorted(get_feature_list())

    if userFeatureIterator == None:
        userFeatureIterator = iterate_user_background_features()

    trX, teX, trY = dict(), dict(), dict()

    for uid, lbls in iterate_training_data():
        if outcome in lbls:
            trY[uid] = lbls[outcome]

    uidLbls = dict()
    for uid, feat in userFeatureIterator:
        if uid in trY:
            trX[uid] = [feat.get(f,0) for f in featureList]
            uidLbls[uid] = True
        else:
            teX[uid] = [feat.get(f,0) for f in featureList]
            uidLbls[uid] = False


    trX = np.array([trX[u] for u in sorted(trX.keys())])
    teX = np.array([teX[u] for u in sorted(teX.keys())])
    trY = np.array([trY[u] for u in sorted(trY.keys())])
    print 'TrX: {}, TrY: {}, TeX: {}'.format(trX.shape, trY.shape, teX.shape)

    return trX, trY, teX, uidLbls



