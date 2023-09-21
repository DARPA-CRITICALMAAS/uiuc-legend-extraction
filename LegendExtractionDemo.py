import os
import cv2
import json
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

log = logging.getLogger('DARPA_CMASS')

# Internal Project files
import src.logging_utils as log_utils
import src.visualizations as my_viz

from src.extraction import extractLegends
from src.scoring import scoreLegendExtraction

sourceDataFolder = '../data/training'
outputFolder = 'scoring_results'

# Load a USGS formated json file
def loadUSGSJson(filepath, polyDataOnly=False):
    with open(filepath, 'r') as fh:
        json_data = json.load(fh)

    if polyDataOnly:
        json_data['shapes'] = [s for s in json_data['shapes'] if s['label'].split('_')[-1] == 'poly']

    # Convert pix coords to int
    for feature in json_data['shapes']:
        feature['points'] = np.array(feature['points']).astype(int)

    return json_data

def safeLoadImg(filepath):
    if not os.path.exists(filepath):
        log.warning('Image file "{}" does not exist. Skipping file'.format(filepath))
        return None

    img = cv2.imread(filepath)
    if img is None:
        log.warning('Could not load {}. Skipping file'.format(filepath))
        return None
    return img

def main():
    logfile = 'Latest.log'
    logdir = 'logs'
    # Start logger
    if os.path.exists(logfile):
        # Rename old latest to the date and move to logs directory
        with open(logfile) as fh:
            newfilename = '{}_{}.log'.format(*(fh.readline().split(' ')[0:2]))
            newfilename = newfilename.replace('/','-').replace(':','-')
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        os.rename(logfile, os.path.join(logdir, newfilename))
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    log_utils.setup_logger(logfile, logging.DEBUG)

    log.info('Running legend extraction on "{}"'.format(sourceDataFolder))
    dataset_name = os.path.basename(sourceDataFolder.replace('\\','/'))

    full_score_df = pd.DataFrame()
    finished, failed = 0, 0
    # Iterate through each json file
    pbar = tqdm([f for f in os.listdir(sourceDataFolder) if f.endswith('json')])
    for file in pbar:
        pbar.set_description('Processing {}'.format(file))
        pbar.refresh()
        log.info('Processing {}'.format(file))
        

        # Load json
        true_json = loadUSGSJson(os.path.join(sourceDataFolder, file), polyDataOnly=True)
        mapname = os.path.splitext(os.path.basename(true_json['imagePath'].replace('\\','/')))[0]
        log.info('\tLoaded {} with {} polygon features'.format(file, len(true_json['shapes'])))

        # Load img
        img = safeLoadImg(os.path.join(sourceDataFolder, (os.path.splitext(file)[0] + '.tif')))
        if img is None:
            failed = failed + 1
            continue
        log.info('\tLoaded {} image'.format(os.path.splitext(file)[0] + '.tif'))
        
        # Generate Predictions
        log.info('\tExtracting Legends')
        predict_json = extractLegends(img)
        log.info('\tPredicted {} polygon legend features'.format(len(predict_json['shapes'])))
  
        log.info('\tScoring predictions')
        score_df = scoreLegendExtraction(img, true_json, predict_json, outputDir=os.path.join(outputFolder, mapname))

        # Do this after scoring so conversion to list doesn't effect anything
        for s in predict_json['shapes']:
            s['points'] = s['points'].tolist()
        with open(os.path.join(outputFolder, mapname + '_predict.json'), 'w') as fh:
            fh.write(json.dumps(predict_json))
        full_score_df = pd.concat([full_score_df, score_df])
        finished = finished + 1
    
    log.info('Finished scoring {} maps. {} maps failed'.format(finished, failed))
    full_score_df.to_csv(os.path.join(outputFolder, '#' + dataset_name + '_Scores.csv'))

    if outputFolder is not None:
        fig, ax = plt.subplots(2,2, figsize=(8,8))
        fig.suptitle(dataset_name + ' Dataset Results')
        my_viz.datasetRecallPlot(ax[0,0], full_score_df)
        my_viz.datasetPrecisionPlot(ax[0,1], full_score_df)
        my_viz.averageMapRecallPlot(ax[1,0], full_score_df)
        my_viz.averageMapPrecisionPlot(ax[1,1], full_score_df)
        fig.savefig(os.path.join(outputFolder, '#' + dataset_name + '_Overview.png' ))

if __name__ == '__main__':
    main()