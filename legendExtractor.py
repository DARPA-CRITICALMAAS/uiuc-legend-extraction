"""
Usage:
    $ python legendExtraction.py --data --outDir --maskDir --truthDir
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

# Internal Project files
import src.logging_utils as log_utils
import src.visualizations as my_viz
import src.IO as io
from src.extraction import extractLegends
from src.scoring import scoreLegendExtraction

log = logging.getLogger('DARPA_CMASS')

def parse_command_line():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('data', help='Path to the directory containing the images to perform legend extraction on', type=str)
    parser.add_argument('-o','--outputDir', default='legend_results', help='Path to the directory that the resulting json will be written to. If automatic scoring or debugging is enabled there respective outputs will be written here as well.', type=str)
    parser.add_argument('-m','--maskDir', default=None, help='Path to a directory containing json files to be used for masking out the legend area.', type=str)
    parser.add_argument('-t','--truthDir',default=None, help='Path to a directory containing json files to be used as truth values in automatic scoring.', type=str)
    parser.add_argument('--debug', help='Flag to set enable debug statements in the log.', action='store_true')

    return parser.parse_args()

def main():
    # Get command line args
    args = parse_command_line()

    # Start logger
    logpath = os.path.join('logs', 'Latest.log')
    loglvl = logging.INFO if not args.debug else logging.DEBUG
    log_utils.start_logger(logpath, loglvl)
    
    # Create Output Directory
    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir)

    log.info('Running legend extraction on "{}"'.format(args.data))
    dataset_name = os.path.basename(args.data.replace('\\','/'))

    if args.truthDir is not None:
        score_df = pd.DataFrame()
    finished, failed = 0, 0
    # Iterate through each file in the input directory
    pbar = tqdm([f for f in os.listdir(args.data)])
    for file in pbar:
        pbar.set_description('Processing {}'.format(file))
        pbar.refresh()
        log.info('Processing {}'.format(file))
        mapname = os.path.splitext(file)[0]

        # Load img
        img = io.safeLoadImg(os.path.join(args.data, file))
        if img is None:
            failed = failed + 1
            continue
        log.info('\tLoaded {} image'.format(file))
        
        # Load mask
        mask_contour = None
        if args.maskDir is not None:
            mask_json = io.loadUnchartedJson(os.path.join(args.maskDir, (mapname + '.json')))
            if mask_json is not None and 'legend_polygons' in mask_json:
                mask_contour = mask_json['legend_polygons']['bounds']
                log.info('\tApplying Legend Area Mask from {}'.format(mapname + '.json'))

        # Generate Predictions
        log.info('\tExtracting Legends')
        legend_predictions = extractLegends(img, legendcontour=mask_contour)
        log.info('\tPredicted {} polygon legend features'.format(len(legend_predictions)))
  
        predict_json = io.generateJsonData(legend_predictions, filename=file, img_dims=img.shape, force_rectangle=True)

        # Score Predictions
        if args.truthDir is not None:
            truth_json = io.loadUSGSJson(os.path.join(args.truthDir, (mapname + '.json')), polyDataOnly=True)
            if truth_json is not None:
                map_score = scoreLegendExtraction(img, truth_json, predict_json, outputDir=os.path.join(args.outputDir, mapname))

        # Save json
        for s in predict_json['shapes']:
            s['points'] = s['points'].tolist()
        with open(os.path.join(args.outputDir, mapname + '_predict.json'), 'w') as fh:
            fh.write(json.dumps(predict_json))
        score_df = pd.concat([score_df, map_score])
        finished = finished + 1
    
    log.info('Finished extraction on {} maps. {} maps failed'.format(finished, failed))
    score_df.to_csv(os.path.join(args.outputDir, '#' + dataset_name + '_Scores.csv'))

    if args.outputDir is not None:
        fig, ax = plt.subplots(2,2, figsize=(8,8))
        fig.suptitle(dataset_name + ' Dataset Results')
        my_viz.datasetRecallPlot(ax[0,0], score_df)
        my_viz.datasetPrecisionPlot(ax[0,1], score_df)
        my_viz.averageMapRecallPlot(ax[1,0], score_df)
        my_viz.averageMapPrecisionPlot(ax[1,1], score_df)
        fig.savefig(os.path.join(args.outputDir, '#' + dataset_name + '_Overview.png' ))
        plt.close(fig)

if __name__ == '__main__':
    main()