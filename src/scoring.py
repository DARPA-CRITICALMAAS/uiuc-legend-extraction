import os
import cv2
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .visualizations import saveContour, mapOverviewPlot

log = logging.getLogger('DARPA_CMASS')

CORRECT_OVERLAP_THRESHOLD = 0.25
# Plot effects
BORDER_THICKNESS = 5
CORRECT_COLOR = (0,255,0) # Green
DIFF_COLOR = (47,255,173) # Green-Yellow
OCR_COLOR = (0,255,255) # Yellow
FAIL_COLOR = (0,0,255) # Red
MISS_COLOR = (255,0,255) # Fuchsia

# Create a mask of just the contour.
def createContourMask(img, contour):
    # Treat contours with only 2 points as a bounding box. The USGS Jsons describe the legends this way.
    if len(contour) == 2:
        cv2.rectangle(img, contour[0], contour[1], (255,255,255), cv2.FILLED)
    # All other contours are drawn normally.
    else:
        cv2.drawContours(img, np.expand_dims(contour, axis=0), 0, color=(255,255,255), thickness=cv2.FILLED)

# Scores two contours by there cross-sectional area
def scoreContourIntersection(contour1, contour2):
    # Calculate the mask size needed to encompass both contours
    min_xy = [min([*contour1, *contour2], key=lambda x: (x[0]))[0], min([*contour1, *contour2], key=lambda x: (x[1]))[1]]
    max_xy = [max([*contour1, *contour2], key=lambda x: (x[0]))[0], max([*contour1, *contour2], key=lambda x: (x[1]))[1]]

    mask_offset = min_xy
    height, width = max_xy[1] - min_xy[1], max_xy[0] - min_xy[0]

    # Create blank mask
    c1mask = np.zeros((height, width), dtype=np.uint8)
    c2mask = np.zeros((height, width), dtype=np.uint8)

    # Draw contour on mask
    createContourMask(c1mask, contour1 - mask_offset)
    createContourMask(c2mask, contour2 - mask_offset)

    # Binary AND masks to get intersecting area
    intersection = c1mask & c2mask
    union = c1mask | c2mask

    iouScore = np.count_nonzero(intersection)/np.count_nonzero(union)
    return iouScore

def scoreLegendExtraction(image, truth_json, predict_json, outputDir=None):
    org_dpi = plt.rcParams['figure.dpi']
    plt.rcParams['figure.dpi'] = 100
    mapname = os.path.splitext(os.path.basename(truth_json['imagePath'].replace('\\','/')))[0]
    if outputDir is not None:
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        #
        annotated_img = image.copy()

    # Score all predicted legends
    score_df = pd.DataFrame(columns = ['Map','Predicted Label','Matched Label','Contour Score','Class'])
    for predict in predict_json['shapes']:
        match_found = False
        for truth in truth_json['shapes']:
            labels_match = predict['label'] == truth['label']
            overlap_score = scoreContourIntersection(predict['points'], truth['points'])

            # Fully Correct
            if labels_match and overlap_score > CORRECT_OVERLAP_THRESHOLD:
                match_found = True
                match_result = {'Map' : mapname, 'Predicted Label' : predict['label'], 'Matched Label' : truth['label'], 'Contour Score' : overlap_score, 'Class' : 'c'}
                score_df.loc[len(score_df)] = match_result
                log.info(f'\t\t{match_result["Predicted Label"]} matched {match_result["Matched Label"]} : {match_result["Contour Score"]*100:.2f}%,  class={match_result["Class"]}')
                if outputDir is not None:
                    cv2.drawContours(annotated_img, [predict['points']], 0, CORRECT_COLOR, BORDER_THICKNESS)
                break

            # OCR matched but contour was in a different spot
            if labels_match:
                match_found = True
                match_result = {'Map' : mapname, 'Predicted Label' : predict['label'], 'Matched Label' : truth['label'], 'Contour Score' : overlap_score, 'Class' : 'd'}
                score_df.loc[len(score_df)] = match_result
                log.info(f'\t\t{match_result["Predicted Label"]} matched {match_result["Matched Label"]} : {match_result["Contour Score"]*100:.2f}%,  class={match_result["Class"]}')
                if outputDir is not None:
                    cv2.drawContours(annotated_img, [predict['points']], 0, DIFF_COLOR, BORDER_THICKNESS)
                break

            # Correct Contour but OCR Failed
            if overlap_score > CORRECT_OVERLAP_THRESHOLD:
                match_found = True
                match_result = {'Map' : mapname, 'Predicted Label' : predict['label'], 'Matched Label' : truth['label'], 'Contour Score' : overlap_score, 'Class' : 'o'}
                score_df.loc[len(score_df)] = match_result
                log.info(f'\t\t{match_result["Predicted Label"]} matched {match_result["Matched Label"]} : {match_result["Contour Score"]*100:.2f}%,  class={match_result["Class"]}')
                if outputDir is not None:
                    cv2.drawContours(annotated_img, [predict['points']], 0, OCR_COLOR, BORDER_THICKNESS)
                    saveContour(annotated_img, predict['points'], os.path.join(outputDir, predict['label'] + '.tif'))
                break

        # No Match
        if not match_found:
            match_result = {'Map' : mapname, 'Predicted Label' : predict['label'], 'Matched Label' : None, 'Contour Score' : 0.0, 'Class' : 'f'}
            score_df.loc[len(score_df)] = match_result
            log.info(f'\t\t{match_result["Predicted Label"]} did not match anything,  class={match_result["Class"]}')
            if outputDir is not None:
                cv2.drawContours(annotated_img, [predict['points']], 0, FAIL_COLOR, BORDER_THICKNESS)
                saveContour(annotated_img, predict['points'], os.path.join(outputDir, predict['label'] + '.tif'))

    # Check for any truth legends that were not matched
    for truth in truth_json['shapes']:
        if truth['label'] not in score_df['Matched Label'].unique():
            match_result = {'Map' : mapname, 'Predicted Label' : None, 'Matched Label' : truth['label'], 'Contour Score' : 0.0, 'Class' : 'm'}
            score_df.loc[len(score_df)] = match_result
            log.info(f'\t\tTrue Feature {match_result["Matched Label"]} was not matched,  class={match_result["Class"]}')
            if outputDir is not None:
                cv2.rectangle(annotated_img, truth['points'][0], truth['points'][1], MISS_COLOR, BORDER_THICKNESS)
                saveContour(annotated_img, truth['points'], os.path.join(outputDir, truth['label'] + '.tif'))

    plt.rcParams['figure.dpi'] = 750 # Need to up resolution to see the full map
    if outputDir is not None and len(score_df) > 1:
        # Generate Annotated Plot
        fig = mapOverviewPlot(annotated_img, score_df)
        fig.savefig(os.path.join(outputDir, '#' + mapname + '_Overview.png'))
        plt.close(fig)
        # Save csv
        score_df.to_csv(os.path.join(outputDir, '#' + mapname + '_Scores.csv'))

    # Revert to the orginal dpi 
    plt.rcParams['figure.dpi'] = org_dpi
    return score_df

### Decided it works better to just lower the overlap threshold then to try and remove the padding
### and set a higher threshold But keeping this function just in case we come back to that again.
# Removes any padding around the truth bounding box.
def removePadding(img, contour):
    min_xy = [min(contour, key=lambda x: (x[0]))[0], min(contour, key=lambda x: (x[1]))[1]]
    max_xy = [max(contour, key=lambda x: (x[0]))[0], max(contour, key=lambda x: (x[1]))[1]]

    mask = img[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0], :].copy()
    grayMask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binaryMask = cv2.threshold(grayMask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Flood fill borders
    height, width = binaryMask.shape[:2]
    corners = [[0,0],[0,height-1],[width-1, 0],[width-1, height-1]]
    for c in corners:
        cv2.floodFill(binaryMask, None, (c[0],c[1]), 0)

    x, y, w, h = cv2.boundingRect(cv2.findNonZero(binaryMask))
    return [[x,y],[x+w,y+h]] + contour[0]