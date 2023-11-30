import os
import cv2
import json
import logging
import numpy as np

log = logging.getLogger('DARPA_CMASS')

# Load a Uncharted formated json file (For legend area mask)
def loadUnchartedJson(filepath):
    if not os.path.exists(filepath):
        log.warning('Json mask file "{}" does not exist. Skipping file'.format(filepath))
        return None
    
    with open(filepath, 'r') as fh:
        json_data = json.load(fh)

    formated_json = {}
    for section in json_data:
        # Convert pix coords to correct format
        section['bounds'] = np.array(section['bounds']).astype(int)
        formated_json[section['name']] = section
        
    return formated_json

# Load a USGS formated json file (For truth jsons)
def loadUSGSJson(filepath, polyDataOnly=False):
    if not os.path.exists(filepath):
        return None
    
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

def generateJsonData(features, filename='', force_rectangle=False, img_dims=None):
    json_data = {
        'version' : '5.0.1',
        'flags' : {'source' :  'UIUC Exported'},
        'shapes' : [],
        'imagePath' : filename,
        'imageData' : None,
        'imageHeight' : None if img_dims is None else img_dims[0],
        'imageWidth' : None if img_dims is None else img_dims[1]
    }

    for f in features:
        if force_rectangle:
            min_xy = [min(f['points'], key=lambda x: (x[0]))[0], min(f['points'], key=lambda x: (x[1]))[1]]
            max_xy = [max(f['points'], key=lambda x: (x[0]))[0], max(f['points'], key=lambda x: (x[1]))[1]]
            f['points'] = np.array([min_xy, max_xy])
        # Determine Shape Type based on amount of points
        shape_type = 'unknown'
        if len(f['points']) == 2:
            shape_type = 'rectangle'
        elif len(f['points']) == 3:
            shape_type = 'triangle'
        elif len(f['points']) == 4:
            shape_type = 'quadrilateral'

        json_data['shapes'].append({
            'label' : '{}_poly'.format(f['label']),
            'points' : f['points'],
            'group_id' : None,
            'shape_type' : shape_type,
            'flags' : {}
        })

    return json_data