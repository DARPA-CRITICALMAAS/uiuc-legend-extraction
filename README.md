# TA1 Legend Extraction

## Description
This is the internal UIUC repo for the Legend Extraction portion of DARPA CMASS TA1 project. I'm still working on adding documentation to this readme and the notebooks but getting everything uploaded for right now.

## Installation
Currently this project is using opencv, matplotlib, pandas, tqdm and easyocr. If we want an enviroment setup script let me know.

## Key Functions

extractLegends(image, filename='')

extractLegends is the main function for generating predicted legends. It takes the map image and returns a json of the predictions in the USGS style. filename is an optional argument to add the img_filename that it was generated on which is consistent with USGS given jsons.

scoreLegendExtraction(image, truth_json, predict_json, outputDir=None)

scoreLegendExtraction is the main function for scoring and takes the map image the json of the truth legends and the json of predicted legends. It then scores these based on the label and intersection of the contours.

outputDir is an optional argument that controls weather the debugging plots are generated.
When the outputDir argument is set scoreLegendExtraction will save the scores of each map as well as an annotated version of the map with some helpful plots and cutouts of all of the missed and ocr failed polygon labels. The legendExtractionDemo adds on this by producing a full dataset score and overview files as well. The structure of this output is as shown below.
```bash
outputDir
├── #[sourceDir]_Overview.png
├── #[sourceDir]_Scores.csv
├── Map1
│   ├── #Map1_Overview.png
│   ├── #Map1_Scores.png
│   ├── Poly1.tif
│   ├── Poly2.tif...
│   └── PolyX.tif
├── Map2...
└── MapX
```

## USGS Json format
Going to write up what the format is and some commments but for now this is how its generated, which covers all of the fields.
```python
json_data = {
        'version' : '5.0.1',
        'flags' : {'source' :  'UIUC Exported'},
        'shapes' : [],
        'imagePath' : filename,
        'imagedata' : None,
        'imageHeight' : image_dims[0],
        'imageWidth' : image_dims[1]
    }

    for f in features:
        json_data['shapes'].append({
            'label' : '{}_poly'.format(f['label']),
            'points' : f['points'],
            'group_id' : None,
            'shape_type' : 'rectangle',
            'flags' : {}
        })
```

## Authors and acknowledgment
This repo was created by and currently maintained by Albert Bode.