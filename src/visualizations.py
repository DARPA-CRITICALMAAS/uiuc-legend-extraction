import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as fx

# Save an image of a predicted contour
def saveContour(image, contour, filename, pad=100):
    x, y, w, h = cv2.boundingRect(contour)
    x1, y1 = max(x-pad,0), max(y-pad,0)
    x2, y2 = min(x+w+pad, image.shape[1]), min(y+h+pad, image.shape[0])
    cv2.imwrite(filename, image[y1:y2, x1:x2, :])

def autoPctText(pct, data):
    absolute = int(np.round(pct/100.*np.sum(data)))
    return f"{absolute:d} : {pct:.1f}%"

def truthPiePlot(ax, score_df):
    # Generate Data from score_df
    vc = score_df['Class'].value_counts()
    missing = 0
    if 'm' in vc:
        missing = vc['m']
    matched = len(score_df['Matched Label'].unique()) - missing
    data = [matched, missing]
    labels = ['Matched\n{}, {:.1f}%'.format(matched, matched*100/sum(data)), 'Missing\n{}, {:.1f}%'.format(missing, missing*100/sum(data))]

    # Plot Donut chart
    ax.set_title('True\nLegends Matched', fontsize=8)
    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), colors=['royalblue','fuchsia']) # lightsteelblue might be okay
    ax.legend(wedges, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=4, fancybox=True, shadow=True)

    return ax

def predictPiePlot(ax, score_df):
    # Generate Data from score_df
    vc = score_df['Class'].value_counts()
    data = []
    for flag in ['c','d','o','f']:
        if flag in vc:
            data.append(vc[flag])
        else:
            data.append(0)
    # No Predictions
    if (sum(data) == 0):
        ax.axis('off')
        return ax
    
    annotations = ['{}, {:.1f}%'.format(c, c*100/sum(data)) for c in data]
    labels = ['Correct','Correct; Diff Contour', 'Correct; OCR Fail', 'Incorrect']

    # Plot Donut chart
    ax.set_title('Predicted\nLegend Results', fontsize=8)
    wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: autoPctText(pct, data), textprops=dict(fontsize=4, color='w'), wedgeprops=dict(width=0.5), colors=['green','greenyellow','yellow','red'])
    for text in autotexts:
        text.set_path_effects([fx.Stroke(linewidth=0.5, foreground='black'), fx.Normal()])
    ax.legend(wedges, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=4, fancybox=True, shadow=True)

    return ax

def mapOverviewPlot(image, score_df):
    plot_layout = [['Main','S1'],
                   ['Main','S2']]
    fig = plt.figure(layout="constrained")
    ax = fig.subplot_mosaic(plot_layout, width_ratios=[7,1], height_ratios=[1,1])

    # Show the annotated_img on the main plot
    ax['Main'].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax['Main'].axis('off')

    truthPiePlot(ax['S1'], score_df)
    predictPiePlot(ax['S2'], score_df)

    return fig

def datasetRecallPlot(ax, df):
    # Generate Data from df
    try :
        missing = df['Class'].value_counts()['m']
    except:
        missing = 0
    total_labels = 0
    for _, map_df in df.groupby('Map'):
        total_labels = total_labels + len(map_df['Matched Label'].unique())
    
    matched = total_labels - missing
    data = [matched, missing]
    labels = ['Matched\n{}, {:.1f}%'.format(matched, matched*100/sum(data)), 'Missing\n{}, {:.1f}%'.format(missing, missing*100/sum(data))]

    # Plot Donut chart
    ax.set_title('Recall : {:.2f}%'.format((matched/total_labels)*100))
    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), colors=['royalblue','fuchsia']) # lightsteelblue might be okay
    ax.legend(wedges, labels, loc='center right', bbox_to_anchor=(0.05, 0.5), fancybox=True, shadow=True)

    return ax

def datasetPrecisionPlot(ax, df):
    # Generate Data from df
    vc = df['Class'].value_counts()
    data = []
    for flag in ['c','d','o','f']:
        if flag in vc:
            data.append(vc[flag])
        else:
            data.append(0)
    annotations = ['{}, {:.1f}%'.format(c, c*100/sum(data)) for c in data]
    labels = ['Correct','Correct; Diff Contour', 'Correct; OCR Fail', 'Incorrect']
    precision = (sum(data[0:3]) / sum(data)) *100

    # Plot Donut chart
    ax.set_title('Precision : {:.2f}%'.format(precision))
    wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: autoPctText(pct, data), textprops=dict(color='w'), wedgeprops=dict(width=0.5), colors=['green','greenyellow','yellow','red'])
    for text in autotexts:
        text.set_path_effects([fx.Stroke(linewidth=2, foreground='black'), fx.Normal()])
    ax.legend(wedges, labels, loc='center left', bbox_to_anchor=(0.95, 0.5), fancybox=True, shadow=True)

    return ax

def averageMapRecallPlot(ax, df):
    # Generate Data from df
    recalls = []
    for _, map_df in df.groupby('Map'):
        missing = 0
        try :
            missing = map_df['Class'].value_counts()['m']
        except:
            pass
        matched = len(map_df['Matched Label'].unique()) - missing
        recalls.append(matched/(matched+missing))
    
    avg_recall = sum(recalls)/len(recalls)
    data = [avg_recall,1-avg_recall]

    # Plot Donut chart
    ax.set_title('Average Map Recall : {:.2f}%'.format(avg_recall*100))
    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), colors=['royalblue','fuchsia']) # lightsteelblue might be okay

    return ax

def averageMapPrecisionPlot(ax, df):
    # Generate Data from score_df
    precisions = []
    for map, map_df in df.groupby('Map'):
        if 'geo_mosaic' in map:
            continue
        vc = map_df['Class'].value_counts()
        data = []
        for flag in ['c','d','o','f']:
            if flag in vc:
                data.append(vc[flag])
            else:
                data.append(0)
        if (sum(data) == 0):
            continue
        precisions.append(sum(data[0:3])/sum(data))

    avg_precision = sum(precisions)/len(precisions)
    data = [avg_precision,1-avg_precision]

    # Plot Donut chart
    ax.set_title('Average Map Precision : {:.2f}%'.format(avg_precision*100))
    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), colors=['green','red']) # lightsteelblue might be okay

    return ax