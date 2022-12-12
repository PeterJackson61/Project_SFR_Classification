import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from io import BytesIO
import datetime

'''______Edit this part as you like______'''
DIRECTION = ['sag','tan']              
NYQUIST = ['ny4']         
ZRANGE = {'ny2': (0.0, 0.4),            
          'ny4': (0.1, 0.7),
          'ny8': (0.4, 0.9)}
COLUMN_NAME = 'sfr_circle_ny8_ROI_'     
'''______________________________________'''

circularCols = ['time', 'barcode', 'LensSN', 'result', 'fail_items']  
plotList = NYQUIST
plt.ioff()

def logPreProcess(df_):
    """
    Preprocess log file: remove header rows, sort, remove duplicates, fill N/A

    :param df_: (pd.DataFrame) log file dataframe
    :param by: (str or list of str) column to sort by
    :param ascending: (Bool or list of Bool) sort order for column in 'by' (ascending=True/False)
    :return: (pd.DataFrame) processed dataframe
    """
    df_ = df_[df_['GlobalTime'] != 'GlobalTime']
    df_ = df_.sort_values(by='GlobalTime', ascending=False)
    df_ = df_.drop_duplicates(subset=['barcode'])
    df_ = df_.fillna(value='')
    # barcode = df_['barcode']
    return df_

def Get_bc(df_):
    barcode_lens = df_[df_['prediction']== '0']
    barcode_tilt = df_[df_['prediction']== '-1']
    barcode_particle = df_[df_['prediction']== '1']
    return barcode_lens,barcode_tilt,barcode_particle

def getBarcodePlotData(barcode, dfAll):
    """
    Extract SFR data of 1 barcode from debugging log dataframe

    :param barcode: (str) a single barcode
    :param dfAll: (pd.DataFrame) processed debugging log dataframe (no dup. barcode)
    :return: (pd.DataFrame) dataframe contain nyquist scores, XY, actualDistance, direction
    """
    dfOne = dfAll[dfAll['barcode'] == barcode]  # get row of barcode only
    if dfOne.empty:  # if barcode not in dfAll
        print(f'[!] {barcode} not found in debugging log!')
    else:
        dfOne = dfOne.apply(pd.to_numeric, errors='coerce')  # convert numbers from str to numeric
        dfOne.reset_index(inplace=True)  # reset the index to use loc[0]
        rowArray = dfOne.loc[0, :].values.tolist()  # convert whole row to list so we can use slicing
        col1 = dfAll.columns.get_loc(COLUMN_NAME+'001') + 1  # get index of ROI_001 1st cell
        # reposition cells to columns in new dataframe
        df_ = pd.DataFrame(list(zip(rowArray[col1::11],          # nyquist8
                                    rowArray[col1 + 1::11],      # nyquist4
                                    rowArray[col1 + 2::11],      # nyquist2
                                    rowArray[col1 + 4::11],      # ROI X
                                    rowArray[col1 + 5::11],      # ROI Y
                                    rowArray[col1 + 7::11],      # actual distance
                                    rowArray[col1 + 10::11])),   # sag/tan direction
                           columns=['ny8', 'ny4', 'ny2', 'x', 'y', 'actualDist', 'direction'])
        # calculate XY distance to plot region contours (0.1, 0.3,...)
        df_['xDist'] = (df_['x'] - df_['x'].min()) / 2
        df_['yDist'] = (df_['y'] - df_['y'].min()) / 2
        df_['direction'] = df_['direction'].map({1: 'tan', 0: 'sag'})  # replace direction number by str to use easier
    return df_

def date_today():
    current_time = datetime.datetime.now()
    # yesterday = current_time - datetime.timedelta(days = )
    string_today = str(current_time)
    date_today = string_today[8:10]
    month_today = string_today[5:7]
    year_today = string_today[0:4]
    wd = year_today + month_today + date_today
    return wd

def plotHeatMap(i, barcode, dfBarcode, direction):
    """
    Plot SFR Heat map of barcode, save to output folder

    :param i: (int) counter from enumerate(barcodeList)
    :param barcode: (str) single barcode to plot
    :param dfBarcode: (pd.DataFrame) dataframe contain nyquist scores, XY, distances, direction
    :return: none
    """
    print(f'[{i}] Plotting {barcode}', end=' > ')
    for ny in plotList:
        print(f'{ny}_{direction}', end=' ')
        title = '_'.join((barcode, ny, direction))  # title of plot & also the name of image file
        zmin, zmax = ZRANGE[ny]  # range of heatmap
        df_ = dfBarcode[dfBarcode['direction'] == direction]  # select only sag/tan rows to plot
        # create meshgrid & interpolate missing data for heatmap & region contours
        xZ, yZ = np.meshgrid(np.arange(df_['x'].min(), df_['x'].max(), 2),
                             np.arange(df_['y'].min(), df_['y'].max(), 2), indexing='ij')
        pointsZ = np.vstack([df_['x'], df_['y']]).T
        gridZ = griddata(pointsZ, df_[ny], (xZ, yZ), method='linear')

        xD, yD = np.meshgrid(np.arange(df_['xDist'].min(), df_['xDist'].max(), 10),
                             np.arange(df_['yDist'].min(), df_['yDist'].max(), 10), indexing='ij')
        pointsD = np.vstack([df_['xDist'], df_['yDist']]).T
        gridDist = griddata(pointsD, df_['actualDist'], (xD, yD), method='linear')

        # start plotting
        # col = col_step()
        plt_data = BytesIO()
        fig = plt.figure(figsize=(4.5, 3), tight_layout=True)  # create plot of size 620x400
        ax = fig.add_subplot()
        ax.set_title(title, fontsize=12, loc='center')
        im = ax.imshow(gridZ.T, cmap='jet', vmin=zmin, vmax=zmax, aspect='auto')  # SFR heatmap
        cbar = fig.colorbar(im, ax=ax, fraction=0.1, format='%.2f')
        cbar.ax.tick_params(labelsize=12)# add color bar1
        # draw region contours and label them
        cont = ax.contour(xD, yD, gridDist, levels=[0.1, 0.3, 0.5, 0.6, 0.75, 0.85], colors='black', linewidths=1)
        ax.clabel(cont, inline=True, fontsize=12)
        # hide X, Y axises labels & ticks
        ax1 = plt.gca()
        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)
        # save plot to image and close plot
        fig.savefig(plt_data, format = 'png')
        plt_data.seek(0)
        plt.close()
    print('')
    return plt_data

def get_worksheet(df):

    clf_categorized = {-1: 'tilt', 0: 'lens', 1: 'particle'}
    df['categories'] = df['prediction'].map(clf_categorized)
    name_ws = df['categories'].unique()
    return df, name_ws

def Heatmap_maker(df_heatmap):

    df_heatmap, name_ws = get_worksheet(df_heatmap)
    col_rp = 0
    df_heatmap_bc_rp = df_heatmap['barcode']
    df_heatmap_cg_rp = df_heatmap['categories']

    for sheet in name_ws:
        work_sheet = work_book.add_worksheet(sheet)
        print('[>] Analyzing {}'.format(sheet))

        df_heatmap_bc = df_heatmap[df_heatmap['categories'] == sheet] 
        df_heatmap_bc_rp = df_heatmap_bc['barcode'] 
        df_heatmap_cg_rp = df_heatmap_bc['categories']
        if df_heatmap_bc.empty:
            print('[!] No barcode found in debugging log!')
        else:
            Counter = 1
            for ii, bc in enumerate(df_heatmap_bc_rp):
                df_heatmapBc = getBarcodePlotData(bc, df_heatmap)
                heatmap_sag = plotHeatMap(ii, bc, df_heatmapBc,'sag')
                work_sheet.insert_image(Counter,0,"", {"image_data": heatmap_sag})
                heatmap_tan = plotHeatMap(ii, bc, df_heatmapBc,'tan')
                work_sheet.insert_image(Counter,7,"", {"image_data": heatmap_tan})
                work_sheet.write(Counter+1, 14, bc)

                Counter += 15
            print(f'[>] Finish plotting in {time.time()-mid:.3f}s')
    work_book.close()
