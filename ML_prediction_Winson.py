import pandas as pd
import numpy as np
import joblib
import os

import time
import datetime

import xlsxwriter

from draw_analysis import *
from heatmap_convert_off import *

limit_worksheet_files = [file for file in os.listdir('./limit_sheet/') if 'LimitsSheet' in file]
lm_ws = pd.read_excel(f'./limit_sheet/{limit_worksheet_files[0]}')
lm_ws.fillna(0)

SFR_spec = lm_ws[lm_ws['LOG'] == 'SC_SfrMedian_Sfr']
SFR_spec = SFR_spec[SFR_spec['header'].str.contains('sfr_circle_40p5cm_')]

files_val = [file for file in os.listdir('./validation')]

data_val = pd.DataFrame()
for file in files_val:
    df_heatmap = pd.read_csv(f'./validation/{file}')
    data_val = pd.concat([data_val, df_heatmap])
    data_val_copy = data_val.copy()
def date_today():
    current_time = datetime.datetime.now()
    string_today = str(current_time)
    date_today = string_today[8:10]
    month_today = string_today[5:7]
    year_today = string_today[0:4]
    wd = year_today + month_today + date_today
    return wd

def logPreProcess(df_heatmap_):
    print("[>] Preprocessing the log files")
    df_heatmap_ = df_heatmap_[df_heatmap_['time'] != 'time']
    df_heatmap_ = df_heatmap_.sort_values(by='time', ascending=False)
    df_heatmap_ = df_heatmap_.drop_duplicates(subset=['barcode'])
    df_heatmap_ = df_heatmap_.fillna(value='')
    df_heatmap_ = df_heatmap_.loc[:, ~df_heatmap_.columns.str.contains('^Unnamed')]
    return df_heatmap_
data_val = logPreProcess(data_val)
data_val_copy = logPreProcess(data_val_copy)
file_ROI = [file for file in os.listdir('./ROI/')]
for file in file_ROI:
    df_heatmap = pd.read_csv(f'./ROI/{file}')
    df_heatmap['list_ROI'].astype('int')
fields = [file.split('_')[2].replace(".csv", "") for file in file_ROI]

def field_ROI_rsptv():
    field_ROI = {}
    for num_file,ROIs in enumerate(file_ROI): 
        df_heatmap = pd.read_csv(f'./ROI/{ROIs}')
        list_ROI = df_heatmap['list_ROI'].to_list()
        field_ROI[fields[num_file]] = list_ROI
    return field_ROI
def change_name():
    field_ROI = field_ROI_rsptv()
    field_ny4_ROI = {}
    ny_ROI = []
    for field in fields:
        for ROI in field_ROI[field]:
            if ROI < 10:
                str_ROI = f'sfr_circle_ny4_ROI_00' + str(ROI)
            elif 10 <= int(ROI) <= 99:
                str_ROI = f'sfr_circle_ny4_ROI_0' + str(ROI)
            else:
                str_ROI = f'sfr_circle_ny4_ROI_' + str(ROI)
            ny_ROI.append(str_ROI)
        field_ny4_ROI[field] = ny_ROI
        ny_ROI = []
    return field_ny4_ROI

field_ny4_ROI = change_name()
field_ny8_ROI = field_ny4_ROI.copy()

for field in fields:
        list_new = [x.replace("ny4","ny8") for x in field_ny4_ROI[field]]
        field_ny8_ROI[field] = list_new

def Features_Insertion(df_heatmap, field_ROI, ny):
    df_heatmap_copy = df_heatmap.copy()
    for field in fields:
        values = df_heatmap_copy[field_ROI[field]].values
        row_num = values.shape[0]
        col_num = values.shape[1]
        idx_fail = {}
        list_tmp = []
        # df_heatmap_copy[f'ROI_{ny} FAIL {field}'] = df_heatmap_copy.index.map(idx_fail)
        df_heatmap_copy[f'sfr_circle_40p5cm_{ny}_{field}_min'] = df_heatmap_copy[field_ROI[field]].min(axis = 1)
        df_heatmap_copy[f'sfr_circle_40p5cm_{ny}_{field}_max'] = df_heatmap_copy[field_ROI[field]].max(axis = 1)
        df_heatmap_copy[f'MEAN_{ny}_{field}'] = df_heatmap_copy[field_ROI[field]].mean(axis = 1)
        df_heatmap_copy[f'sfr_circle_40p5cm_{ny}_{field}_range'] = df_heatmap_copy[f'sfr_circle_40p5cm_{ny}_{field}_max'] - df_heatmap_copy[f'sfr_circle_40p5cm_{ny}_{field}_min']
    return df_heatmap_copy

def Spec_define():
    
    lm_ws = pd.read_excel('./limit_sheet/SC23_MI_LimitsSheet_rev1_ERSrev2_20220706_modified.xlsx')
    lm_ws.fillna(0)

    SFR_spec = lm_ws[lm_ws['LOG'] == 'SC_SfrMedian_Sfr']
    SFR_spec = SFR_spec[SFR_spec['header'].str.contains('sfr_circle_40p5cm_')]
    SFR_spec = SFR_spec[~SFR_spec['Test Item'].str.contains('TB')]
    SFR_spec = SFR_spec[~SFR_spec['Test Item'].str.contains('LR')]
    # pd.to_numeric(SFR_spec)
    SFR_spec['PROD LSL'] = SFR_spec['PROD LSL'].fillna(0)
    SFR_spec['PROD USL'] = SFR_spec['PROD USL'].fillna(2)
    symptom = SFR_spec['Test Item']
    LSL = SFR_spec['PROD LSL']
    USL = SFR_spec['PROD USL']
    Spec_LSL = dict(zip(symptom, LSL))
    Spec_USL = dict(zip(symptom, USL))

    List_Symptom = list(Spec_LSL.keys())
    # List_Symptom_USL = list(Spec_USL.keys())
    List_Spec_LSL = list(Spec_LSL.values())
    List_Spec_USL = list(Spec_USL.values())
    return Spec_USL, List_Symptom, List_Spec_LSL, List_Spec_USL

def Spec_for_ROI():
    _, List_Symptom, List_Spec_LSL,_ = Spec_define()

    keys_for_slicing_ny4 = [x for x in List_Symptom if 'min' in x and 'ny4' in x]
    keys_for_slicing_ny8 = [x for x in List_Symptom if 'min' in x and 'ny8' in x]

    dict_spec = dict(zip(List_Symptom, List_Spec_LSL))

    sliced_spec_for_ROI_ny4 = {key: dict_spec[key] for key in keys_for_slicing_ny4}
    sliced_spec_for_ROI_ny8 = {key: dict_spec[key] for key in keys_for_slicing_ny8}

    fields_for_ROI = ["edge", "75F", "60F", "30F", 'cen']

    list_spec_ny4 = sorted(list(sliced_spec_for_ROI_ny4.values()))
    list_spec_ny8 = sorted(list(sliced_spec_for_ROI_ny8.values()))

    spec_ny4_ROI = dict(zip(fields_for_ROI, list_spec_ny4))
    spec_ny8_ROI = dict(zip(fields_for_ROI, list_spec_ny8))

    return spec_ny4_ROI, spec_ny8_ROI

def ROI_Insertion(df_heatmap, ny):
    df_heatmap.reset_index(inplace = True)
    spec_ny4_ROI, spec_ny8_ROI = Spec_for_ROI()
    for field in fields:
        if ny == 'ny4':
            values = df_heatmap[field_ny4_ROI[field]].values
            spec = spec_ny4_ROI[field]
        if ny == 'ny8':
            values = df_heatmap[field_ny8_ROI[field]].values
            spec = spec_ny4_ROI[field]
        row_num = values.shape[0]
        col_num = values.shape[1]
        idx_fail = {}
        idx_ROI_fail = {}
        list_tmp = []
        for i in range(row_num):
            for j in range(col_num):
                if values[i][j] <= spec:              
                    list_tmp.append(field_ny4_ROI[field][j].split("_")[-1])
                    all_fail_ROI = "/".join(list_tmp)
                    idx_ROI_fail[i] = all_fail_ROI
                    idx_fail[i] = list_tmp
            list_tmp = []
        df_heatmap[f'ROI_{ny} FAIL {field}'] = df_heatmap.index.map(idx_fail)
        df_heatmap[f'ROI {ny}_{field}_fail_write'] = df_heatmap.index.map(idx_ROI_fail)
    return df_heatmap
def Symptom_Insertion(feature, Symptom, LSL,USL):
        df_heatmap = feature.copy()
        df_heatmap.reset_index(inplace = True)
        list_tmp = []
        idx_fail = {}
        idx_symptom = {}
        symptom_val = df_heatmap[Symptom].values
        row_num = symptom_val.shape[0]
        col_num = symptom_val.shape[1]
        for i in range(row_num):
            for j in range(col_num):
                if symptom_val[i][j] <= LSL[j] or symptom_val[i][j] >= USL[j]:
                    symptom = Symptom[j].split("_")
                    list_tmp.append('_'.join(symptom[len(symptom)-3: len(symptom)]))
                    symptom_all = '/'.join(map(str, list_tmp))
                    # idx_fail[i] = list_tmp
                    idx_symptom[i] = symptom_all
            list_tmp = []
        # df_heatmap['fail_items'] = df_heatmap.index.map(idx_fail)
        df_heatmap['fail_items'] = df_heatmap.index.map(idx_symptom)
        return df_heatmap

def data_ROI_extract(df_heatmap):
    module_ID = ["barcode", "LensSN"]
    ROI_fail = [x for x in df_heatmap.columns if 'fail_write' in x]
    item_fail = ["fail_items"]
    headers_to_chose = module_ID + item_fail + ROI_fail
    df_heatmap = df_heatmap[headers_to_chose]
    return df_heatmap

def feature_extraction(ny):
    feature_g1 = [col for col in Feat_insert.columns if f'sfr_circle_{ny}_ROI_' in col] 
    feature_g2 = [col for col in Feat_insert.columns if 'max' in col or 'min' in col or 'range' in col or 'MEAN' in col]
    feature_ny_g2 = [col for col in feature_g2 if f'{ny}' in col]
    feature_ny_combined = feature_g1 + feature_ny_g2
    return feature_ny_combined


def Model_training():

    clf_svc = joblib.load('./model/SFR_Machine_Model.pkl')

    return clf_svc

def Copy_to_excel(df_heatmap, worksheet):
    for i,col in enumerate(df_heatmap.columns):
        worksheet.write(0,i,col)
    col_num = 0
    for col in df_heatmap.columns:
        row_num = 1
        df_heatmap_to_write = df_heatmap[col] 
        for data in df_heatmap_to_write:
            try:
                worksheet.write(row_num,col_num,data)
            except:
                pass
            row_num += 1   
        col_num += 1
    row_num = 1

def workbook_creation(df_heatmap_copy, df_heatmap_ROI):

    work_book = xlsxwriter.Workbook(f'./report/SFR_clf_{date_today()}_{round(time.time(),0)}.xlsx')
    prediction_ws = work_book.add_worksheet("Prediction")
    report_ws = work_book.add_worksheet("ROI_Report")
    Copy_to_excel(df_heatmap_copy, report_ws)
    ROI_ws = work_book.add_worksheet("ROI_Analysys")
    ROI_map = draw_map(df_heatmap_ROI)
    ROI_ws.insert_image(0,0,"", {"image_data": ROI_map})

    df_heatmap, name_ws = get_worksheet(data_val_copy)

    col_rp = 0
    df_heatmap_bc_rp = df_heatmap['barcode']
    df_heatmap_cg_rp = df_heatmap['categories']
    df_heatmap_for_copy = df_heatmap[['barcode', 'categories']]
    Copy_to_excel(df_heatmap_for_copy, prediction_ws)

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
    print('[>] Finish')
    work_book.close()

if __name__ == '__main__':
    Feat_insert_ny4 = Features_Insertion(data_val, field_ny4_ROI ,'ny4')
    Feat_insert = Features_Insertion(Feat_insert_ny4, field_ny8_ROI, 'ny8')
    feature_ny8_combined = feature_extraction('ny8')
    Feat_for_predict = Feat_insert[feature_ny8_combined]

    clf_svc = Model_training()
    data_val_copy['prediction'] = clf_svc.predict(Feat_for_predict.values)
    pred_barcode = dict(zip(data_val_copy['barcode'], data_val_copy['prediction']))
    

    Spec_USL, List_Symptom, List_Spec_LSL, List_Spec_USL = Spec_define()
    Symptom_added = Symptom_Insertion(Feat_insert, List_Symptom, List_Spec_LSL, List_Spec_USL)
    Symptom_barcode = dict(zip(Symptom_added['barcode'], Symptom_added['fail_items']))

    data_ROI = ROI_Insertion(data_val, 'ny4')
    data_ROI = ROI_Insertion(data_ROI, 'ny8')
    
    data_ROI['prediction'] = data_ROI['barcode'].map(pred_barcode)
    data_ROI['fail_items'] = data_ROI['barcode'].map(Symptom_barcode)

    data_ROI_copy = data_ROI.copy()
    data_ROI = data_ROI_extract(data_ROI)
    Feat_insert.to_csv('./data_ROI.csv')
    data_val.to_csv(f'./output/Final_Judgement_{date_today()}_{round(time.time(),0)}.csv')
    workbook_creation(data_ROI, data_ROI_copy)

    
