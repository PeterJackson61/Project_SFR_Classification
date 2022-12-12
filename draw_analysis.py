import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
import matplotlib.collections as mc
from io import BytesIO

def draw_map(df):
    fail_field = [col for col in df.columns if 'FAIL' in col or 'cirlce_x' in col or 'circle_y' in col and '_write' not in col]
    df_lens = df[df['prediction'] == 0]

    bunk_ROI_lens = df_lens[fail_field].values
    bunk_ROI_lens = np.reshape(bunk_ROI_lens,-1)
    counts = defaultdict(int)

    sum_num = 0
    for i in bunk_ROI_lens:
        try:
            for j in i:
                counts[(j)] +=1
        except:
            pass
    fail_num = df_lens.shape[0]
    res = Counter({key : round((counts[key] / fail_num *10000),0) for key in counts})
    x = []
    y = []
    for i in res.keys():

        x.append(int(float(df_lens[f'sfr_circle_x_ROI_{i}'][1:2])))
        y.append(int(float(df_lens[f'sfr_circle_y_ROI_{i}'][1:2])))
    len(x), len(y)

    sizes = np.array(list(res.values()))
    xy = np.array(list(zip(x,y)))
    ROI = list(res.keys())

    plt.rcParams["figure.figsize"] = [16, 12]
    plt.rcParams["figure.autolayout"] = True

    patches = [plt.Circle(center, size) for center, size in zip(xy, sizes)]
    fig, ax = plt.subplots()
    ax.invert_yaxis()
    collection = mc.CircleCollection(sizes, offsets=xy, transOffset=ax.transData, color='red', alpha = 0.5)
    ax.add_collection(collection)

    for label_ROI,xy in zip(ROI,xy):
        plt.annotate(f"{label_ROI}", xy=xy, ha= 'center', va = 'bottom')
    plt.title("ROI Failure Map", size = 18)
    plt.xlim([0, 4032])
    plt.ylim([3024,0])
    ax.margins(0.01)
    ROI_ana = BytesIO()
    plt.savefig(ROI_ana, format = 'png')
    ROI_ana.seek(0)
    plt.close()

    return ROI_ana
