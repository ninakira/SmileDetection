import numpy as np
import pandas as pd
import shutil

data_train = pd.read_csv('/data/affectnet_extracted/OneDrive-2021-05-01/training.csv')

data_validation = pd.read_csv('/data/affectnet_extracted/OneDrive-2021-05-01//validation.csv', names=data_train.columns)

data_train = data_train.drop(['face_x', 'face_y', 'face_width',
       'face_height', 'facial_landmarks', 'valence', 'arousal'], axis=1)

data_validation = data_validation.drop(['face_x', 'face_y', 'face_width',
       'face_height', 'facial_landmarks', 'valence', 'arousal'],axis=1)

data = pd.concat([data_train, data_validation])

data = data.rename(columns={'subDirectory_filePath': 'path', 'expression': 'target'}, inplace=False)

data['target'] = np.where(data.target == 1,  1, 0)

for index, row in data.iterrows():
    orig_path = '/data/affectnet_extracted/{}'
    name = row['path'].split('/')[1]
    dest = '/data/affectnet_01/{}'.format(row["target"])
    # print("orig", orig_path.format(row['path']))
    # print("dest", f'{dest}/{name}')
    shutil.copy(orig_path.format(row['path']), f'{dest}/{name}')

print("DONE!!!!!")

