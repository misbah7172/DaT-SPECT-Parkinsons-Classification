from src.dataset import NiftiDataset

try:
    ds = NiftiDataset(csv_file='Dataset/train_labels.csv',
                      images_dir='Dataset', target_shape=(64, 64, 64))
    print('records:', len(ds))
    print('first:', ds.records[0])
except Exception as e:
    print('ERROR:', e)
