class FeatureDataset(Dataset):
    def __init__(self, dataframe, scaler):
        # dataframe must include 'label'
        drop_cols = [c for c in ['label', 'filename', 'video_id'] if c in dataframe.columns]
        features = dataframe.drop(columns=drop_cols).values.astype('float32')
        # Apply same scaling as fitted on training data
        self.X = scaler.transform(features).astype('float32')
        self.y = dataframe['label'].values.astype('int64')
        self.len = len(self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y


def get_video_id(fname):
    base = str(fname).split('/')[-1]
    base = base.replace('.jpg','').replace('.png','')
    if "_frame" in base:
        return base.rsplit("_frame", 1)[0]
    return base
