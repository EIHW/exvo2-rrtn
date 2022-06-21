import audiofile
import numpy as np
import pandas as pd
import torch


class CachedDataset(torch.utils.data.Dataset):
    r"""Dataset of cached features.

    Args:
        df: partition dataframe containing labels
        features: dataframe with paths to features
        target_column: column to find labels in (in df)
        transform: function used to process features
        target_transform: function used to process labels
    """

    def __init__(
            self,
            df: pd.DataFrame,
            features: pd.DataFrame,
            target_column: str,
            transform=None,
            target_transform=None,
    ):
        self.df = df
        self.features = features
        self.target_column = target_column
        self.transform = transform
        self.target_transform = target_transform
        self.indices = list(self.df.index)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        index = self.indices[item]
        signal = np.load(self.features.loc[index, 'features'] + '.npy')
        target = self.df[self.target_column].loc[index]
        if signal.shape[0] == 2:
            signal = signal.mean(0, keepdims=True)
        if isinstance(self.target_column, list) and len(self.target_column) > 1:
            target = np.array(target.values)

        if self.transform is not None:
            signal = self.transform(signal)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return signal, target.astype(np.float32)


class CCCLoss(torch.nn.Module):
    def forward(self, output, target):
        out_mean = torch.mean(output)
        target_mean = torch.mean(target)

        covariance = torch.mean((output - out_mean) * (target - target_mean))
        target_var = torch.mean((target - target_mean) ** 2)
        out_var = torch.mean((output - out_mean) ** 2)

        ccc = 2.0 * covariance / \
              (target_var + out_var + (target_mean - out_mean) ** 2 + 1e-10)
        loss_ccc = 1.0 - ccc

        return loss_ccc


if __name__ == '__main__':
    import os
    import audtorch

    EMOTIONS = [
        'Awe',
        'Excitement',
        'Amusement',
        'Awkwardness',
        'Fear',
        'Horror',
        'Distress',
        'Triumph',
        'Sadness',
        'Surprise'
    ]

    target_column = EMOTIONS

    data_root = '/home/xinjing/Documents/gpu5/nas/staff/data_work/Meishu/0_ExVo22/baseline'
    features_path = '/home/xinjing/Documents/gpu5/nas/staff/data_work/Sure/1_Xin/exvo/features/features.csv'

    df = pd.read_csv(os.path.join(data_root, 'data_info.csv'))
    df['file'] = df['File_ID'].apply(lambda x: x.strip('[').strip(']') + '.wav')
    df.set_index('file', inplace=True)
    df_train = df.loc[df['Split'] == 'Train']
    df_dev = df.loc[df['Split'] == 'Val']
    df_test = df.loc[df['Split'] == 'Val']

    features = pd.read_csv(features_path).set_index('file')
    features['features'] = features['features'].apply(
        lambda x: os.path.join(os.path.dirname(features_path), os.path.basename(x)))

    db_args = {
        'features': features,
        'target_column': target_column,
        'transform': audtorch.transforms.RandomCrop(250, axis=-2),
    }

    train_dataset = CachedDataset(
        df_train,
        **db_args
    )
