import pandas as pd
from underthesea import word_tokenize, text_normalize
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils.config import Config

# Processing CSV file
df_train = pd.read_csv('/kaggle/working/train.csv')
df_test = pd.read_csv('/kaggle/working/test.csv')
df_dev = pd.read_csv('/kaggle/working/dev.csv')

df_train['Question'] = [word_tokenize(text_normalize(x), format='text') for x in df_train['Question']]
df_train['Answer'] = [word_tokenize(text_normalize(str(x)), format='text') for x in df_train['Answer']]

df_dev['Question'] = [word_tokenize(text_normalize(x), format='text') for x in df_dev['Question']]
df_dev['Answer'] = [word_tokenize(text_normalize(str(x)), format='text') for x in df_dev['Answer']]

df_test['Question'] = [word_tokenize(text_normalize(x), format='text') for x in df_test['Question']]

# Processing Datdset
class VLSP_Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anno_id, img_id, image_path, question, answer = self.data.iloc[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return anno_id, img_id, image, question, answer
    
transforms = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor()
                                ])


train_vlsp_dataset = VLSP_Dataset(df_train, transform=transforms)
test_vlsp_dataset = VLSP_Dataset(df_test, transform=transforms)
dev_vlsp_dataset = VLSP_Dataset(df_dev, transform=transforms)

train_loader = DataLoader(train_vlsp_dataset, batch_size=Config.TRAIN_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_vlsp_dataset, batch_size=Config.TEST_BATCH_SIZE, shuffle=False)
dev_loader = DataLoader(dev_vlsp_dataset, batch_size=Config.VAL_BATCH_SIZE, shuffle=False)

if __name__ == "__main__":
    random_indices = np.random.choice(len(train_vlsp_dataset), 10)

    for idx in random_indices:
        anno_id, img_id, image, question, answer = train_vlsp_dataset[idx]
        
        image = image.permute(1, 2, 0).numpy()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title("Question: " + question +"\n"+ "Answer:" + answer, fontsize=16)
        plt.axis('off')
        plt.show()