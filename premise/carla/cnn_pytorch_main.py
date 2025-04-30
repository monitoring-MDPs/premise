import pandas as pd
import numpy as np
import os
import torch
import torchvision

print("CUDA enabled:", torch.cuda.is_available())


class resNet(torch.nn.Module):
    """
    Use restnet from torchvision
    """

    def __init__(self, layers="18", pre_trained=False):
        super(resNet, self).__init__()
        if layers == "18":
            self.model = torchvision.models.resnet18(pretrained=pre_trained)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.model(x)


class convNet(torch.nn.Module):
    """
    CNN to train from scratch
    """

    def __init__(self):
        super(convNet, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2),
            # torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
        )

    def forward(self, x):
        h = self.model(x)
        return h


class CNN(torch.nn.Module):
    def __init__(self, resnet=False, pretrained=False):
        super(CNN, self).__init__()
        if resnet:
            self.model = resNet(pre_trained=pretrained)
        else:
            self.model = convNet()
        # self.head = torch.nn.Sequential(
        #     torch.nn.Flatten(),
        #     torch.nn.Linear(1024, 1)
        # )
        # self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(4096,1024)
        self.head = torch.nn.Linear(1024, 1)
        print(self.model)

    def forward(self, x):
        #x1 = self.model(x) #ANTONINA
        x1 = self.model(x)
        
        #x = torch.cat((x1),dim=1) #ANTONINA 
        x = self.fc1(x1)

        h = self.head(x)
        
        # if not self.training:
        #     return self.relu(self.head(h))
        return h


def train_cnn(
    train_dataset,
    val_dataset,
    save_path,
    batch_size=32,
    n_epochs=500,
    lr=0.0005,
    device="cuda",
    resnet=False,
    model_name="model",
):
    """
    Train a CNN on the given data
    """
    model = CNN(resnet=resnet).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.9, patience=10
    )
    criterion = torch.nn.MSELoss()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    best_val_loss = np.inf
    model.eval()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Number of parameters: {params}")
    with torch.no_grad():
        test_loss = 0
        for x_batch, y_batch in test_loader:
            y_batch = y_batch.to(device)
            y_pred = model(x_batch.to(device) )
            test_loss += criterion(y_pred, y_batch).item()
        test_loss /= len(test_loader)
    print(f"Epoch {-1}: val loss = {test_loss}")
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for loaded_batch in train_loader:
            x_batch, y_batch = loaded_batch
            optimizer.zero_grad()
            y_batch = y_batch.to(device)
            y_pred = model(x_batch.to(device))
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for loaded_batch in test_loader:
                x_batch, y_batch = loaded_batch
                y_batch = y_batch.to(device)
                y_pred = model(x_batch.to(device))
                test_loss += criterion(y_pred, y_batch).item()
            test_loss /= len(test_loader)
        scheduler.step(test_loss)
        print(
            f"Epoch {epoch}: train loss = {total_loss / len(train_loader)} val loss = {test_loss}"
        )
        # save model
        if test_loss < best_val_loss and epoch > 50:
            best_val_loss = test_loss
            torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}_{epoch}.pth"))
        if epoch % 50 == 0 or epoch == n_epochs-1:
            torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}_{epoch}.pth"))

    return model


from torch.utils.data import Dataset
from torchvision.io import read_image

def load_data_csv(
    data_dir,
    split_ratio=0.1,
    seed=0,
):
    """
    Load data and split into train validation
    """

    with open(os.path.join(data_dir, "data.csv")) as f:
        data = pd.read_csv(f)
    data_size = len(data)
    print(data_size)
    # split into train and test
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    train_sample = int(data_size * (1 - split_ratio))
    xtrain_image = data.loc[:train_sample]
    xval_image = data.loc[train_sample:]
    # save xtrain_image and xval_image to data_dir
    xtrain_image.to_csv(os.path.join(data_dir, "xtrain_image.csv"), index=False)
    xval_image.to_csv(os.path.join(data_dir, "xval_image.csv"), index=False)
    train_dataset = CustomImageDataset(
        annotations_file=os.path.join(data_dir, "xtrain_image.csv"),
        img_dir=os.path.join(data_dir, "imgs"),
    )
    val_dataset = CustomImageDataset(
        annotations_file=os.path.join(data_dir, "xval_image.csv"),
        img_dir=os.path.join(data_dir, "imgs"),
    )
    return train_dataset, val_dataset

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path) / 255  # squeeze into [0, 1]
        label = self.img_labels.iloc[idx, 1]

        return image, torch.FloatTensor([label])
    
    
class NPZDataset(Dataset):

    def __init__(self, input_file, transform=None):
        super().__init__()
        self.path = input_file
        self.transform = transform

        with np.load(self.path, allow_pickle=True) as fh:
            self.data = fh["arr_0"]
            self.labels = fh["arr_1"]
            #self.turn = fh["arr_1"][:,1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = (torch.from_numpy(self.data[index]) / 255).permute(2,0,1)[[2,1,0],:,:]
        
        #turn = self.turn[index]
        label = self.labels[index]
        
        #print(f"Sample {index} shape: {img.shape}")  #ANTONINA

        if self.transform:
            return self.transform(img), torch.FloatTensor([label])
        else:
            return img, torch.FloatTensor([label])
        
def load_data_np( 
    data_dir,
    split_ratio=0.1,
    seed=0,
):
    """
    Load data and split into train validation
    """
    # df_files = pd.read_csv(data_dir + "total_results.csv")
    # files = df_files["simulation"]
    paths = [data_dir + e for e in os.listdir(data_dir) if "npz" in e]
    ds = NPZDataset(paths[0])
    for i in range(1, len(paths)):
        ds = torch.utils.data.ConcatDataset([ds,NPZDataset(paths[i])])

    train_dataset, val_dataset = torch.utils.data.random_split(ds, [1-split_ratio, split_ratio])
    
    return train_dataset, val_dataset



Train = False
if __name__ == "__main__":
    if Train == True: 
        import argparse

        parser = argparse.ArgumentParser(description="Train CNN") 
        parser.add_argument("--data_dir", type=str, default="training_data/") #'/mimer/NOBACKUP/groups/naiss2024-22-404/cones_npz_datasets/' + f"/cones_npz_dataset_{i}.npz"
        parser.add_argument("--model_dir", type=str, default="models/") #/cephyr/users/skurka/Alvis/MODD/experiments/contextual_MODD/carla/lane_keeping/cones_cnn
        parser.add_argument("--model_name", type=str, default="model") #cones_model
        args = parser.parse_args()

        # build test.csv where is row is {i}.jpg
        import os

        os.makedirs(args.model_dir, exist_ok=True)
    
        train_dataset, val_dataset = load_data_np(args.data_dir)
        print("Total of: ", len(train_dataset) + len(val_dataset), " images")
        train_cnn(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            save_path=args.model_dir,
            resnet=False,
            model_name=args.model_name,
        )
    else:
        device="cuda"

        model = CNN().to(device)
        model_path = '/cephyr/users/skurka/Alvis/MODD/models/model_499.pth'
        model.load_state_dict(torch.load(model_path))
        model.eval()

        
        train_dataset, val_dataset = load_data_np('/mimer/NOBACKUP/groups/naiss2024-22-404/cones_npz_datasets/')
        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
     

        for x_batch, y_batch in test_loader:
            y_batch = y_batch.to(device)
            y_pred = model(x_batch.to(device) )
            print(x_batch.shape)
            print(y_pred, y_batch)



        

        








