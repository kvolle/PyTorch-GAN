import torch
import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
from torchvision import datasets
#import singlefile

class NPZLoader(Dataset):
    def __init__(self, path, transform=None,shuffle=True):
        super(Dataset).__init__()
        #assert end > start
        self.path = path
        self.files = list(Path(path).glob('*.0.npz'))
        self.transform = transform
        #print("It worked")
        #print(self.files)
        #self.batchsize = batchsize
        self.singlefile = self.files[0]
        self.shuffle = shuffle
        self.numpy_array = np.load(str(self.singlefile))['arr_0'] * (1./128.) - 1.



    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        #numpy_array = np.load(str(self.singlefile))['arr_0']
        torch_array = torch.from_numpy(self.numpy_array[item,:,:,:]).permute((2, 0, 1))
        if self.transform is not None:
            torch_array = self.transform(torch_array)
        return torch_array,0
    
if __name__ == "__main__":
    datasets
    training_data= NPZLoader("./data/train/")
    
    test_data = NPZLoader("./data/test/")
    #print(test_data)
    #train_features, train_labels = next(iter(training_data))
    #print(f"Feature batch shape: {train_features.size()}")
    train_dataloader=DataLoader(training_data,batch_size=64,shuffle=True)
    #test_dataloader=DataLoader(test_data,batch_size=64,shuffle=True)
    #i = 0 
    #while i >= 0:
    #for i, data in enumerate(training_data):
    #    print("Data: ", data)
    img, _ = next(iter(train_dataloader))
    print(len(img))
    #print(img.shape)
    #img = training_data.__getitem__(i)
    #    i = i + 1
        #print(img)

    img_np = img[0].numpy()
    img_np = np.moveaxis(img_np,0,2)
    plt.figure()
    plt.imshow(img_np)
    plt.show()
    imgs = [img]
    #train_loader = DataLoader()


"""
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
"""