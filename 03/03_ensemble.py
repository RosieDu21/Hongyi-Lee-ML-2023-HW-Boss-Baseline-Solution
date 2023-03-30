# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
# This is for the progress bar.
from tqdm.auto import tqdm

_exp_name = "ensemble_large"

base_dir = './'

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
n_epochs = 5
patience = 10
warmup = 0
start_factor = 0.01
end_factor = 1.0
lr = 0.001
lr_decay = 0.9
weight_decay = 1e-4
dropout = 0.5
clip_grad = 1.0
input_dim = 11*4
output_dim = 11
num_layers = 5
hidden_dim = 256


myseed = 6666  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


tfms = [
    transforms.RandomResizedCrop((224,224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4))
]
# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((224,224)),
    # You may add some transforms here.
    #transforms.RandomGrayscale(0.1),
    *tfms,
    #transforms.RandomInvert(0.1),
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor()
])


vgg = torchvision.models.vgg19_bn(num_classes=11,dropout=dropout)
vgg.load_state_dict(torch.load(os.path.join(base_dir,'vgg19_bn_lbsmth_best.ckpt')))
vgg.to(device)
vgg.eval()

resnext = torchvision.models.resnext101_64x4d(num_classes=11)
resnext.load_state_dict(torch.load(os.path.join(base_dir,'resnext101_64x4d_lbsmth_best.ckpt')))
resnext.to(device)
resnext.eval()

resnet = torchvision.models.resnet152(num_classes=11)
resnet.load_state_dict(torch.load(os.path.join(base_dir,'resnet152_lbsmth_best.ckpt')))
resnet.to(device)
resnet.eval()

shufflenet = torchvision.models.shufflenet_v2_x2_0(num_classes=11)
shufflenet.load_state_dict(torch.load(os.path.join(base_dir,'shufflenet_v2_x2_0_lbsmth_best.ckpt')))
shufflenet.to(device)
shufflenet.eval()


class BasicBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(BasicBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.layer(x)


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim, dropout):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            *[BasicBlock(hidden_dim,dropout) for _ in range(num_layers)],
            nn.Linear(hidden_dim,output_dim)
        )

    def _forward_impl(self, x):
        with torch.no_grad():
            x1 = vgg(x)
            x2 = resnext(x)
            x3 = resnet(x)
            x4 = shufflenet(x)
            votes = torch.cat((x1,x2,x3,x4),dim=-1)
        return self.net(votes)

    def _transform(self, x, tfm):
        return torch.stack([tfm(y.squeeze(dim=0)) for y in x.split(1)],dim=0)

    def forward(self, x):
        if self.training:
            return self._forward_impl(x)
        else:
            ave_logits = torch.stack([self._forward_impl(self._transform(x,tfm)) for tfm in tfms], dim=0).mean(dim=0)
            return self._forward_impl(x)*0.8 + ave_logits*0.2


class FoodDataset(Dataset):

    def __init__(self, path, tfm=test_tfm, files=None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files

        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)

        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1  # test has no label

        return im, label


# Construct train and valid datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = FoodDataset(os.path.join(base_dir,"train"), tfm=train_tfm)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_set = FoodDataset(os.path.join(base_dir,"valid"), tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_acc = 0

# Initialize a model, and put it on the device specified.
model = Classifier(input_dim,output_dim,num_layers,hidden_dim,dropout).to(device)

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def lr_lambda(epoch):
    if epoch<warmup:
        return start_factor + (end_factor-start_factor)*epoch/warmup
    else:
        return lr_decay ** (epoch-warmup)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

for epoch in range(n_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    print(f'Current lr:{scheduler.get_last_lr()[0]}')
    for batch in tqdm(train_loader):
        # A batch consists of image data and corresponding labels.
        ims, labels = batch
        # imgs = imgs.half()
        # print(imgs.shape,labels.shape)

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(ims.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)

    scheduler.step()

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):
        # A batch consists of image data and corresponding labels.
        ims, labels = batch
        # imgs = imgs.half()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(ims.to(device))


        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)
        # break

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    # update logs
    if valid_acc > best_acc:
        with open(f"./{_exp_name}_log.txt", "a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
    else:
        with open(f"./{_exp_name}_log.txt", "a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    # save models
    if valid_acc > best_acc:
        print(f"Best model found at epoch {epoch+1}, saving model")
        torch.save(model.state_dict(), os.path.join(base_dir, f"{_exp_name}_best.ckpt"))  # only save best to prevent output memory exceed error
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break

# Construct test datasets.
# The argument "loader" tells how torchvision reads the data.
test_set = FoodDataset(os.path.join(base_dir,"test"), tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

model_best = Classifier(input_dim,output_dim,num_layers,hidden_dim,dropout).to(device)
model_best.load_state_dict(torch.load(os.path.join(base_dir,f"{_exp_name}_best.ckpt")))
model_best.eval()
prediction = []
with torch.no_grad():
    for data,_ in tqdm(test_loader):
        test_pred = model_best(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()


# create test csv
def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(len(test_set))]
df["Category"] = prediction
df.to_csv(os.path.join(base_dir,_exp_name+"_submission.csv"),index = False)