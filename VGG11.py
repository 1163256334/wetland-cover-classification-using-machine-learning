import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
from tkinter import _flatten

starttime=time.time()

torch.manual_seed(1)    # reproducible

#GPU
if torch.cuda.is_available():
    print('GPU is available in this computer')
else:
    print('GPU is not available in this computer')

# Hyper Parameters
EPOCH = 30     # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001             # learning rate
DOWNLOAD= False
transform_train=torchvision.transforms.Compose(
    [
        torchvision.transforms.CenterCrop((32,32)),
        torchvision.transforms.RandomRotation((0,10)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
transform_test=torchvision.transforms.Compose(
    [
        torchvision.transforms.CenterCrop((32,32)),
        torchvision.transforms.RandomRotation((0,10)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

#define the class MyDataset

def default_loader(path):
    return Image.open(path).convert('RGB')
class MyDataset(Dataset):
  def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
    super(MyDataset, self).__init__()
    fh = open(txt, 'r')
    imgs = []
    for line in fh:
        line = line.strip('\n')
        line = line.rstrip('\n')
        words = line.split()
        imgs.append((words[0], int(words[1])))
    self.imgs = imgs
    self.transform = transform
    self.target_transform = target_transform
    self.loader = loader

  def __getitem__(self, index):
    fn, label = self.imgs[index]
    img = self.loader(fn)
    if self.transform is not None:
        img = self.transform(img)
    return img, label
  def __len__(self):
    return len(self.imgs)


#obtain the data
train_data = MyDataset(
    txt='./wetlandclass_data/Image/train.txt',
    transform=transform_train,    # Converts a PIL.Image or numpy.ndarray to
                                                # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
)
test_data = MyDataset(
    txt='./wetlandclass_data/image/test.txt',
    transform=transform_test,    # Converts a PIL.Image or numpy.ndarray to
                                                # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
)
print(len(train_data),'|',len(test_data))

plt.figure('train')
plt.imshow(train_data.loader(train_data.imgs[2][0]))
plt.title('%i' %train_data.imgs[2][1])
plt.xticks([])
plt.yticks([])
plt.figure('test')
plt.imshow(test_data.loader(test_data.imgs[2][0]))
plt.title('%i' %test_data.imgs[2][1])
plt.xticks([])
plt.yticks([])
# plt.show()
plt.draw()
plt.pause(2)
plt.close()

#load the data
train_loader=Data.DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
test_loader=Data.DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False)
(x,y)=next(iter(train_loader))
print(x.shape,y.shape,x.min(),x.max(),x[0][0].shape,y[0].shape)
plt.imshow(x[2][1])
plt.title(y[2].item())
# plt.show()
plt.draw()
plt.pause(2)
plt.close()

#!!!!!!!!!!!!!!!!!!!!!
#construction of vgg11
class vgg11(nn.Module):
    def __init__(self):
        super(vgg11, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),  # activation
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),

        )
        self.fc = nn.Sequential(
           nn.Linear(512 * 1 * 1, 4096),
           nn.ReLU(True),
           nn.Dropout(0.2),
           nn.Linear(4096, 4096),
           nn.ReLU(True),
           nn.Dropout(0.2),
           nn.Linear(4096, 8),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #return  x    # return x for visualization

#GPU!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print('------------model structure-------------')
cnn = vgg11.cuda()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR,weight_decay=0.001)
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.1)

#train cnn
print('------------start training--------------')
loss_count=[]
train_error=[]
test_error=[]
for epoch in range(EPOCH):
    for batch_idx, (x,y) in enumerate(train_loader):
        #print(x.shape, y.shape)
           #break
        #GPU!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        x=x.cuda()
        y=y.cuda()
        cnn.train()
        output = cnn(x)
        loss = loss_func(output, y)#print(loss.shape)is torch.Size([])
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        scheduler.step()
        #lossvalue
        loss.cpu()
        loss_count.append(loss.item())

        #error
        #train_error
        for a1, b1 in train_loader:
            # GPU!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            a1=a1.cuda()
            # NO Dropout!!!!!!!!!!!!!!!!!!!!
            cnn.eval()
            # CPU!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            out1 = cnn(a1).cpu()
            # print('test_out:\t',torch.max(out,1)[1])
            # print('test_y:\t',test_y)
            accuracy1 = torch.max(out1, 1)[1].numpy() == b1.numpy()
            cnn.train()
            break
        train_error.append(1-accuracy1.mean().item())

        #test_error
        for a2, b2 in test_loader:
            # GPU!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            a2=a2.cuda()
            # NO Dropout!!!!!!!!!!!!!!!!!!!!
            cnn.eval()
            # CPU!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            out2 = cnn(a2).cpu()
            # print('test_out:\t',torch.max(out,1)[1])
            # print('test_y:\t',test_y)
            accuracy2 = torch.max(out2, 1)[1].numpy() == b2.numpy()
            cnn.train()
            break
        test_error.append(1-accuracy2.mean().item())
        if batch_idx % 10==0:
            print('iteration:',epoch, batch_idx,loss.item())
            #save model
            torch.save(cnn, r'D:\Workingplace\pycharm\roige wetland cover mapping\saved cnnmodel\vgg11.cnn')
            print('train_acc: %.4f\t'% accuracy1.mean(),'|  test_acc:%.4f' % accuracy2.mean())
            print('LR:',optimizer.state_dict()['param_groups'][0]['lr'])
            print('=============================')


#plot the loss
plt.figure('CNN_Loss')
plt.plot(loss_count,color='red',label='Loss')
plt.xlabel('Step')
plt.ylabel('Loss value')
plt.legend(loc='upper right')
#plt.show()
plt.savefig('CNN_Loss.jpg',bbox_inches='tight')

#plot the error
plt.figure('Error')
plt.plot(train_error,color='blue',label='Train_error')
plt.plot(test_error,color='red',label='Test_error')
plt.xlabel('Step')
plt.ylabel('Error value')
plt.legend(loc='upper right')
plt.ylim((0,1.0))
new_ticks=np.linspace(0,1.0,6)
plt.yticks(new_ticks)
#plt.show()
plt.savefig('Error.jpg',bbox_inches='tight')


#validation of the cnn

print('------------test--------------')
cnnmodel = torch.load(r'D:\Workingplace\pycharm\roige wetland cover mapping\saved cnnmodel\vgg11.cnn').cpu()
# NO Dropout!!!!!!!!!!!!!!!!!!!!
cnnmodel.eval()
accuracy_sumt = []
pred_valuet=[]
real_valuet=[]
for i,(test_x,test_y) in enumerate(train_loader):
    # GPU!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # test_x=test_x.cuda()
    # CPU!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    out = cnnmodel(test_x)
    # print('test_out:\t',torch.max(out,1)[1])
    # print('test_y:\t',test_y)
    pred_valuet.append(torch.max(out,1)[1].numpy().tolist())
    real_valuet.append(test_y.numpy().tolist())
    accuracy = torch.max(out, 1)[1].numpy() == test_y.numpy()
    accuracy_sumt.append(accuracy.mean())
acc_train=sum(accuracy_sumt)/len(accuracy_sumt)
print('training acc\t',acc_train)
#np.savetxt('./output/pred_valuet.csv', np.array(_flatten(pred_valuet)),fmt="%d")
#np.savetxt('./output/real_valuet.csv', np.array(_flatten(real_valuet)),fmt="%d")


#test set
accuracy_sumv = []
pred_valuev=[]
real_valuev=[]
for i,(test_x,test_y) in enumerate(test_loader):
    # GPU!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # test_x = test_x.cuda()
    # CPU!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    out = cnnmodel(test_x)
    # print('test_out:\t',torch.max(out,1)[1])
    # print('test_y:\t',test_y)
    pred_valuev.append(torch.max(out,1)[1].numpy().tolist())
    real_valuev.append(test_y.numpy().tolist())
    accuracy = torch.max(out, 1)[1].numpy() == test_y.numpy()
    accuracy_sumv.append(accuracy.mean())
acc_test=sum(accuracy_sumv)/len(accuracy_sumv)
print('test acc\t',acc_test)

#np.savetxt('./output/pred_valuev.csv', np.array(_flatten(pred_valuev)),fmt="%d")
#np.savetxt('./output/real_valuev.csv', np.array(_flatten(real_valuev)),fmt="%d")


endtime=time.time()
print('running times',endtime-starttime,'s')

