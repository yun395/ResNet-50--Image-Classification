import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import numpy as np
from mymodels import *
from PIL import Image
import torch.utils.data as data
from torch.utils.data import DataLoader
from utils import *
tmp_dir = "E:/PyTorch/PyTorch/9/resnet50_places365.pth"
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import json
def get_im_list(im_dir, file_path):
    im_list = []
    im_labels = []
    im_origin = []
    # 统一im_dir的格式：确保末尾是/，且转成/分隔
    im_dir = im_dir.rstrip('\\/') + '/'  # 先去掉末尾的\或/，再添加/
    with open(file_path, 'r') as fi:
        for line in fi:
            im_list.append(im_dir + line.split()[0])
            im_labels.append(int(line.split()[-1]))
            im_origin.append(line.split()[0])
            array = line.split('/')
    return im_list, im_labels, im_origin
def sun397_sdict():
    category_path = 'E:/shujuji/MIT67/indoorCVPR_09/ClassNames.txt'
    sdict = {}
    with open(category_path, 'r') as fi:
        for sid, line in enumerate(fi):
            sname = line.strip()
            sdict[sid] = sname
    return sdict
_sdict = sun397_sdict()


# arch = 'resnet50'
# # load the pre-trained weights
# model_file = '%s_places365.pth.tar' % arch
# if not os.access(model_file, os.W_OK):
#     weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
#     os.system('wget ' + weight_url)
# model = resnet50(num_classes=365)
# checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
# state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
# model.load_state_dict(state_dict)
# model.fc = torch.nn.Linear(2048,67)
# model.eval()

import os
import torch
# 确保你有resnet50的定义，这里假设是从合适的库导入
# 如果是从torchvision导入，需要调整num_classes参数的设置方式
try:
    from torchvision.models import resnet50
except ImportError:
    # 如果是自定义的resnet50实现
    def resnet50(num_classes=365):
        # 这里替换为你实际的resnet50定义
        raise NotImplementedError("请替换为你自己的resnet50模型定义")

# -------------------------- 核心修改部分 --------------------------
arch = 'resnet50'
# 1. 手动指定你下载好的模型文件路径（请替换为你的实际路径！）
# 例如：model_file = "/home/user/Downloads/resnet50_places365.pth"
model_file = "E:/PyTorch/PyTorch/9/resnet50_places365.pth"  # 如果文件在当前目录，直接写文件名

# 2. 检查文件是否存在（改为检查读权限，因为只需要加载文件）
if not os.path.exists(model_file):
    raise FileNotFoundError(f"模型文件不存在！请检查路径是否正确：{model_file}")
if not os.access(model_file, os.R_OK):
    raise PermissionError(f"没有读取模型文件的权限：{model_file}")

# 3. 初始化模型（num_classes=365对应预训练权重的类别数）
model = resnet50(num_classes=365)

# 4. 加载本地pth文件（适配手动下载的.pth格式）
try:
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    # 处理state_dict：移除可能存在的module.前缀（多GPU训练的权重会有这个前缀）
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # 如果是.tar格式的权重（原代码逻辑），读取state_dict字段
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    else:
        # 如果是直接保存的state_dict（.pth格式），直接使用
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint.items()}
    
    # 加载权重到模型
    model.load_state_dict(state_dict, strict=False)  # strict=False兼容少量层不匹配的情况
    print(f"成功加载预训练权重：{model_file}")
except Exception as e:
    raise RuntimeError(f"加载模型权重失败：{e}")

# 5. 修改最后一层全连接层，改为输出67类
model.fc = torch.nn.Linear(2048, 67)

# 6. 设置模型为评估模式
model.eval()
print("模型加载完成，已切换为评估模式，最后一层输出类别数：67")




"""
model = resnet50(num_classes=67)
pretrained = torch.load("/home/yyh/fineTune/mit67_place/model_epoch_30.pth").module
state_dict = pretrained.state_dict()
model.load_state_dict(state_dict)
model.eval()

"""





# load the image transformer
transform_train = trn.Compose([
        # trn.Scale(256),
        # trn.RandomSizedCrop(224),
        trn.Resize(256),  # 替换Scale为Resize
        trn.RandomResizedCrop(224),  # 替换RandomSizedCrop为RandomResizedCrop
        trn.RandomHorizontalFlip(),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_test = trn.Compose([
        #trn.Scale(256),
        trn.Resize(256),  # 替换Scale为Resize
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# load the class label

def default_loader(path):
    return Image.open(path).convert('RGB')
class MyDataset(data.Dataset):
    def __init__(self, images, labels,loader=default_loader,transform=None):
        self.images = images
        self.labels = labels
        self.loader = loader
        self.transform = transform
    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        #print(img)
        img = self.loader(img)
        if self.transform is not None:
            img = self.transform(img)
        #print(img)
        return img, target
    def __len__(self):
        return len(self.images)

imdir = "E:/shujuji/MIT67/indoorCVPR_09/Images"
train_file = "E:/shujuji/MIT67/indoorCVPR_09/TrainImages.label"
test_file = "E:/shujuji/MIT67/indoorCVPR_09/TestImages.label"
#train_file = test_file
train_list, train_labels,img_path= get_im_list(imdir, train_file)
test_list, test_labels ,img_path_2= get_im_list(imdir, test_file)
batch_size = 16
net = model
net.cuda()



#print(test_js)


for i in range(0, len(train_list)):
    path = img_path[i]
    save = []
    print(path)
    json_name = (path.replace("/", "_")).replace(".jpg", ".json")
    f_train = open("E:/shujuji/MIT67/indoorCVPR_09/annotated_json/" + json_name)
    train_js = json.load(f_train)
    if len(train_js) == 0:
        train_js.append( {"classname":"unknown","bbox":[0,0,223,223],"score":1})
    for j in range(0, len(train_js)):
        data, target = train_list[i], train_labels[i]
        data = Image.open(data).convert('RGB')
        json_data = train_js[j]["bbox"]
        data = data.resize((224, 224), Image.ANTIALIAS)
        print(json_data)
        data = data.crop([json_data[0], json_data[1], json_data[2], json_data[3]])
        data = data.resize((224, 224), Image.ANTIALIAS)
        data = transform_test(data)
        newdata = torch.zeros(1, 3, 224, 224)
        newdata[0] = data
        data = Variable(newdata).cuda()
        output, record = net(data)
        data = record.cpu().detach().numpy()
        save.append(data)

    data = save[0]
    for j in range(1, len(train_js)):
        data += save[j]
    data = data / len(train_js)
    # print(data)
    # target = Variable(target).cuda()

    # print(output)
    # print(output["avgpool"].cpu().shape)
    root = "E:/shujuji/MIT67/feature_extractor/loc_224_npy/" + path.split("/")[0]
    if not os.path.exists(root):
        os.makedirs(root)
    dir = "E:/shujuji/MIT67/feature_extractor/loc_224_npy/" + path.replace(".jpg",".npy")
    np.save(dir, data)
    print(i)


for i in range(0, len(test_list)):
    path = img_path_2[i]
    save = []
    print(path)
    json_name = (path.replace("/", "_")).replace(".jpg", ".json")
    f_test = open("E:/shujuji/MIT67/indoorCVPR_09/annotated_json/" + json_name)
    test_js = json.load(f_test)
    if len(test_js) == 0:
        test_js.append( {"classname":"unknown","bbox":[0,0,223,223],"score":1})
    for j in range(0, len(test_js)):
        data, target = test_list[i], test_labels[i]
        data = Image.open(data).convert('RGB')
        json_data = test_js[j]["bbox"]
        data = data.resize((224, 224), Image.ANTIALIAS)
        print(json_data)
        data = data.crop([json_data[0], json_data[1], json_data[2], json_data[3]])
        data = data.resize((224, 224), Image.ANTIALIAS)
        data = transform_test(data)
        newdata = torch.zeros(1, 3, 224, 224)
        newdata[0] = data
        data = Variable(newdata).cuda()
        output, record = net(data)
        data = record.cpu().detach().numpy()
        save.append(data)


    data = save[0]
    for j in range(1, len(test_js)):
        data += save[j]
    data = data / len(test_js)
    print(data)
    root = "E:/shujuji/MIT67/feature_extractor/loc_224_npy/" + path.split("/")[0]
    if not os.path.exists(root):
        os.makedirs(root)
    dir = "E:/shujuji/MIT67/feature_extractor/loc_224_npy/" + path.replace(".jpg",".npy")
    np.save(dir, data)
    print(i)

    time.sleep(10)








print(net)
train_net = torch.nn.DataParallel(net, device_ids=[0])
optimizer = optim.SGD(params=train_net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
scheduler = StepLR(optimizer, 30, gamma=0.1)
trainer = Trainer(train_net, optimizer, F.cross_entropy, save_dir="E:/PyTorch/PyTorch/9/")
#trainer.loop(130, train_loader, test_loader, scheduler)
