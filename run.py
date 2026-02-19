import torch
from torch.autograd import Variable as V  # 保留但不再使用，避免大改
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
# 修复：补充StepLR导入（解决scheduler未定义）
from torch.optim.lr_scheduler import StepLR

tmp_dir = "E:/PyTorch/PyTorch/9/resnet50_places365.pth"
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import json

# 修复：清理路径前缀和分隔符，避免JSON/图像路径错误
def get_im_list(im_dir, file_path):
    im_list = []
    im_labels = []
    im_origin = []
    # 统一im_dir格式：确保末尾是/，兼容\和/
    im_dir = im_dir.rstrip('\\/') + '/'
    # 修复：指定编码，避免中文/特殊字符读取错误
    with open(file_path, 'r', encoding='utf-8') as fi:
        for line in fi:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            parts = line.split()
            if len(parts) < 2:  # 跳过格式错误行
                continue
            rel_path = parts[0]
            # 核心：清理路径，去掉images前缀、统一分隔符为/
            rel_path = rel_path.replace('\\', '/')
            if rel_path.lower().startswith('images/'):
                rel_path = rel_path[len('images/'):]
            # 拼接完整图像路径
            im_list.append(im_dir + rel_path)
            im_labels.append(int(parts[-1]))
            im_origin.append(rel_path)  # 保存清理后的路径
            array = line.split('/')
    return im_list, im_labels, im_origin

# 保留原有类别字典函数，仅修复编码
def sun397_sdict():
    category_path = 'E:/shujuji/MIT67/indoorCVPR_09/ClassNames.txt'
    sdict = {}
    with open(category_path, 'r', encoding='utf-8') as fi:
        for sid, line in enumerate(fi):
            sname = line.strip()
            sdict[sid] = sname
    return sdict
_sdict = sun397_sdict()

import os
import torch
# 保留原有模型导入逻辑
try:
    from torchvision.models import resnet50
except ImportError:
    def resnet50(num_classes=365):
        raise NotImplementedError("请替换为你自己的resnet50模型定义")

# -------------------------- 核心修复：模型加载+重写forward --------------------------
arch = 'resnet50'
model_file = "E:/PyTorch/PyTorch/9/resnet50_places365.pth"

# 检查模型文件
if not os.path.exists(model_file):
    raise FileNotFoundError(f"模型文件不存在！请检查路径：{model_file}")
if not os.access(model_file, os.R_OK):
    raise PermissionError(f"无读取模型权限：{model_file}")

# 初始化模型
model = resnet50(num_classes=365)

# 加载预训练权重
try:
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    else:
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict, strict=False)
    print(f"成功加载预训练权重：{model_file}")
except Exception as e:
    raise RuntimeError(f"加载权重失败：{e}")

# 修改最后一层为67类
model.fc = torch.nn.Linear(2048, 67)

# 关键修复：重写forward函数，返回(分类输出, 特征)，解决解包错误
def custom_forward(self, x):
    # ResNet原始特征提取流程
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    # avgpool层输出作为特征（record）
    feature = self.avgpool(x)
    feature = torch.flatten(feature, 1)  # 展平为2048维特征
    # 全连接层输出作为分类结果（output）
    output = self.fc(feature)
    
    # 返回两个值，匹配代码解包需求
    return output, feature

# 绑定自定义forward到模型
models.resnet.ResNet.forward = custom_forward.__get__(model, models.resnet.ResNet)

# 设置评估模式
model.eval()
print("模型加载完成，已切换为评估模式，最后一层输出类别数：67")

# -------------------------- 图像变换（保留原有修改） --------------------------
transform_train = trn.Compose([
        trn.Resize(256),  # 替换旧的Scale
        trn.RandomResizedCrop(224),  # 替换旧的RandomSizedCrop
        trn.RandomHorizontalFlip(),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_test = trn.Compose([
        trn.Resize(256),  # 替换旧的Scale
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 保留原有数据加载器
def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(data.Dataset):
    def __init__(self, images, labels, loader=default_loader, transform=None):
        self.images = images
        self.labels = labels
        self.loader = loader
        self.transform = transform
    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        img = self.loader(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    def __len__(self):
        return len(self.images)

# 数据路径配置（保留）
imdir = "E:/shujuji/MIT67/indoorCVPR_09/Images"
train_file = "E:/shujuji/MIT67/indoorCVPR_09/TrainImages.label"
test_file = "E:/shujuji/MIT67/indoorCVPR_09/TestImages.label"
train_list, train_labels, img_path = get_im_list(imdir, train_file)
test_list, test_labels, img_path_2 = get_im_list(imdir, test_file)
batch_size = 16
net = model
net.cuda()

# -------------------------- 训练集特征提取（修复JSON+ANTIALIAS+Variable） --------------------------
for i in range(0, len(train_list)):
    path = img_path[i]
    save = []
    print(path)
    # 生成正确的JSON文件名（无images前缀）
    json_name = (path.replace("/", "_")).replace(".jpg", ".json")
    # 拼接正确的JSON路径（类别子目录）
    class_name = path.split("/")[0] if "/" in path else "unknown"
    json_path = f"E:/shujuji/MIT67/indoorCVPR_09/annotated_json/{class_name}/{json_name}"
    
    # 修复：JSON文件异常处理
    try:
        f_train = open(json_path, 'r', encoding='utf-8')
        train_js = json.load(f_train)
        f_train.close()  # 显式关闭文件
    except FileNotFoundError:
        print(f"警告：JSON文件不存在 {json_path}，使用默认bbox")
        train_js = [{"classname":"unknown","bbox":[0,0,223,223],"score":1}]
    except json.JSONDecodeError:
        print(f"警告：JSON文件格式错误 {json_path}，使用默认bbox")
        train_js = [{"classname":"unknown","bbox":[0,0,223,223],"score":1}]
    
    if len(train_js) == 0:
        train_js.append({"classname":"unknown","bbox":[0,0,223,223],"score":1})
    
    for j in range(0, len(train_js)):
        data, target = train_list[i], train_labels[i]
        # 检查图像文件是否存在
        if not os.path.exists(data):
            print(f"错误：图像文件不存在 {data}，跳过")
            continue
        data = Image.open(data).convert('RGB')
        json_data = train_js[j]["bbox"]
        # 修复：替换ANTIALIAS为Resampling.LANCZOS（新版PIL）
        data = data.resize((224, 224), Image.Resampling.LANCZOS)
        print(json_data)
        data = data.crop([json_data[0], json_data[1], json_data[2], json_data[3]])
        data = data.resize((224, 224), Image.Resampling.LANCZOS)
        data = transform_test(data)
        newdata = torch.zeros(1, 3, 224, 224)
        newdata[0] = data
        # 修复：移除Variable，直接用cuda()（新版PyTorch）
        data = newdata.cuda()
        # 现在能正常解包为(output, record)
        output, record = net(data)
        data = record.cpu().detach().numpy()
        save.append(data)

    if not save:  # 避免空列表报错
        print(f"警告：无有效特征 {path}，跳过")
        continue
    
    # 特征平均
    data = save[0]
    for j in range(1, len(train_js)):
        data += save[j]
    data = data / len(train_js)
    
    # 修复：exist_ok=True避免重复创建目录报错
    root = f"E:/shujuji/MIT67/feature_extractor/loc_224_npy/{path.split('/')[0]}"
    os.makedirs(root, exist_ok=True)
    dir = f"E:/shujuji/MIT67/feature_extractor/loc_224_npy/{path.replace('.jpg','.npy')}"
    np.save(dir, data)
    print(f"训练集 {i} 处理完成，特征保存至：{dir}")

# -------------------------- 测试集特征提取（和训练集修复逻辑一致） --------------------------
for i in range(0, len(test_list)):
    path = img_path_2[i]
    save = []
    print(path)
    json_name = (path.replace("/", "_")).replace(".jpg", ".json")
    class_name = path.split("/")[0] if "/" in path else "unknown"
    json_path = f"E:/shujuji/MIT67/indoorCVPR_09/annotated_json/{class_name}/{json_name}"
    
    try:
        f_test = open(json_path, 'r', encoding='utf-8')
        test_js = json.load(f_test)
        f_test.close()
    except FileNotFoundError:
        print(f"警告：JSON文件不存在 {json_path}，使用默认bbox")
        test_js = [{"classname":"unknown","bbox":[0,0,223,223],"score":1}]
    except json.JSONDecodeError:
        print(f"警告：JSON文件格式错误 {json_path}，使用默认bbox")
        test_js = [{"classname":"unknown","bbox":[0,0,223,223],"score":1}]
    
    if len(test_js) == 0:
        test_js.append({"classname":"unknown","bbox":[0,0,223,223],"score":1})
    
    for j in range(0, len(test_js)):
        data, target = test_list[i], test_labels[i]
        if not os.path.exists(data):
            print(f"错误：图像文件不存在 {data}，跳过")
            continue
        data = Image.open(data).convert('RGB')
        json_data = test_js[j]["bbox"]
        data = data.resize((224, 224), Image.Resampling.LANCZOS)
        print(json_data)
        data = data.crop([json_data[0], json_data[1], json_data[2], json_data[3]])
        data = data.resize((224, 224), Image.Resampling.LANCZOS)
        data = transform_test(data)
        newdata = torch.zeros(1, 3, 224, 224)
        newdata[0] = data
        data = newdata.cuda()
        output, record = net(data)
        data = record.cpu().detach().numpy()
        save.append(data)

    if not save:
        print(f"警告：无有效特征 {path}，跳过")
        continue
    
    data = save[0]
    for j in range(1, len(test_js)):
        data += save[j]
    data = data / len(test_js)
    print(f"测试集 {i} 特征形状：{data.shape}")
    
    root = f"E:/shujuji/MIT67/feature_extractor/loc_224_npy/{path.split('/')[0]}"
    os.makedirs(root, exist_ok=True)
    dir = f"E:/shujuji/MIT67/feature_extractor/loc_224_npy/{path.replace('.jpg','.npy')}"
    np.save(dir, data)
    print(f"测试集 {i} 处理完成，特征保存至：{dir}")

    time.sleep(10)  # 保留原有延时，可按需注释

# -------------------------- 保留原有训练配置（未启用） --------------------------
print("模型结构：", net)
train_net = torch.nn.DataParallel(net, device_ids=[0])
optimizer = optim.SGD(params=train_net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
scheduler = StepLR(optimizer, 30, gamma=0.1)
trainer = Trainer(train_net, optimizer, F.cross_entropy, save_dir="E:/PyTorch/PyTorch/9/")
# trainer.loop(130, train_loader, test_loader, scheduler)  # 保留注释，需实现Trainer类