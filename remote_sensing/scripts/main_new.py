import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2 
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from glob import glob
from torchvision.transforms import functional as F
from PIL import Image


def random_zero_out(image, keep_ratio=0.05):
    """
    将输入的彩色图像的95%区域随机置零，保留5%区域的颜色不变。

    参数:
    image (numpy.ndarray): 输入的彩色图像，形状为 (3, height, width)。
    keep_ratio (float): 保留的比例，默认为0.05，即5%。

    返回:
    numpy.ndarray: 处理后的图像。
    """
    # 确保输入图像是numpy数组
    if not isinstance(image, np.ndarray):
        raise ValueError("输入的图像必须是numpy数组")

    # 获取图像的形状
    channels, height, width = image.shape

    # 计算需要保留的像素数量
    total_pixels = height * width
    keep_pixels = int(total_pixels * keep_ratio)

    # 创建一个全零的掩码
    mask = np.zeros((height, width), dtype=bool)

    # 随机选择一些位置保留
    indices = np.random.choice(total_pixels, keep_pixels, replace=False)
    mask.ravel()[indices] = True
    mask_3chn = np.repeat(mask[np.newaxis, :, :], channels, axis=0)
    masked_image = image * mask_3chn
    # # 创建一个全零的图像
    # output_image = np.zeros_like(image)

    # # 将原始图像的保留部分复制到输出图像
    # for c in range(channels):
    #     output_image[c, mask] = image[c, mask]

    return masked_image


def generate_weak_labels(label, sample_fraction=0.1):
    label_1channel = label
    weak_label = np.zeros_like(label)
    indices = np.argwhere(label > 0)
    sample_size = int(len(indices) * sample_fraction)
    sampled_indices = indices[np.random.choice(len(indices), sample_size, replace=False)]
    for idx in sampled_indices:
        weak_label[idx[0], idx[1]] = label[idx[0], idx[1]]
    return weak_label


class RemoteSensingDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = []
        self.full_label_files = []
        self.point_label_files = []
        self.image_ids = []
        self.image_type = 'RGB'  # IRRG, RGBIR
        
        # Collect all valid image-label pairs
        image_filenames = sorted(glob(image_dir + '/*RGB.tif'))
        label_filenames = sorted(glob(label_dir + '/*label.tif'))
        
        # Create a set of label ids for quick lookup
        label_ids = {filename.split('/')[-1].split('label.tif')[0] for filename in label_filenames}
        
        for image_filename in image_filenames:
            if '4_12' in image_filename or '6_7' in image_filename:
                continue
            image_id = image_filename.split('/')[-1].split(self.image_type)[0]
            if image_id in label_ids:
                image_path = os.path.join(image_dir, image_filename)
                with rasterio.open(image_path) as src:
                    image = src.read()
                    # image = image.permute(0, 2, 1)
                    # image = np.transpose(image, (0,2,1))
                    image_profile = src.profile
                self.image_files.append(image)
                
                label_path = os.path.join(label_dir, f"{image_id}label.tif")
                with rasterio.open(label_path) as label_src:
                    full_label = label_src.read()
                self.full_label_files.append(full_label)
                # print('unique labels', np.unique(full_label), image_id)
                ratio = 1
                point_label_fpath = label_path.replace('label.tif', f'label_point_{ratio}.tif')
                if not os.path.exists(point_label_fpath):
                    print(f'do not exist, will be created')
                    point_label_ = random_zero_out(full_label, keep_ratio=ratio)
                    with rasterio.open(point_label_fpath, 'w', **image_profile) as dst:
                        dst.write(point_label_)
                with rasterio.open(point_label_fpath) as label_src:                    
                    point_label = label_src.read()
                self.point_label_files.append(point_label) 
                              
                self.image_ids.append(image_id)

        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):     
        image_id = self.image_ids[idx]
        image = self.image_files[idx]
        full_label = self.full_label_files[idx]
        point_label = self.point_label_files[idx]
            # label = label.permute(0, 2, 1)
            # label = np.transpose(label, (0,2,1))

        # # 加载 DSM 图像
        # with rasterio.open('path/to/dsm_image.tif') as src:
        #     dsm_image = src.read(1)  # DSM 是单通道图像
        #     dsm_profile = src.profile

            
        # image = Image.open(image_path).convert("RGB")
        # label = Image.open(label_path).convert("L")
        

        img_min, img_max = np.min(image), np.max(image)
        normalized_image = (image - img_min) / (img_max - img_min)  # normalize to [0, 1]

        normalized_image = torch.tensor(normalized_image.astype('float32'))
        full_label = torch.tensor(full_label.astype('float32'))
        point_label = torch.tensor(point_label.astype('float32'))
        if self.transform:
            # normalized_image, full_label, point_label  = self.transform(normalized_image, full_label, point_label)
            # full_label = self.transform(full_label)
            # point_label = self.transform(point_label)
            seed = torch.randint(0, 2**32, (1,)).item()  # Generate a random seed

            torch.manual_seed(seed)
            normalized_image  = self.transform(normalized_image)
            
            torch.manual_seed(seed)
            full_label = self.transform(full_label)
            
            torch.manual_seed(seed)
            point_label = self.transform(point_label)
            
            
            
        return normalized_image, full_label, point_label, image_id


def random_crop(image, crop_size):
    """Randomly crop the image into crop_size."""
    _, height, width = image.shape
    top = np.random.randint(0, height - crop_size + 1)
    left = np.random.randint(0, width - crop_size + 1)
    cropped_image = image[:, top:top + crop_size, left:left + crop_size]
    return cropped_image

def sliding_window(image, crop_size, stride):
    """Slide a window across the image and yield crops of crop_size."""
    _, height, width = image.shape
    for top in range(0, height - crop_size + 1, stride):
        for left in range(0, width - crop_size + 1, stride):
            yield image[:, top:top + crop_size, left:left + crop_size], top, left

def reconstruct_image(patches, original_size, crop_size, stride):
    """Reconstruct the image from patches."""
    channels, height, width = original_size
    reconstructed_image = np.zeros((channels, height, width), dtype=np.float32)
    count_matrix = np.zeros((height, width), dtype=np.float32)
    
    for patch, top, left in patches:
        reconstructed_image[:, top:top + crop_size, left:left + crop_size] += patch
        count_matrix[top:top + crop_size, left:left + crop_size] += 1

    count_matrix[count_matrix == 0] = 1  # Avoid division by zero
    reconstructed_image /= count_matrix
    return reconstructed_image

def save_fig(image, full_label, weak_label, image_id, epoch=None, pred=None):

    if isinstance(image, torch.Tensor):
        image = image.to('cpu').numpy() 
        full_label = full_label.detach().to('cpu').numpy() 
        weak_label = weak_label.detach().to('cpu').numpy() 
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().to('cpu').numpy() 
            pred = pred * 40 # scale labels
            pred = np.transpose(pred, (1,2,0))  # 将图像的通道维度调整到最后
            nb = 4
        else:
            nb = 3

    image1 = np.transpose(image, (1,2,0))  # 将图像的通道维度调整到最后
    # label = label.squeeze().numpy()  # 去除标签的通道维度
    full_label = np.transpose(full_label, (1,2,0))  # 将图像的通道维度调整到最后
    weak_label = np.transpose(weak_label, (1,2,0))  # 将图像的通道维度调整到最后
    

    plt.figure(figsize=(12, nb*1.5))
    plt.subplot(1, nb, 1)
    plt.imshow(image1, interpolation='none')
    plt.title('Image')
    plt.axis('off')
    
    plt.subplot(1, nb, 2)
    plt.imshow(full_label.astype('int'), interpolation='none')
    plt.title('Full label')
    plt.axis('off')
    
    plt.subplot(1, nb, 3)
    plt.imshow(weak_label.astype('int'), interpolation='none')
    plt.title('Point label')
    plt.axis('off')
    
    if nb==4:
    
        plt.subplot(1, nb, 4)
        plt.imshow(pred.astype('int'), interpolation='none')
        plt.title('Prediction')
        plt.axis('off')
    plt.show()
    plt.savefig(f"/home/jjia/data/remote_sensing/results/epoch_{epoch}_" + f"sample_{image_id}.png")
    plt.close
    print(f"save image at results/epoch_{epoch}_" + f"sample_{image_id}.png")
    return None
    
def preview_images(dataset, num_images=5):
    # 从数据集中随机选择几张图像和标签
    indices = range(num_images)
    
    # 遍历选择的图像和标签，并显示它们
    for i, idx in enumerate(indices):
        image, full_label, weak_label, image_id = dataset[idx]
        save_fig(image, full_label, weak_label, image_id)

    
    
def all_loaders():
    # data_dir = '/home/jjia/data/dataset/pointnet/Potsdam/'
    # image_dir = data_dir + '2_Ortho_RGB'
    # label_dir = data_dir + '5_Labels_all'  # for_participants
    
    train_image_dir = "/home/jjia/data/airs/data/potsdam/train_images"
    train_label_dir = "/home/jjia/data/airs/data/potsdam/train_masks"

    # Define the specific transformations
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, scale=(0.8,1.2)),
        transforms.RandomCrop(2048),
    ])

    # train_transform = JointTransform(train_transform)

    # train_transform = v2.Compose([
    #     # transforms.Pad((6016, 6016)),
    #     # transforms.ToTensor(),  # ToTensor will change the order of the channels
    #     # transforms.Lambda(lambda x: np.float())  # 确保张量是浮点数类型
    #     v2.RandomAffine(degrees=10, scale=(0.8,1.2)),
    #     v2.RandomCrop(1024),
    # ])
    train_dataset = RemoteSensingDataset(image_dir=train_image_dir, label_dir=train_label_dir, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    


    test_image_dir = "/home/jjia/data/airs/data/potsdam/test_images"
    test_label_dir = "/home/jjia/data/airs/data/potsdam/test_masks"
    # test_transform = transforms.Compose([
    #     transforms.CenterCrop(1024),
    # ])
    test_transform = transforms.Compose([
        transforms.CenterCrop(2048),

    ])

    # test_transform = JointTransform(test_transform)

    test_dataset = RemoteSensingDataset(image_dir=test_image_dir, label_dir=test_label_dir, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_dataset, train_dataloader, test_dataset, test_dataloader

def chn1_to_3(single_channel_label):
    # 定义颜色映射 rgb
    colors = torch.tensor([
        [0, 0, 0],        # unknown, black color
        [255, 255, 255],  # Impervious surfaces, white color
        [0, 0, 255],      # Building, blue color
        [255, 255, 0],    # Car, yellow color
        [0, 255, 0],      # Tree, green color
        [0, 255, 255],    # Low vegetation, cyan color
        [255, 0, 0],      # Clutter/background, red color
    ], dtype=torch.uint8)
    
    batch_size, height, width = single_channel_label.shape
    # 创建一个空的张量，用于存储结果
    result = torch.zeros((batch_size, 3, height, width), dtype=torch.uint8)

    for label, color in enumerate(colors):
        # 找到 single_channel_label 中所有等于当前标签的像素
        match = (single_channel_label == label)
        result[:, 0, :, :][match] = color[0]
        result[:, 1, :, :][match] = color[1]
        result[:, 2, :, :][match] = color[2]
    return result


def chn3_to_1(multi_class_mask):
    colors = torch.tensor([
        [0, 0, 0],        # unknown, black color
        [255, 255, 255],  # Impervious surfaces, white color
        [0, 0, 255],      # Building, blue color
        [255, 255, 0],    # Car, yellow color
        [0, 255, 0],      # Tree, green color
        [0, 255, 255],    # Low vegetation, cyan color
        [255, 0, 0],      # Clutter/background, red color
    ], dtype=torch.uint8)
    
    batch_size, chn, height, width = multi_class_mask.shape
    # 创建一个空的张量，用于存储结果
    result = torch.zeros((batch_size, height, width), dtype=torch.long)

    for label, color in enumerate(colors):
        # 找到 mask 中所有与当前颜色匹配的像素
        match = (multi_class_mask == color.view(3, 1, 1)).all(dim=1)
        result[match] = label
    return result

def chn3_to_1_to_3(multi_class_mask):
    colors = torch.tensor([
        [0, 0, 0],        # unknown, black color
        [255, 255, 255],  # Impervious surfaces, white color
        [0, 0, 255],      # Building, blue color
        [255, 255, 0],    # Car, yellow color
        [0, 255, 0],      # Tree, green color
        [0, 255, 255],    # Low vegetation, cyan color
        [255, 0, 0],      # Clutter/background, red color
    ], dtype=torch.uint8)
    
    batch_size, chn, height, width = multi_class_mask.shape
    # 创建一个空的张量，用于存储结果
    single_channel_label = torch.zeros((batch_size, height, width), dtype=torch.long)
    result = torch.zeros_like(multi_class_mask)

    for label, color in enumerate(colors):
        # 找到 mask 中所有与当前颜色匹配的像素
        match = (multi_class_mask == color.view(3, 1, 1)).all(dim=1)
        single_channel_label[match] = label
                
    

    for label, color in enumerate(colors):
        # 找到 single_channel_label 中所有等于当前标签的像素
        match = (single_channel_label == label)
        result[:, 0, :, :][match] = color[0]
        result[:, 1, :, :][match] = color[1]
        result[:, 2, :, :][match] = color[2]
    return result



if __name__ == "__main__":
    
    device = torch.device("cuda")  # 'cuda'

    train_dataset, train_dataloader, test_dataset, test_dataloader = all_loaders()
    preview_images(train_dataset, num_images=2)

    model = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=7,                      # 6 classes + 1 unknown
    )
    model = model.to(device)

    # preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')
    crop_size = 512
    stride = 512
        
    # setting
    loss_fun = torch.nn.CrossEntropyLoss()  # target value of 0 should be ignored  ignore_index=0
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(50):
        loss_acc = 0
        for idx, (images, full_labels,point_labels, image_id) in enumerate(train_dataloader):
            # if idx > 2:
            #     continue
            images = images.to(device)
            labels = chn3_to_1(point_labels).to(device)


            opt.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = loss_fun(output, labels)
                
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            print(f"loss: {loss: .2f}")
            loss_acc += loss
        loss_ave = loss_acc / len(train_dataloader)
        print(f"train ave loss: {loss_ave: .2f} at epoch {epoch}")

        if epoch % 5 == 0:
            loss_acc = 0
            for images, full_labels,point_labels, image_id in test_dataloader:

                images = images.to(device)
                labels = chn3_to_1(point_labels).to(device)
                
                opt.zero_grad()
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        output = model(images)
                        loss = loss_fun(output, labels)

                print(f"test loss: {loss}")
                loss_acc += loss        
                segmentation_output = torch.argmax(output, dim=1)

                segmentation_chn3 = chn1_to_3(segmentation_output)
                for img, full_, point, img_id, out in zip(images, full_labels, point_labels, image_id, segmentation_chn3):
                    save_fig(img, full_, point, img_id, epoch, out)
                    
                # rec_labels = chn3_to_1_to_3(full_labels)
                # for img, full_, point, img_id, out in zip(images, full_labels, rec_labels, image_id, segmentation_output):
                #     save_fig(img, full_, point, img_id, epoch, out)
                    
            loss_ave = loss_acc / len(test_dataloader)
            print(f"test ave loss: {loss_ave: .2f} at epoch {epoch}")
            
            


