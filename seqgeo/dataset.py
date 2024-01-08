import glob
import random
import os
import json
import math
import time
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

def LatLngToPixel(lat, lng, centerLat, centerLng, zoom):
    x, y = LatLngToGlobalPixel(lat, lng, zoom)
    cx, cy = LatLngToGlobalPixel(centerLat, centerLng, zoom)
    return x - cx, y - cy

def LatLngToGlobalPixel(lat, lng, zoom):
    siny = math.sin(lat * math.pi / 180.0)
    siny = min(max(siny, -0.9999), 0.9999)

    return [(256 * (0.5 + lng / 360.0)) * (2 ** zoom), (256 * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)))*(2 ** zoom)]

class ImageDataset(Dataset):
    def __init__(self, root="dataset/json", transforms_street=[transforms.ToTensor(),],transforms_sat=[transforms.ToTensor(),], sequence_size = 7, mode='train', zoom=20):
        self.zoom = zoom
        self.transforms_street = transforms.Compose(transforms_street)
        self.transforms_sat = transforms.Compose(transforms_sat)
        self.seqence_size = sequence_size
        self.mode = mode

        if mode == "train" or mode == "val" or "dev":
            self.year = "2019"
        else:
            raise RuntimeError("no such mode")
        
        self.json_files = sorted(glob.glob(os.path.join(root, self.year+"_JSON") + '/*.json'), key=lambda x:int(x.split("/")[-1].split(".json")[0]))
        if self.year == "2019":
            # train인 경우에 self.json_files[:int(len(self.json_files)*0.8)]이 뭐를 의미하는지 확인해보기
            if mode == "train":
                self.json_files = self.json_files[:int(len(self.json_files)*0.8)]
            elif mode == "val":
                self.json_files = self.json_files[int(len(self.json_files)*0.8+1):]
            elif mode == "dev1":
                self.json_files = self.json_files[:int(len(self.json_files)*0.05)]
            elif mode == "dev2":
                self.json_files = self.json_files[int(len(self.json_files)*0.99):]

        self.val_center = []
        if mode == "val" or mode == "dev1" or mode == "dev2":
            for i in self.json_files:
                f = open(i, 'r')
                meta_data = json.load(f)#load json
                center_lat, center_lon = meta_data["center"]
                self.val_center.append([center_lat, center_lon])
                f.close()

    # 사용 x
    def get_sat_center(self, idx):
        if len(self.val_center) > 0:
            return self.val_center[idx]

    def __getitem__(self, index):
        f = open(self.json_files[index])#open json
        meta_data = json.load(f)#load json
        center_lat, center_lon = meta_data["center"]    # json에서 latitude, longtitude 불러와서 저장
        f.close()

        street_images = []
        sate_imgs = []

        dir_sate_img = meta_data["satellite_views"][str(self.zoom)]
        dir_sate_img = dir_sate_img.split("\\")[1:]
        dir_sate_img = "/".join(dir_sate_img)
        sate_img = self.transforms_sat(Image.open(os.path.join("dataset/satellite", dir_sate_img)))
        sate_imgs.append(sate_img)
        # 여기까지 진행하면, sate_imgs에 dataset/satellite/2019_satellite/{zoom}/{idx}.png 파일들이 담겨 있음

        sate_imgs = torch.stack(tuple(sate_imgs), 0)
        # torch.stack(..., dim=0) 하면, (dims, w ,h) 형식으로 저장되는 것 같음. dims는 이미지 수?

        all_street_views = meta_data["street_views"]
        # if one sequence >7 random drop some
        # 여기는 아마 갖고 있는 street_views가 설정한 seq_size보다 크면 랜덤으로 dropout 한다는 것 같음
        if len(all_street_views.keys()) > self.seqence_size: 
            if self.mode == "train":
                for d in range(len(all_street_views.keys()) - self.seqence_size):
                    all_street_views.pop(random.choice(list(all_street_views.keys())))
            else:
                for d in range(len(all_street_views.keys()) - self.seqence_size):
                    all_street_views.pop(list(all_street_views.keys())[-1])

        if len(all_street_views.keys()) < 7:
            print(self.json_files[index])

        # street_views에서 하나씩
        for k in sorted(all_street_views.keys()):
            v = all_street_views[k]
            px, py = LatLngToPixel(v["lat"],v["lon"],center_lat, center_lon,20)
            dir_img = v["name"]
            dir_img = dir_img.split("\\")[1:]
            dir_img = "/".join(dir_img)
            dir_img = os.path.join("dataset/street", os.path.join(str(self.year)+"_street", dir_img))
            img = self.transforms_street(Image.open(dir_img))

            street_images.append(img)
            # 마찬가지로 여기도 지상 이미지들 저장해놓음

        #stack to torch tensors on dim=0
        street_images = torch.stack(tuple(street_images), 0)
        return {"street":street_images, "satellite":sate_imgs}


    def __len__(self):
        return len(self.json_files)

if __name__ == "__main__":
    # Configure data loader
    # transforms_ = [ transforms.RandomHorizontalFlip(),
    #     transforms.Lambda(lambda img: img.crop((0, int(img.size[1] * 0.45), img.size[0], img.size[1]))),
    #     transforms.Resize((200 + 15, 640 + 25)),
    #     transforms.RandomCrop((200, 640)),
    #     transforms.RandomAffine(degrees=8, translate=(0.1, 0.1), scale=(0.9, 1), shear=(6, 6)),
    #     transforms.ColorJitter(0.2, 0.2, 0.2),
    #     transforms.RandomGrayscale(),
    #     transforms.ToTensor(),
    #     transforms.RandomVerticalFlip(p=0.2),
    #     transforms.RandomErasing(p=0.2,scale=(0.02, 0.1), ratio=(0.6, 1.5)),
    #     transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]) 
    #     ]

    STREET_IMG_WIDTH = 256
    STREET_IMG_HEIGHT = 144
    SATELLITE_IMG_WIDTH = 224
    SATELLITE_IMG_HEIGHT = 224

    transforms_sate = [transforms.Resize((SATELLITE_IMG_WIDTH, SATELLITE_IMG_HEIGHT)),
                    transforms.ColorJitter(0.2, 0.2, 0.2),
                    transforms.RandomGrayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]) 
                    ]
    transforms_street = [transforms.Resize((STREET_IMG_HEIGHT, STREET_IMG_WIDTH)),
                    transforms.ColorJitter(0.2, 0.2, 0.2),
                    transforms.RandomGrayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]) 
                    ]

    dataloader = DataLoader(ImageDataset(transforms_street=transforms_street,transforms_sat=transforms_sate,mode="train"),\
         batch_size=4, shuffle=True, num_workers=8)
    
    print(len(dataloader))
    total_time = 0
    start = time.time()
    for i,b in enumerate(dataloader):
        end = time.time()
        elapse = end - start
        total_time += elapse
        print(elapse)
        start = end
        print("===========================")
        print(b["street"].shape)
        print(b["headings"].shape)
        print(b["locations"].shape)
        print(b["satellite"].shape)
        print("===========================")
        time.sleep(2)

    print(total_time / i)
