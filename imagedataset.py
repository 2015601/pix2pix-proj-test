from PIL import Image
from torch.utils.data import Dataset
from os.path import join
from os import listdir
from utils import load_img,is_img_file
import torchvision.transforms as transforms

class DataLoad(Dataset):
    def __init__(self,img_dir,flag):
        super(DataLoad, self).__init__()
        self.flag = flag
        self.path_a = join(img_dir,"a")
        self.path_b = join(img_dir,"b")
        self.img_file_list = []

        self.img_file_list = [file for file in listdir(self.path_a) if is_img_file(file)]
        #for file in listdir(self.path_a):
            #if is_img_file(file):
                #self.img_file_list.append(file)

        self.transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                     ])
    def __getitem__(self, item):
        img_a = Image.open(join(self.path_a,self.img_file_list[item])).convert('RGB')
        img_b = Image.open(join(self.path_b,self.img_file_list[item])).convert('RGB')

        img_a = img_a.resize((286, 286), Image.BICUBIC)
        img_b = img_b.resize((286, 286), Image.BICUBIC)

        img_a = self.transform(img_a)
        img_b = self.transform(img_b)

        if self.flag == "a2b":
            return img_a,img_b
        else:
            return img_b,img_a

    def __len__(self):
        return len(self.img_file_list)

def get_trainingset(root_dir,flag):
    train_dir = join(root_dir,"train")
    return DataLoad(train_dir,flag)

def get_testset(root_dir,flag):
    test_dir = join(root_dir,"test")
    return DataLoad(test_dir,flag)









