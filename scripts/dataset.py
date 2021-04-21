import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import os
import ast
import glob
import cv2 



#  dowload and extract
def download_TUsimple_dataset(version, url="https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/{version}_set.zip"):
    url = url.format(version=version)
    ziped_file_name = version+"_set.zip"
    file_name = version+"_set"

    # download dataset
    if not os.path.exists(ziped_file_name):
        print("Downloading: "+url)
        import urllib.request
        urllib.request.urlretrieve(url, ziped_file_name)
    else:
        print("Dataset already downloaded")

    # extract dataset
    if not os.path.exists(file_name):
        print("Unzipping dataset file")
        import zipfile
        with zipfile.ZipFile(ziped_file_name, 'r') as zip:
            zip.extractall(file_name)
    else:
        print("Dataset already unziped")

# create an api for the dataset from the stored files in the hardrive
class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, size=None, limit=None):

        self.size = size
        self.limit = limit
        self.dataset_path = dataset_path 
        # get the iamge paths
        self.images_paths = glob.glob(dataset_path + '/clips/*/*/*.jpg', recursive=False)
        # sort the paths for better video creation
        self.images_paths.sort(key=lambda x: (x.split("/")[-3], int(x.split("/")[-2]), int(x.split("/")[-1].split(".")[0])) )

        # read the labels from jsons
        labels_byte_dicts_list = []
        for f in glob.glob(dataset_path + "/*.json"):
            with open(f, "rb") as infile:
                labels_byte_dicts = infile.read().splitlines()
                labels_byte_dicts_list.extend(labels_byte_dicts)
        print(len(labels_byte_dicts_list))

        # create a dict for labels with key the image unique path
        self.labels_dict = {}
        for byte_dict in labels_byte_dicts_list:
            label_dict = ast.literal_eval(byte_dict.decode("UTF-8"))
            self.labels_dict[dataset_path + "/" + label_dict["raw_file"]] = label_dict
        print(len(self.labels_dict))


    def __len__(self):
        if (self.limit is not None) and (self.limit<=len(self.images_paths)) :
            return self.limit
        return len(self.images_paths)

    def __getitem__(self, index):

        # get the image
        image_path = self.images_paths[index]
        image = cv2.imread(image_path)
        image_original_shape = image.shape
        # resize if shape is not none
        if self.size is not None:
            image = cv2.resize(image, self.size)

        # create the image label
        label_dict = self.labels_dict["/".join(self.images_paths[index].split('/')[:-1])+"/20.jpg"]


        label_lanes = label_dict['lanes']
        y_samples = label_dict['h_samples']
        # raw_file = label_dict['raw_file']

        # gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in label_lanes]


        gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in label_lanes]
        # pred_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in pred_lanes]
        # img_vis = cv.imread(train_images_paths[1000])
        label = np.zeros(image_original_shape, dtype="uint8")

        for lane in gt_lanes_vis:
            # cv2.polylines(img_vis, np.int32([lane]), isClosed=False, color=(0,255,0), thickness=5)
            cv2.polylines(label, np.int32([lane]), isClosed=False, color=(255,255,255), thickness=5)

        # resize if shape is not none
        if self.size is not None:
            label = cv2.resize(label, self.size)

        return image, label



# transform dataset to pytorch tensors
class Pytorch_Dataset(torch.utils.data.Dataset):
    def __init__(self, directory, dataset=None, limit=None):
        self.directory = directory
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            if limit is not None:
                self.dataset_size = limit
            else:
                self.dataset_size = int(len(glob.glob("%s/*"%(directory), recursive=False))/2)        
            return None
        self.dataset = dataset
        self.dataset_size = len(self.dataset)
        # self.dataloader = DataLoader(self.dataset, batch_size=self.dataset_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=True, collate_fn=Pytorch_Dataset.cv2_to_pytorch_transformer)
        for item in range(self.dataset_size):
            data, labels = self.dataset[item]
            data = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1)
            labels = torch.tensor(labels, dtype=torch.float32).permute(2, 0, 1)

            # resize lables channel from 3 to 1
            labels = torchvision.transforms.Grayscale(num_output_channels=1)(labels)
            # resize data channel from 3 to 1
            data = torchvision.transforms.Grayscale(num_output_channels=1)(data)


            
            # data_mean   = torch.mean(data, [1,2])
            # labels_mean = torch.mean(labels, [1,2])
            # data_std    = torch.std(data, [1,2])
            # labels_std  = torch.std(labels, [1,2])
            # print(data_mean, data_std)
            data   = torchvision.transforms.Normalize([127.5], [127.5], inplace=False)(data)
            labels = torchvision.transforms.Normalize([127.5], [127.5], inplace=False)(labels)
            # data   = torchvision.transforms.Normalize(data_mean, data_std, inplace=False)(data)
            # labels = torchvision.transforms.Normalize(data_mean, data_std, inplace=False)(labels)
            # save image
            torch.save(data, "%s/data_%s.pt"%(self.directory,str(item)))
            torch.save(labels, "%s/label_%s.pt"%(self.directory,str(item)))

        # data   = [item[0] for item in self.dataset]
        # labels = [item[1] for item in self.dataset]
        # data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2)
        # labels = torch.tensor(labels, dtype=torch.float32).permute(0, 3, 1, 2)


        # convert everything to pytorch tensors
        # start_time = time.time()
        # self.pytorch_dataset = next(iter(self.dataloader))
        # end_time = time.time()


        # self.data   = self.pytorch_dataset[0] 
        # self.labels = self.pytorch_dataset[1] 
        # print(self.data.size())
        # print(self.labels.size())
        # print("Time of conversion {%f} seconds"%(end_time-start_time))
    
    # def get_data(self):
    #     return self.data
    
    # def get_labels(self):
    #     return self.labels
    
    @staticmethod
    def cv2_to_pytorch_transformer(batch):
        # convert cv2 to pytorch tesnors
        data   = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2)
        labels = torch.tensor(labels, dtype=torch.float32).permute(0, 3, 1, 2)

        # resize lables channel from 3 to 1
        labels = torchvision.transforms.Grayscale(num_output_channels=1)(labels)
        # resize data channel from 3 to 1
        data = torchvision.transforms.Grayscale(num_output_channels=1)(data)

        # normalize pixels 0-255 -> 0-1
        # find mean and std
        data_mean   = torch.mean(data, [0,2,3])
        labels_mean = torch.mean(labels, [0,2,3])
        data_std    = torch.std(data, [0,2,3])
        labels_std  = torch.std(labels, [0,2,3])
        
        # print(data_mean, data_std)
        data   = torchvision.transforms.Normalize(data_mean, data_std, inplace=False)(data)
        labels = torchvision.transforms.Normalize(labels_mean, labels_std, inplace=False)(labels)

        return data, labels
        

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        return torch.load("%s/data_%d.pt"%(self.directory,index)), torch.load("%s/label_%d.pt"%(self.directory,index))
