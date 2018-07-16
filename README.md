Image Captioning
===

## Step 1. Download COCO dataset
Install gsutil
```sh
curl https://sdk.cloud.google.com | bash
```
After restart the terminal, download images and save them to 'coco/images'
```sh
gsutil -m rsync gs://images.cocodataset.org/[dataset_name] [save_to_path]
```
Download annotations and save them to 'coco/annotations'
```sh
gsutil -m rsync gs://images.cocodataset.org/annotations [save_to_path]
```
Datasets can also be downloaded directly from http://cocodataset.org/#download

## Step 2. Download and install COCO API
After cd into the directory you want to save the installation files
```sh
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI/
python setup.py build_ext install
```
COCO API demo: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb