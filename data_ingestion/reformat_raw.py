
import yaml
import pandas as pd
from config import Config
from pathlib import Path
import hashlib

## TODO rather than writing to temp dir better to write directly to s3
temp_dir = "./tmp"
path = f"{temp_dir}/brand_catalogue.csv"
Path(path).parent.mkdir(parents=True, exist_ok=True)
CONFIG = Config()

def save_brand_catalogue(data_metadata):
    '''
    Save brand with id
    
    NOTE: Id is currently a monotonically increasing value, 
    but might alter or enhance approach depending on potential inclusion of different datasets in the future
    '''
    brand_names = data_metadata["names"]
    brand_df = pd.DataFrame({"id": [i for i in range(data_metadata["nc"])], "name": brand_names})
    brand_df.to_csv(f"{temp_dir}/brand_catalogue.csv", index=False) ## TODO: make this path work with Modal volume

def get_classes_in_img(parent_dir, img_name):
    '''
    Finds and formats class_ids from a label file. 
    Raises an exception if the matching label file is not found in expected location
    '''
    label_path = f"{parent_dir}/labels/{str(img_name)}.txt"
    try:
        with open(label_path, 'r') as file:
            classes = list(set(int(line.strip().split(" ")[0]) for line in file))
            classes.sort()
            return classes
    except FileNotFoundError:
        raise Exception(f"File {label_path} not found")
    except PermissionError:
        raise Exception(f"Permission denied to read {label_path}")
    except Exception as e:
        raise Exception(f"Error reading file: {e}")

def save_image_and_label_catalogue(data_metadata, parent_dir):
    '''
    Input: parsed yaml file from dataset that gives paths to the dataset splits
    
    Take all data from existing dataset splits (train, val, test) and save each image/label pair into the image catalogue
    An exception is thrown and catalogue doesn't save if a matching label cannot be found for an image. 
    (The same exception is not thrown if there doesn't exist a matching image to a given label file.)
    
    Assumptions: 
    (1) Image and label pairs are uniquely identifiable by their file name (the image id is a hash of this name)
    ex: 
    0_jpeg.rf.7f31cb6315d6f40d5245c8358e01f620.jpg
    0_jpeg.rf.7f31cb6315d6f40d5245c8358e01f620.txt
    (2) Provided labels are already in Yolo format
    
    NOTE in the future might want to add dataset versioning and naming to config or as a commandline input
    
    Image catalogue: csv with columns
    - id (str): hash of image_path stem (aka without the .jpg or .png ending, irregardless of parent directory)
    - s3_img_path (str): image location in s3
    - s3_label_path (str): matching YOLO formatted label location in s3
    - class_ids (array of ints): a deduplicated and ordered list of class ids present in the image
    - split (str): one of ["train", "val", "test"]
    '''

    potential_splits = ["train", "val", "test"]
    catalogue_dict = {
        "id": [], "s3_img_path":[], "s3_label_path":[], "class_ids":[], "split":[]
    }
    for split in potential_splits: 
        if split in data_metadata:
            # process entire directory name
            # TODO construct path based on Model volume location
            split_dir = data_metadata[split].replace("/images", "").replace("../", "./")
            full_split_dir = f"{parent_dir}/{split_dir}"
            source_path = Path(f"{full_split_dir}/images")
            
            # process data (for each image, confirm that a matching label exists)
            for src_file in source_path.rglob('*'):
                rel_path = src_file.relative_to(source_path)
                rel_path_stem = rel_path.stem
                hashed_id = hashlib.sha256(str(rel_path_stem).encode()).hexdigest()
                classes = get_classes_in_img(full_split_dir, rel_path_stem)
                
                catalogue_dict["id"].append(hashed_id)
                catalogue_dict["s3_img_path"].append(f"{CONFIG.image_dump_path}/{rel_path}")
                catalogue_dict["s3_label_path"].append(f"{CONFIG.label_dump_path}/{rel_path}")
                catalogue_dict["class_ids"].append(classes)
                catalogue_dict["split"].append(split)
                

    pd.DataFrame(catalogue_dict).to_csv(f"{temp_dir}/image_catalogue.csv", index=False) ## TODO: save this to s3


if __name__ == "__main__": 
    # TODO modal volume location for raw data to be passed in as argument
    modal_volume_location = "./F1 Logos.v8i.yolov11_sample"
    data_yaml_path = f"{modal_volume_location}/data.yaml"
    with open(data_yaml_path) as stream:
        try:
            data_metadata = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise Exception(exc)
    
    # TODO: edit local dataset paths (include modal volume location)
    print("Saving brand catalogue..")
    save_brand_catalogue(data_metadata)
    print("Saving image and label catalogue..")
    save_image_and_label_catalogue(data_metadata, parent_dir=modal_volume_location)
    print("Migrating image and label raws to s3..") # TODO

