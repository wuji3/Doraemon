import datasets                                                                                                                                                                                                                                
import os
from PIL import Image
from typing import Dict, List
from datasets import Dataset, Features, Image as ImageFeature, Value
         
class FaceDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
         
    def generate_examples(self):
        examples = []
        identity_dirs = sorted([d for d in os.listdir(self.data_dir)
                              if os.path.isdir(os.path.join(self.data_dir, d))])
        id_to_label = {id_name: idx for idx, id_name in enumerate(identity_dirs)}
         
        for identity in identity_dirs:
            identity_dir = os.path.join(self.data_dir, identity)
            label = id_to_label[identity]
         
            for file_name in os.listdir(identity_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(identity_dir, file_name)
                    try:
                        examples.append({
                            "image": image_path,
                            "label": label,
                            "class_name": identity,
                            "file_name": file_name
                        })
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
                        continue
        return examples
         
def create_and_upload_dataset(data_dir):
    # Create dataset instance
    face_dataset = FaceDataset(data_dir)
         
    # Generate examples
    examples = face_dataset.generate_examples()
         
    # Create Dataset object
    dataset = Dataset.from_list(
        examples,
        features=Features({
            "image": ImageFeature(),
            "label": Value("int64"),
            "class_name": Value("string"),
            "file_name": Value("string")
        })
    )    
         
    # Upload to Hub
    dataset.push_to_hub(
        "User/DatasetName",
        private=False,
        max_shard_size="500MB"
    )    
         
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the face data directory")
    args = parser.parse_args()

    create_and_upload_dataset(args.data_dir)