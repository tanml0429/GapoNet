"""
创建混合数据集

metadata: json文件

{
    "version": "1.0",
    "metadata": {
        "name": "polyp_data",
        "description": "Hybrid data for images",
        "labeler": "Menglu Tan",
        "original_datasets": [],
        "original_datasets_dir": "<path to root dir>",
    },
    "data": [
        {
            "id": "0000001",
            "image": "<path to xx>",
            "source": "<源数据集>",
        },
        {
            "id": "0000002",
            "image": "<path to xx>",
            "source": "<源数据集>",
        },
        ...
    ]
}

"""


import os
import json
from pathlib import Path
here = Path(__file__).parent


class HybridData:
    def __init__(self, original_datasets_dir: str):
        self.file_path = f"{here}/hybrid_data2.json"
        self.data = self.init_data(original_datasets_dir)
        self.save_json()
        print(f"Hybrid data initialized: {self.data}")

    @property
    def metadata(self):
        return self.data["metadata"]
    
    @property
    def entries(self):
        return self.data["entries"]

    def init_data(self, original_datasets_dir: str):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                data = json.load(f)
            return data
        else:
            data = dict()
            data["version"] = "1.0"
            data["metadata"] = dict()
            data["metadata"]["name"] = "polyp_data"
            data["metadata"]["description"] = "Hybrid data for images"
            data["metadata"]["labeler"] = "Menglu Tan"
            data["metadata"]["original_datasets"] = []
            data["metadata"]["original_datasets_dir"] = original_datasets_dir
            data["entries"] = []
            return data
            
    def save_json(self):
        with open(self.file_path, "w") as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    def load_data(self):
        dataset_dir = "polyp/data/data"

    def get_img_files(self, path):
        files = sorted([f'{path}/{x}' for x in os.listdir(path) if Path(x).suffix in ['.jpg', '.png', '.jpeg']])
        # files = sorted([f'{path}/{x}' for x in os.listdir(path)])
        return files

    def fill_entries(self):
        original_datasets_dir = self.metadata["original_datasets_dir"]
        dataset_names = sorted([x for x in os.listdir(original_datasets_dir) if x != '.DS_Store'])
        
        for i, dataset_name in enumerate(dataset_names):
            ori_dataset_names = [x["name"] for x in self.metadata["original_datasets"]]
            if dataset_name in ori_dataset_names:
                continue
            train_dir = f"{original_datasets_dir}/{dataset_name}/images/train"  
            train_img_files = self.get_img_files(train_dir)
            test_dir = f"{original_datasets_dir}/{dataset_name}/images/test"
            test_img_files = self.get_img_files(test_dir)
            img_files = train_img_files + test_img_files
            img_files = sorted(img_files)
            print(len(img_files))

            for j, img_file in enumerate(img_files):
                entry = dict()
                entry["id"] = f"{len(self.entries) + 1:0>8}"
                entry["image"] = str(Path(img_file).relative_to(original_datasets_dir))
                entry["source"] = dataset_name
                self.entries.append(entry)
                print(f"\rEntry {j + 1} filled: {entry}", end="")
            ori_data_set_info = dict()
            ori_data_set_info["name"] = dataset_name
            ori_data_set_info["num_images"] = len(img_files)
            self.metadata["original_datasets"].append(ori_data_set_info)

            print()
            self.save_json()
            pass
        pass

if __name__ == "__main__":
    original_datasets_dir = "/data/tml/mixed_polyp_v5_format"
    data = HybridData(original_datasets_dir)
    data.fill_entries()
    pass