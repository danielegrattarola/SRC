import os
import os.path as osp
import shutil
import zipfile
from glob import glob

import requests
from joblib import Parallel, delayed
from spektral.data import Dataset
from spektral.utils import one_hot
from tqdm import tqdm

from src.modules.off import read_off


class ModelNet(Dataset):
    url = {
        "10": "http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
        "40": "http://modelnet.cs.princeton.edu/ModelNet40.zip",
    }

    def __init__(self, name, test=False, **kwargs):
        if name not in self.available_datasets:
            raise ValueError(
                "Unknown dataset {}. Possible: {}".format(name, self.available_datasets)
            )
        self.name = name
        self.test = test
        self.n_jobs = kwargs.pop("n_jobs", -1)
        self.true_path = osp.join(self.path, "ModelNet" + self.name)
        super().__init__(**kwargs)

    def read(self):
        folders = glob(osp.join(self.true_path, "*", ""))
        dataset = "test" if self.test else "train"
        classes = [f.split("/")[-2] for f in folders]
        n_out = len(classes)

        print("Loading data")

        def load(fname):
            graph = read_off(fname)
            graph.y = one_hot(i, n_out)
            return graph

        output = []
        for i, c in enumerate(tqdm(classes)):
            fnames = osp.join(self.true_path, c, dataset, "{}_*.off".format(c))
            fnames = glob(fnames)
            output_partial = Parallel(n_jobs=self.n_jobs)(
                delayed(load)(fname) for fname in fnames
            )
            output.extend(output_partial)

        return output

    def download(self):
        print("Downloading ModelNet{} dataset.".format(self.name))
        url = self.url[self.name]
        req = requests.get(url)
        if req.status_code == 404:
            raise ValueError(
                "Cannot download dataset ({} returned 404).".format(self.url)
            )
        os.makedirs(self.path, exist_ok=True)

        fname = osp.join(self.path, "ModelNet" + self.name + ".zip")
        with open(fname, "wb") as of:
            of.write(req.content)
        with zipfile.ZipFile(fname, "r") as of:
            of.extractall(self.path)

        shutil.rmtree(osp.join(self.path, "__MACOSX"), ignore_errors=True)

    @property
    def available_datasets(self):
        return ["10", "40"]
