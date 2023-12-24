import multiprocessing
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import trimesh
from torch.utils.data import Dataset
from torchvision.datasets.utils import extract_archive
from tqdm import tqdm
from typing import Sequence


class ModelNet10Dataset(Dataset):

    SHAPES = {
        "bathtub",
        "bed",
        "chair",
        "desk",
        "dresser",
        "monitor",
        "night_stand",
        "sofa",
        "table",
        "toilet",
    }

    def __init__(self, root: str | Path, shapes: str | Sequence[str] = "all", samples: int = 2 ** 11, download: bool = True):
        super().__init__()

        self.root = Path(root)
        self.samples = samples

        if download:
            self.download()

        self.files = pd.DataFrame(columns=["shape", "split", "index", "path", "mesh", "mean", "std"])

        match shapes:
            case "all":
                self.shapes = self.SHAPES
            case list() | set() | tuple() as multiple_shapes:
                self.shapes = set(multiple_shapes)
            case str() as single_shape:
                self.shapes = {single_shape}
            case _:
                raise TypeError(f"Cannot handle shapes of type {type(shapes)}.")

        unknown_shapes = self.shapes - self.SHAPES
        if unknown_shapes:
            raise ValueError(f"Unknown shapes: {unknown_shapes}")

        self._preload_meshes()

    def __getitem__(self, item):
        row = self.files.iloc[item]
        mesh = row["mesh"]

        points = mesh.sample(self.samples).astype(np.float32)
        points = (points - row["mean"]) / row["std"]

        return torch.from_numpy(points)

    def __len__(self):
        return len(self.files)

    def download(self):
        if self.root.is_dir() and any(self.root.iterdir()):
            print(f"Found non-empty {self.__class__.__name__} in {self.root}, skipping download...")
            return

        url = "https://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"

        # download the archive
        response = requests.get(url, stream=True)

        # determine total size of download
        total_size = int(response.headers.get("Content-Length", 0))

        self.root.mkdir(exist_ok=True, parents=True)

        archive = self.root / "ModelNet10.zip"
        target = self.root

        with open(archive, "wb") as f:
            it = response.iter_content(chunk_size=1024 * 1024)
            for chunk in tqdm(it, desc="Downloading", total=total_size // (1024 * 1024), unit="MiB"):
                if chunk:
                    f.write(chunk)

        print(f"Extracting {archive} to {target}...")
        extract_archive(str(archive), str(target), remove_finished=True)

    def _preload_meshes(self):
        mesh_paths = []
        for shape in self.shapes:
            path = self.root / "ModelNet10" / shape
            shape_meshes = list(path.glob("**/*.off"))
            mesh_paths.extend(shape_meshes)

        rows = []

        with multiprocessing.Pool() as pool:
            # this is hyper-optimized, do not touch
            processes = pool.imap_unordered(self._load_mesh, mesh_paths, chunksize=8)
            for mesh in tqdm(processes, desc="preloading meshes", total=len(mesh_paths)):
                rows.append(mesh)

        self.files = pd.concat(rows, ignore_index=True)

        self.files = self.files.sort_values(by=["shape"]).reset_index()

    def _load_mesh(self, mesh_path):
        row = {
            "shape": mesh_path.parent.parent.name,
            "split": mesh_path.parent.name,
            "index": int(re.match(r".*_(\d+)\.off", mesh_path.name).group(1)),
            "path": str(mesh_path),
            "mesh": trimesh.load(mesh_path),
        }

        samples = row["mesh"].sample(self.samples).astype(np.float32)
        row["mean"] = samples.mean(axis=0, keepdims=True)
        row["std"] = samples.std(axis=0, keepdims=True)

        row = {k: [v] for k, v in row.items()}
        row = pd.DataFrame(row, index=[0])

        return row

