
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import extract_archive

import numpy as np

import pandas as pd

from tqdm import tqdm

from typing import Sequence

import requests
import shutil
import multiprocessing


import src.utils as U


class LidarCSDataset(Dataset):
    SENSORS = {
        "Livox",
        "ONCE-40",
        "VLD-16",
        "VLD-32",
        "VLD-128",
        "VLD-128",
        "VLD-64-1.0m",
        "VLD-64-1.5m",
        "VLD-64-2.5m",
        "VLD-64-3.0m",
    }

    BASE_URL = "https://ad-apolloscape.cdn.bcebos.com/LiDAR_CS/"

    def __init__(self, root: str | Path, sensors: Sequence[str], samples: int = 2 ** 11, download: bool = True):
        super().__init__()
        self.root = Path(root)
        self.samples = samples

        match sensors:
            case "all":
                self.sensors = self.SENSORS
            case list() | set() | tuple() as multiple_sensors:
                self.sensors = set(multiple_sensors)
            case str() as single_sensor:
                self.sensors = {single_sensor}
            case _:
                raise TypeError(f"Cannot handle shapes of type {type(sensors)}.")

        unknown_sensors = self.sensors - self.SENSORS
        if unknown_sensors:
            raise ValueError(f"Unknown sensors: {unknown_sensors}")

        if download:
            self.download()

        self.files = pd.DataFrame(columns=["sensor", "path", "mean", "std"])

        self._preload_files()

    def __getitem__(self, item):
        row = self.files.iloc[item]
        path = row["path"]

        samples = np.fromfile(path, dtype=np.float32).reshape(-1, 4)

        points = torch.from_numpy(samples[:, :3])
        subset = torch.randperm(len(points))[:self.samples]
        points = points[subset]
        points = (points - row["mean"]) / row["std"]

        return points

    def __len__(self):
        return len(self.files)

    def download(self):
        need_sensors = self.sensors
        have_sensors = set()
        if self.root.is_dir():
            for sensor in self.root.iterdir():
                if sensor.is_dir():
                    have_sensors.add(sensor.name)
        else:
            self.root.mkdir(exist_ok=True, parents=True)

        if need_sensors.issubset(have_sensors):
            print(f"Found all requested sensors in {self.root}, skipping download...")
            return
        elif have_sensors:
            print(f"Found {have_sensors} in {self.root}, downloading missing sensors...")
            need_sensors -= have_sensors
        else:
            print(f"Found empty {self.root}, downloading all requested sensors...")

        pbar = tqdm(self.sensors, desc="Downloading Sensor Data")
        for sensor in pbar:
            pbar.set_postfix_str(f"Sensor: {sensor}")

            self._download_sensor(sensor, extract=True)

    def _download_sensor(self, sensor, extract=True):
        """ Download archives for a single sensor. Checks automatically if the archive is split or not. """
        url = self.BASE_URL + sensor + ".tar.gz"

        # check if the base file exists
        response = requests.head(url)
        if response.status_code == 200:
            # the non-split file exists, just download it
            print(f"Archive for sensor {sensor} is not split, downloading single file...")
            self._download_single(sensor, extract=extract)
            return

        if response.status_code == 404:
            # the file does not exist, it is probably split
            # try downloading all parts
            print(f"Archive for sensor {sensor} is split, downloading all parts...")
            self._download_splits(sensor, extract=extract)
            return

        # something went wrong
        response.raise_for_status()
        raise RuntimeError(f"Unexpected status code {response.status_code} for {url}.")

    def _download_single(self, sensor, suffix=".tar.gz", extract=False):
        """ Download a non-split archive or a single split from a split archive """
        url = self.BASE_URL + sensor + suffix

        archive = self.root / (sensor + suffix)
        if archive.is_file():
            return
        print(f"Downloading {url} to {archive}...")
        U.download_file(url, archive)

        if extract:
            target = self.root
            print(f"Extracting {archive} to {target}...")
            extract_archive(str(archive), str(target), remove_finished=True)

    def _download_splits(self, sensor, extract=False):
        """ Download all splits from a split archive """
        archive = self.root / (sensor + ".tar.gz")

        with archive.open("wb") as f:
            i = 0
            while True:
                split_url = self.BASE_URL + sensor + f".tar.gz{i:03d}"
                response = requests.head(split_url)

                if response.status_code == 404:
                    # the file does not exist, we are done
                    break
                elif response.status_code != 200:
                    # something went wrong
                    response.raise_for_status()
                    raise RuntimeError(f"Unexpected status code {response.status_code} for {split_url}. "
                                       f"Note: this may leave a partial archive in {archive}.")

                # the split exists, download it
                suffix = f".tar.gz{i:03d}"
                self._download_single(sensor, suffix=suffix, extract=False)
                split = self.root / (sensor + suffix)

                # concatenate the split to the archive
                with split.open("rb") as g:
                    shutil.copyfileobj(g, f)

                # remove the split
                split.unlink()

                i += 1

        if extract:
            target = self.root
            print(f"Extracting {archive} to {target}...")
            extract_archive(str(archive), str(target), remove_finished=True)

    def _preload_files(self):
        files = []
        for sensor in self.sensors:
            sensor_dir = self.root / sensor / "bin"
            files.extend(list(sensor_dir.glob("*.bin")))

        rows = []

        with multiprocessing.Pool() as pool:
            processes = pool.imap_unordered(self._load_file, files, chunksize=8)
            for mesh in tqdm(processes, desc="Pre-Loading Files", total=len(files)):
                rows.append(mesh)

        self.files = pd.concat(rows, ignore_index=True)

        self.files = self.files.sort_values(by=["sensor"]).reset_index()

    def _load_file(self, path):
        sensor = path.parent.parent.name
        path = str(path.resolve())
        samples = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        samples = samples[:, :3]
        mean = np.mean(samples, axis=0, keepdims=True)
        std = np.std(samples, axis=0, keepdims=True)

        row = {
            "sensor": sensor,
            "path": path,
            "mean": mean,
            "std": std,
        }
        row = {k: [v] for k, v in row.items()}
        row = pd.DataFrame(row, index=[0])

        return row
