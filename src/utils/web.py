
import requests
from tqdm import tqdm


def download_file(url, target):
    # download the archive
    response = requests.get(url, stream=True)

    # determine total size of download
    total_size = int(response.headers.get("Content-Length", 0))

    with open(target, "wb") as f:
        it = response.iter_content(chunk_size=1024 * 1024)
        for chunk in tqdm(it, desc="Downloading", total=total_size // (1024 * 1024), unit="MiB"):
            if chunk:
                f.write(chunk)

    return target
