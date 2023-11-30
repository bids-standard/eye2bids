"""Download test data from OSF, unzip and install in the correct location.

If rerun, all previous data is removed and redownloaded.
"""

import shutil
import zipfile
from pathlib import Path

import requests

download_url = "https://files.de-1.osf.io/v1/resources/jdv7n/providers/osfstorage/?zip="

root_dir = Path(__file__).parent.parent

output_dir = root_dir / "tests" / "data" / "osf"

if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True)

output_file = output_dir / "test_data.zip"

# Download the zip file
r = requests.get(download_url, stream=True)
with open(output_file, "wb") as f:
    for chunk in r.iter_content(chunk_size=1024):
        if chunk:
            f.write(chunk)

# unzip the file
with zipfile.ZipFile(output_file, "r") as zip_ref:
    zip_ref.extractall(output_dir / ".")
    output_file.unlink()
