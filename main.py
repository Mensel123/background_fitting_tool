import asyncio

import pandas as pd
import pyodide
import js
from io import BytesIO


async def get_file(e):
    files = e.target.files.to_py()
    for file in files:
        array_buf = await file.arrayBuffer()  # Get arrayBuffer from file
        file_bytes = array_buf.to_bytes()  # convert to raw bytes array
        file_content = BytesIO(file_bytes)  # Wrap in Python BytesIO file-like object

        df = pd.read_csv(file_content)

        js.document.getElementById("content").innerHTML = df.columns


def main():
    get_file_proxy = pyodide.ffi.create_proxy(get_file)
    js.document.getElementById("fileinput").addEventListener("change", get_file_proxy)


main()