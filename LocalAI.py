import os
import shutil
from io import BytesIO
from typing import Dict

import pandas as pd

# Import the EvaDB package
import evadb

# Connect to EvaDB and get a database cursor for running queries
import requests
from PIL import Image
from io import BytesIO

cursor = evadb.connect().cursor()

print(cursor.query("SHOW FUNCTIONS;").df())


def display_image_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:  # HTTP状态码200表示请求成功
        image_data = BytesIO(response.content)
        image = Image.open(image_data)
        image.show()
    else:
        print(f"Failed to retrieve the image. HTTP Status Code: {response.status_code}")


query = cursor.query("""
    CREATE FUNCTION
    IF NOT EXISTS GenerateImage
    IMPL 'test.py';
""")
# cursor.execute(query)
response = query.df()
input = "okok"

cursor.query("DROP TABLE IF EXISTS History").df()

cursor.query("""
    CREATE TABLE History
    (id INTEGER,
    command TEXT(30),
    data TEXT(30));
""").df()

cursor.query(f"""
    INSERT INTO History (id, command, data ) VALUES
    (1,
    '{input}',
    "null");
""").df()

query = cursor.query("""
    SELECT * FROM History;
""").df()

# print(query)

query = cursor.query("""
    SELECT GenerateImage(command).result FROM History;
""").df()
print(query)

# List all the built-in functions in EvaDB
print(cursor.query("SHOW FUNCTIONS;").df())

url_image = query['generateimage.result'].iloc[0]
display_image_from_url(url_image)

cursor.query("DROP FUNCTION GenerateImage").df()
