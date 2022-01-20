# %%
# import os
import requests
import urllib.request
import random
import numpy as np
from PIL import Image

# %%
# Generate array of random ids
ran_list = [random.randint(1, 10000) for i in np.arange(12)]

# create list of object URLs
for i in ran_list:
    url_list = []
    url = f'https://collectionapi.metmuseum.org/public/collection/v1/objects/{i}'
    response = requests.get(url, stream=True)
    json_response = response.json()
    # fetch image
    try:
        img_url = json_response['primaryImageSmall']
        print(img_url)
    except:
        continue
    url_list.append(img_url)

url_list


# %%
response = requests.get(url, stream=True)
json_response = response.json()
# fetch image
img_url = json_response['primaryImageSmall']

# download image
urllib.request.urlretrieve(
  img_url, "image_1.jpg")

# save and load image
img = Image.open("image_1.jpg")
img.show()
# %%


# %%
