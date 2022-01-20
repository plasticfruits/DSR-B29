# Import Libs
import requests
import json
import wikipedia


# global vars
countries = ["Vatican City", "Monaco", "Nauru", "Tuvalu", "San Marino", "Liechtenstein", "Maldives", "Malta"]
title = countries[0]
url = f"https://en.wikipedia.org/w/api.php?action=parse&page={title}&format=json"


for i in countries:
    # static local vars
    title = i
    os.mkdir(i) # create directory
    filename = f"{i}/{i}.json" # .json file name
    res = requests.get(url)
    data = res.json()
    dict_url = {'url': url}
    dict_title = {'title': data['parse']['title']}
    
    # get lines
    lines = []
    for i in data['parse']['sections']:
        line = i['line']
        lines.append(line)

    # build dictionary
    dict_lines = {'lines':lines} # convert to dictionary
    dict_images = {'images': data['parse']['images']} # dict with list of images
    dict_combined = {**dict_url, **dict_title, **dict_lines, **dict_images}
    
    # store json in folder
    with open(filename, "w") as f:
        json.dump(dict_combined, f)

# %%
