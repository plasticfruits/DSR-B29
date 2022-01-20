# **Key Python Notes**

R to Pandas: https://pandas.pydata.org/pandas-docs/stable/getting_started/comparison/comparison_with_r.html
R to Pandas 2: https://gist.github.com/conormm/fd8b1980c28dd21cfaf6975c86c74d07
<br>

# **response**
* `response = requests.get(url, stream=True)`
* `json_response = response.json()`
* `img_url = json_response['url']` 

<br>  

## **Images**
* `urllib.request.urlretrieve(img_url, "image_1.jpg")` to downlaod image from URL
* `img = Image.open("image_1.jpg")` to load image
* `img.show()` to display image
* `img = Image.fromarray(inv_img)` save img from array

<br>  

# **Numpy**
* `astype("int")` - convert to type
* `.shape[0] rows` and `.shape[1]` cols
* `arange(start, stop, step)` for creating arrays
* `np.dot(A, B)` for matrix multiplication
* `linspace(start, end, step)` evenly spaced points
* `np.random.rand(2,4)` sampling random uniform
* `.replace(i, j)` change shape of matrix
    * `-1` used as free dimension: `reshape(2, -1)`
    * `data.reshape(-1)` to flatten matrix
    * `.flatten()` 
* `.ravel()` returns view of array
* `np.zeros_like(data)` vs `np.ones_like(data)` returns matrix of 0's and 1's
    * `full_like(parent, 3)` for custom value
    * `np.empty(4)` for random RAM data
    * `np.eye(4)` for identity matrix


<br>
<br>  

# **Pandas**  
### **Functions**


* `drop_duplicates()`
* `loc` uses labels while `iloc` uses indexes  
    * `wine.loc[:, "country"].value_counts().sort_index()`  
    * `df.loc[(df['date'].dt.year == 2012) & (df['city'] == 'madrid')]`
* `reset_index()` - useful for filling in
* `describe()` -
* `info()` -
* `apply()` - for applying function,`axis=0` col-wise `axis=1` row-wise 
* `df.isnull().values.any()` - check if DF has null values
* `idxmax()` - extract function by index
    * `df['sales'].idxmax()`
* `.size()` counts NA's while `.count()` does not
* `df.drop_duplicates(subset='col1')` returns df with unique values of col1

<br>  

### **Commands**
* `value_counts()`
* `sort_index()`
* select unique
    * `nunique()` number of unique non-null values
    * `unique()` all unique values as a List
    
* double loc for filtering  
    * `us = wine.loc[wine.loc[:, "country"]=="US"]`
* setup args variable:  
    * `aggs = {"price": "mean", "points": "mean"}`   
    * `aggs = {"price": ["min", "mean", "max", "std"]}`
* droplevel(level=0)
* sort_values() 
    `sort_values(by="mean", ascending=False)`
* filter, group by and find max by index.
    * `df.loc[df['name'].isin(['pepsi'])].groupby('item').size().idxmax()`
    * `df.groupby('group1', as_index=False)`
* check for null values in df - `df.isnull().values.any()`

<br>  

# **Dates**
* `datetime.strptime()` to extract datetime from string  
    * `datetime.strptime(s, "%d %B, %Y")`
* `strftime()` to convert datetime to string  
    * `d.strftime("%Y-%m-%d")`
* resample() to combine data in different ways  
    * `energy.resample("D").mean()` gets avg per day (D) 
*  

  <br>  

# **List Comprehension**
* for loop: `np.array([10**n for n in range(6)]).reshape(3,2)`
* for loop with if statement
    * `for i, index_level in enumerate(index_level_names):`  
    `aggregated.index.set_names(index_level, level=i, inplace=True)`
    
* 



    
