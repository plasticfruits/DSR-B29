# **Key Python Notes**

- `dir()` to get info on functions 

- **Useful Libs**:
    - isort: shorts libraries imported
    - 

- **Some useful links:** 
    - R to Pandas: https://pandas.pydata.org/pandas-docs/stable/getting_started/comparison/comparison_with_r.html  
    - R to Pandas 2: https://gist.github.com/conormm/fd8b1980c28dd21cfaf6975c86c74d07  
    - undersampling & over-sampling to reduce bias in datasets (e.g. recipes...)

- **Requirements.txt**
Export requirements.txt file:
    - `pip list --format=freeze > requirements.txt`
<br>

# Timing tools
* `%timeit` add it to line to print time: 
    - Exmaple: `%timeit cross_val_score(reg, X, y, cv=3)`

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

### Load Data
* import form multiple JSON files:
    ```
    import glob

    dfList = []
    for file in sorted(glob.glob("./data/return-data/*")):
        one_day_df = pd.read_json(file, lines=True)
        dfList.append(one_day_df)
    df = pd.concat(dfList).reset_index(drop=True)
    ```

### **NA's**

* `store_clean['StoreType'].value_counts()`
* `store_clean[store_clean['Store'].isna()]`
* `.fillna('missing')`
* `print(df.isnull().sum())`print number of nas by col
### **Filtering**
* `data.drop(columns="Survived")` drop column 
    - drop by defualt wants to drop rows
* filter df for specific values in a column
    ```
    list_of_vals = ["a", "b"]
    col_name = "PClass"
    bool_in_list = data.loc[:, col_name].isin(list_of_vals)
    data.loc[bool_in_list, :]
    ```
* `rossman_df[~((rossman_df.Sales<1)&(rossman_df.Open==1))]`



### **Env adjustments**
* precision options: 
    ``` 
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    %precision 4
    np.set_printoptions(precision=4, suppress=True)
    ``` 

### **Libraries**
* `pd.optios.plotting.backend = 'plotly'` - to use plotly express as default for   
    * Example:
    ``` 
    import pandas as pd
    pd.options.plotting.backend = 'plotly'

    df.plot.bar(
        x='patient', 
        y='Result', 
        color='Treatment',
        barmode='group',
        title='Medical Treatment Results'
    )
    ``` 

<br>  

### **Functions**
* `drop_duplicates()`
* `loc` uses labels while `iloc` uses indexes  
    * `wine.loc[:, "country"].value_counts().sort_index()`  
    * `df.loc[(df['date'].dt.year == 2012) & (df['city'] == 'madrid')]`
* `reset_index()` - useful for filling in
* `describe()` -
* `info()` -
* `df.dtypes()`
* `apply()` - for applying function,`axis=0` col-wise `axis=1` row-wise 
* `df.isnull().values.any()` - check if DF has null values
* `idxmax()` - extract function by index
    * `df['sales'].idxmax()`
* `.size()` counts NA's while `.count()` does not
* `df.drop_duplicates(subset='col1')` returns df with unique values of col1
* get_dummies() - to encode variables
    - `pd.get_dummies(data.loc[:, "Sex"], drop_first=True)`
    - You need an *exhaustive list of categories* to avoid model from braking or make an "other" category for all un-important vars

* Best practices:
    ```
    from typing import List, Dict, Tuple, Any

    ["string1", "string2"]
    """
    This function is blablabla
    sex (int): is klaklakla
    Return:
        float blablabla
    """
    def some_func(List[str]) -> str:
        pass
    ``` 

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
* `[x+1 for x in [1,2,3]]`
* for loop: `np.array([10**n for n in range(6)]).reshape(3,2)`
* for loop with if statement
    * `for i, index_level in enumerate(index_level_names):`  
    `aggregated.index.set_names(index_level, level=i, inplace=True)`
    
* **Tidy** from wide to long
    ```
    # original df format: [patient, treatment_a, treatment_b]
    # tidy format: [patient, treatment, result]
    
    tidy = messy.melt(
        id_vars=['patient'],
        value_vars=['Treatment A', 'Treatment B'],
        var_name='Treatment',
        value_name='Result',
    )
    ```
* **Concat** - to tidy a DF by adding new common column
    ```
    # original format: [Xi, Yi, Xf, Yf, color]
    # tidy format: [x, y, color, initial]
    
    result = pd.concat([
        table[['Xi','Yi','color']].assign(initial=True).rename(columns={'Xi':'x', 'Yi':'y'}),
        table[['Xf','Yf','color']].assign(initial=False).rename(columns={'Xf':'x', 'Yf':'y'}),
    ])
    ```


* **Loops**
    * using enumerate:
    ``` 
    for i, model in enumerate[a, b]:
        print(i)
        print(model)
    ```
    * using `zip()`
    ``` 
    for name, age, country in zip(
        ["A", "B", "C"],
        [20, 21, 22],
        ["D", "E", "F"]
    ):
        print(name)
        print(age)
        print(country)
    ```
        
