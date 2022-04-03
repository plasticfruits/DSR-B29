# **Plotyl**  

<br>  
* Check https://streamlit.io/ !!!
* Use as default with Pands: `pd.options.plotting.backend = 'plotly'`
    * use `df.plot(...)` to call plotly :)

## **Plotly RAW**  

* Full example:
``` 
import plotly as py # Import the library
import plotly.graph_objs as go # Building blocks of Plotly plots
import numpy as np

# Constructing a figure
x = np.linspace(0, 2*np.pi)

# Traces
trace0 = dict(
    type='scatter', 
    x=x, 
    y=np.sin(x), 
    name='sin(x)'
)
trace1 = dict(
    type='scatter', 
    x=x, 
    y=np.cos(x), 
    name='cos(x)'
)

# Layout
layout = dict(
    title='SIN and COS functions',
    xaxis=dict(title='x'),
    yaxis=dict(title='f(x)')
)

# Figure
fig = go.Figure(data=[trace0, trace1], layout=layout) 
```


* declaring a python dictionary:
    * `dict(a = 3, b = 4)`
    * `{"a": 3, "b": 4}`

* Defining a trace: 
    ```
    trace0 = dict(
    type='scatter', 
    x=x, 
    y=np.sin(x), 
    name='sin(x)'
    mode = 'lines'
    )
    ```

    * Modes that can be passed to definition of a trace:
        * `lines` (default)
        * `lines+markers`
        * `markers`

* Adding title and axis names:
    ```
    layout = dict(
    title='SIN and COS functions',
    xaxis=dict(title='x'),
    yaxis=dict(title='f(x)')
    )
    ```
* Don't display plotly logo:
    `fig.show(config= {'displaylogo': False})`

<br>  

## **Subplots**  

* import:`from plotly.subplots import make_subplots`
* full example code:
    ````
    # Teacher code
    fig = (
        make_subplots(
            rows=2, 
            cols=1, 
            subplot_titles=('Initial values', 'Final values'), 
            shared_xaxes=True,
            vertical_spacing=0.05,
        )
        .add_scatter(
            x=table.Xi,
            y=table.Yi, 
        #    marker=dict(color=table.color), # color for this trace only
            name='Initial',
            row=1,
            col=1,
        )
        .add_scatter(
            x=table.Xf,
            y=table.Yf,
            hovertext=table.color, 
        #    marker=dict(color=table.color), # color for this trace only
            name='Final',
        #    showlegend=False, # for this trace only
            row=2,
            col=1,
        )
        .update_layout(
            title = 'Random clusters',
            xaxis2_range=[0,500],
            width = 600,
            height = 900,
            showlegend = False,
        )
        .update_xaxes(
            title='X', 
        #    range=[0, 500],
            row=2,
        )
        .update_yaxes(
            title='Y', 
            range=[0, 500],
        )
        .update_traces(
            mode='markers',
            marker_color=table.color, # we set here so applies to all
        )
    )
    ```
<br>  

## **Inspect and Export plots**  

* `fig.to_dict()`  introspect dictionary representation
* `fig.full_figure_for_development(as_dict=True)` - show defaults
* `fig_show()`- display it in the notebook
    * `fig_show(render='png')` 
* `fig.write_html(file='sin_cos.html', include_plotlyjs='cdn')` - export as stand-alone html file with a source to cdn (content delivery network) 
* `fig.write_html(file='sin_cos.html', include_plotlyjs=False, full_html=False)` - export as <div> to paste in webpage (won't render alone)
* `fig.write_image(file='sin_cos.png', width=700, height=500)` - export as static png image. NOTE: you have to specify WxH
 
 <br>  

## **Plotly Express**   

* Example:
    ```
    import plotly.express as px

    px.scatter(
        data_frame= gapminder[gapminder.year.isin([1952, 2007])], 
        x= 'gdpPercap', 
        y= 'lifeExp', 
        log_x = True,
        color= 'continent', 
        size= 'pop',
        size_max= 60, 
        facet_col='year',
        width= 800,
        height= 500,
        title= 'Life Expectancy vs. GDP per capita',
    )
    ```

* You could continue modifying your plotly express figure by following previous figure with `.update_layout(...)`

<br>  

## Pandas Profiling
* `from pandas_profiling import ProfileReport`
* `profile = ProfileReport(data, title="Pandas Profiling Report", explorative=True)`

## **Hosting**  
* GitHub pages recommended for hosting images
    * upload as `index.html` to display directly, in repo go to settings/pages/ set to main and save.

## **DASH**  
* `pip install plotly dash gunicorn``
*  
