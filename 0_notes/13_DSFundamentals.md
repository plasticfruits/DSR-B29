# Data Science Fundamentals

## 
* "Data Analytics is about the past, ML is about the future"
* "We have data" - Ok, but is it actually useful for explaining the data?
* "90% of the problems int eh industry are supervised problems"
* CRISP-DM: process for data mining
    - Business understanding
        - whats the impact / cost-saving of model?
        - (what the key, false + or fals - ?)
    - Data understanding
    - Data Prep
    - Modelling
        - ideal scenario: 3 months from start modelling to deployment
    - Evaluation
        -(Back to) business understanding
        - Deployment

## ML Processes

### 0. Framing the problem

### 1. Collecting data
* how to find target and features?
* is your data labelled?
* is your data stable? (i.e. staying the same over time)
* is the data likely to help you model?
* do you have enough positive labels?

### 2. Data cleaning
* quality: outliers / NA's
* quantity: rows & cols
* diversity: does distribution match the test set?
* cardinality: number of unique values
    - problem: by time of deployment company had 10 prods, 1 year later had 20
* dimensionality
* sparsity