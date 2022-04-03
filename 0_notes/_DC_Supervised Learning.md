# Supervised learning with Scikit-Learn @DataCamp

y = df['party'].values
X = df.drop('party', axis=1).values

* `df.keays()` print col names
`.values` attribute ensures that they are npArrays


* `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)`
    - stratify ensure labels are equally distributed across sets

* Fit on train `knn.fit(X_train, y_train)`
* Predict on test: `knn.score(X_test)`
* Score on test: `knn.score(X_test, y_test)`


## K-Neareest Neighbors
* Default `model.score(X, y)`: Accuracy -- fraction of correct predictions   
* loop with different k (neighbors) and plot
    ``` 
    # Setup arrays to store train and test accuracies
    neighbors = np.arange(1, 9)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    # Loop over different values of k
    for i, k in enumerate(neighbors):
        # Setup a k-NN Classifier with k neighbors: knn
        knn = KNeighborsClassifier(n_neighbors=k)

        # Fit the classifier to the training data
        knn.fit(X_train, y_train)
        
        #Compute accuracy on the training set
        train_accuracy[i] = knn.score(X_train, y_train)

        #Compute accuracy on the testing set
        test_accuracy[i] = knn.score(X_test, y_test)

    # Generate plot
    plt.title('k-NN: Varying Number of Neighbors')
    plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
    plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.show()
    ``` 

## Regression

### Linear Regression
* Default `model.score(X, y)`: R2 - quantifies amount of variance in target variable predicted from the feature variables
* RMSE: `rmse = np.sqrt(mean_squared_error(y_test, y_pred))`


### Cross-validation
k-fold CV (cv=5 standard practice)
* `cv_scores = cross_val_score(reg, X, y, cv=5)`
    - good practice to check mean: `np.mean(cv_scores)`

### Regularized regression
*  Penalises large coefficients to avoid overfitting
* Ridge regression: "first choice for regression models"
    - penalises large coefficients the higher the alpha value is set to
    - Example:
        ```
        # Import necessary modules
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score

        # Setup the array of alphas and lists to store scores
        alpha_space = np.logspace(-4, 0, 50)
        ridge_scores = []
        ridge_scores_std = []

        # Create a ridge regressor: ridge
        ridge = Ridge(normalize=True)

        # Compute scores over range of alphas
        for alpha in alpha_space:

            # Specify the alpha value to use: ridge.alpha
            ridge.alpha = alpha
            
            # Perform 10-fold CV: ridge_cv_scores
            ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
            
            # Append the mean of ridge_cv_scores to ridge_scores
            ridge_scores.append(np.mean(ridge_cv_scores))
            
            # Append the std of ridge_cv_scores to ridge_scores_std
            ridge_scores_std.append(np.std(ridge_cv_scores))

        # Display the plot using function in ./utils.py
        display_plot(ridge_scores, ridge_scores_std)
        ``` 
* Lasso regression: used to select important features   
    - Example:
    ````
    # Import Lasso
    from sklearn.linear_model import Lasso

    # Instantiate a lasso regressor: lasso
    lasso = Lasso(alpha=0.4, normalize=True)

    # Fit the regressor to the data
    lasso.fit(X,y)

    # Compute and print the coefficients
    lasso_coef = lasso.fit(X,y).coef_
    print(lasso_coef)

    # Plot the coefficients
    plt.plot(range(len(df_columns)), lasso_coef)
    plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
    plt.margins(0.02)
    plt.show()
    ````


### Logistic Regression (Classification)
* Generate the confusion matrix and classification report
    - `print(confusion_matrix(y_test, y_pred))`
    - `print(classification_report(y_test, y_pred))`
* **ROC Curve**: provides a way to visually evaluate models
    ```
    # Import necessary modules
    from sklearn.metrics import roc_curve

    # Compute predicted probabilities: y_pred_prob
    y_pred_prob = logreg.predict_proba(X_test)[:,1]

    # Generate ROC curve values: fpr, tpr, thresholds
    fpr, tpr, thresholds = roc_curve(y_test,y_pred_prob)

    # Plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    ``` 
* AUC: area under ROC curve, the > the better the model!
    - you can use CV with `scoring='roc_auc'` for estimating it or `roc_auc_score()`
    - If the AUC is greater than 0.5, the model is better than random guessing


### Hyperparameter Tuning
* Using `GridsearchCV`
* Using `RandomizedSearchCV`: less computational expensive as not all hyperparams are tried but a a fix number of settings. -- Great for Decission Trees


## Pipelines

### NA's
* impute missing values using `from sklearn.preprocessing   import Imputer`
* `print(df.isnull().sum())`
* `imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)`
* pass to pipeline: `steps = [('imputation', imp), ('model', clf)]`

* **Pipeline fro Classification:**
    ```
    # Setup the pipeline
    steps = [('scaler', StandardScaler()),
            ('SVM', SVC())]

    pipeline = Pipeline(steps)

    # Specify the hyperparameter space
    parameters = {'SVM__C':[1, 10, 100],
                'SVM__gamma':[0.1, 0.01]}

    # Create train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

    # Instantiate the GridSearchCV object: cv
    cv = GridSearchCV(pipeline, parameters)

    # Fit to the training set
    cv.fit(X_train, y_train)

    # Predict the labels of the test set: y_pred
    y_pred = cv.predict(X_test)

    # Compute and print metrics
    print("Accuracy: {}".format(cv.score(X_test, y_test)))
    print(classification_report(y_test, y_pred))
    print("Tuned Model Parameters: {}".format(cv.best_params_))

    ```

* **Pipeline for Regression:**
```
    # Setup the pipeline steps: steps
    steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
            ('scaler', StandardScaler()),
            ('elasticnet', ElasticNet())]

    # Create the pipeline: pipeline 
    pipeline = Pipeline(steps)

    # Specify the hyperparameter space
    parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}

    # Create train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Create the GridSearchCV object: gm_cv
    gm_cv = GridSearchCV(pipeline, parameters)

    # Fit to the training set
    gm_cv.fit(X_train, y_train)

    # Compute and print the metrics
    r2 = gm_cv.score(X_test, y_test)
    print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
    print("Tuned ElasticNet R squared: {}".format(r2))

```