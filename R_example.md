---
title: Habits
author: John Doe
date: March 22, 2005
output: 
 html_document: 
  theme: flatly
---

# Data crunch example R script
sweet-richard  
Jan 29, 2022 

Last updated: 2022-01-29 21:48:44
required packages: tidyverse, feather, xgboost
subfolders *data* and *submission* assumed to exist.


```r
library(tidyverse)
library(feather)
```

## Parameters


```r
file_name_train = "train_data.feather"
file_name_test ="test_data.feather"
is_download = FALSE # set this to true to download new data
is_upload = FALSE # set this to true to upload a submission
nrounds = 100 # in this kind of problem, more round are used at xgboost, 500 is just used for speed
```

## Functions


```r
getCorrMeasure = function(actual, predicted) {
  cor_measure = cor(actual, predicted, method="spearman")
  return(cor_measure)
}
```

## Download data
after the download, data is stored in feather format to be read on demand quickly


```r
if( is_download ) {
  
  cat("\n start  download")
  train_datalink_X = 'https://tournament.datacrunch.com/data/X_train.csv'  
  train_datalink_y = 'https://tournament.datacrunch.com/data/y_train.csv' 
  hackathon_data_link = 'https://tournament.datacrunch.com/data/X_test.csv' 
  
  train_dataX = read_csv(url(train_datalink_X))
  train_dataY = read_csv(url(train_datalink_y))
  test_data  = read_csv(url(hackathon_data_link))
  
  train_data = 
    bind_cols( train_dataX, train_dataY)
  
  train_data = train_data %>% mutate_at(vars(starts_with("feature_")), list(~as.integer(.*100)))
  feather::write_feather(train_data, path = paste0("./data/", file_name_train))
  
  test_data = test_data %>% mutate_at(vars(starts_with("feature_")), list(~as.integer(.*100)))
  feather::write_feather(test_data, path = paste0("./data/", file_name_test))
  
  names(train_data)
  
  nrow(train_data)
  nrow(test_data)
  
  cat("\n data is downloaded")
} else {
  train_data = feather::read_feather(path = paste0("./data/", file_name_train))
  test_data =  feather::read_feather(path = paste0("./data/", file_name_test))
}

## set vars used for modelling 
model_vars = setdiff(names(test_data), c("id","Moons"))
```

## Fit xgboost


```r
library(xgboost)

# custom loss function for eval  
corrmeasure <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  corrm <- as.numeric(cor(labels, preds, method="spearman"))
  return(list(metric = "corr", value = corrm))
}
  
eval_metric_string =  "rmse"
my_objective = "reg:squarederror"
  

tree.params = list(
  booster = "gbtree", eta = 0.01, max_depth = 5,
  tree_method = "hist", #  tree_method = "auto",
  objective = my_objective)

cat("\n starting xgboost \n")
```

```
## 
##  starting xgboost
```

```r
# first target target_r then g and b
################
current_target = "target_r"
  
dtrain = xgb.DMatrix(train_data %>% select(one_of(model_vars)) %>% as.matrix(), label = train_data %>% select(one_of(current_target))  %>% as.matrix())
  
xgb.model.tree = xgb.train(data = dtrain,  
                           params = tree.params, nrounds = nrounds, verbose = 1, 
                           print_every_n = 500L, eval_metric = corrmeasure) 
  
xgboost_tree_train_pred1 = predict(xgb.model.tree, train_data %>% select(one_of(model_vars)) %>% as.matrix())
xgboost_tree_live_pred1 = predict(xgb.model.tree, test_data %>% select(one_of(model_vars)) %>% as.matrix())
  
cor_train = getCorrMeasure(train_data %>% select(one_of(current_target)), xgboost_tree_train_pred1)
  
cat("\n : metric: ", eval_metric_string, "\n")
```

```
## 
##  : metric:  rmse
```

```r
print(paste0("Corrm on train: ", round(cor_train,4)))
```

```
## [1] "Corrm on train: 0.2055"
```

```r
print(paste("xgboost", current_target, "ready"))
```

```
## [1] "xgboost target_r ready"
```

```r
# second target target_g
################
current_target = "target_g"
  
dtrain = xgb.DMatrix(train_data %>% select(one_of(model_vars)) %>% as.matrix(), label = train_data %>% select(one_of(current_target))  %>% as.matrix())

xgb.model.tree = xgb.train(data = dtrain,  
                           params = tree.params, nrounds = nrounds, verbose = 1, 
                           print_every_n = 500L, eval_metric = corrmeasure) 

xgboost_tree_train_pred2 = predict(xgb.model.tree, train_data %>% select(one_of(model_vars)) %>% as.matrix())
xgboost_tree_live_pred2 = predict(xgb.model.tree, test_data %>% select(one_of(model_vars)) %>% as.matrix())

cor_train = getCorrMeasure(train_data %>% select(one_of(current_target)), xgboost_tree_train_pred2)

cat("\n : metric: ", eval_metric_string, "\n")
```

```
## 
##  : metric:  rmse
```

```r
print(paste0("Corrm on train: ", round(cor_train,4)))
```

```
## [1] "Corrm on train: 0.2581"
```

```r
print(paste("xgboost", current_target, "ready"))
```

```
## [1] "xgboost target_g ready"
```

```r
# third target target_b
################
current_target = "target_b"

dtrain = xgb.DMatrix(train_data %>% select(one_of(model_vars)) %>% as.matrix(), label = train_data %>% select(one_of(current_target))  %>% as.matrix())

xgb.model.tree = xgb.train(data = dtrain,  
                           params = tree.params, nrounds = nrounds, verbose = 1, 
                           print_every_n = 500L, eval_metric = corrmeasure) 

xgboost_tree_train_pred3 = predict(xgb.model.tree, train_data %>% select(one_of(model_vars)) %>% as.matrix())
xgboost_tree_live_pred3 = predict(xgb.model.tree, test_data %>% select(one_of(model_vars)) %>% as.matrix())

cor_train = getCorrMeasure(train_data %>% select(one_of(current_target)), xgboost_tree_train_pred3)

cat("\n : metric: ", eval_metric_string, "\n")
```

```
## 
##  : metric:  rmse
```

```r
print(paste0("Corrm on train: ", round(cor_train,4)))
```

```
## [1] "Corrm on train: 0.284"
```

```r
print(paste("xgboost", current_target, "ready"))
```

```
## [1] "xgboost target_b ready"
```

## Submission
simple histograms to check the submissions


```r
hist(xgboost_tree_live_pred1)
```

![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-6-1.png)

```r
hist(xgboost_tree_live_pred2)
```

![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-6-2.png)

```r
hist(xgboost_tree_live_pred3)
```

![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-6-3.png)

create submission file


```r
sub_df = tibble(target_r = xgboost_tree_live_pred1, 
                target_g = xgboost_tree_live_pred2, 
                target_b = xgboost_tree_live_pred3)
  
file_name_submission = paste0("gbTree_", gsub("-","",Sys.Date()), ".csv")

sub_df %>% readr::write_csv(file = paste0("./submissions/", file_name_submission))
nrow(sub_df)
```

```
## [1] 347878
```

```r
cat("\n submission file written")
```

```
## 
##  submission file written
```

## Upload submission


```r
if( is_upload ) {
  library(httr)
  
  API_KEY = "YourKeyHere"
  
  response <- POST(
    url = "https://tournament.crunchdao.com/api/v2/submissions",
    query = list(apiKey = API_KEY),
    body = list(
      file = upload_file(path = paste0("./submissions/", file_name_submission))
    ),
    encode = c("multipart")
  );
  
  status <- status_code(response)
  
  if (status == 200) {
    print("Submission submitted :)")
  } else if (status == 400) {
    print("ERR: The file must not be empty")
    print("You have send a empty file.")
  } else if (status == 401) {
    print("ERR: Your email hasn't been verified")
    print("Please verify your email or contact a cruncher.")
  } else if (status == 403) {
    print("ERR: Not authentified")
    print("Is the API Key valid?")
  } else if (status == 404) {
    print("ERR: Unknown API Key")
    print("You should check that the provided API key is valid and is the same as the one you've received by email.")
  } else if (status == 409) {
    print("ERR: Duplicate submission")
    print("Your work has already been submitted with the same exact results, if you think that this a false positive, contact a cruncher.")
    print("MD5 collision probability: 1/2^128 (source: https://stackoverflow.com/a/288519/7292958)")
  } else if (status == 422) {
    print("ERR: API Key is missing or empty")
    print("Did you forget to fill the API_KEY variable?")
  } else if (status == 423) {
    print("ERR: Submissions are close")
    print("You can only submit during rounds eg: Friday 7pm GMT+1 to Sunday midnight GMT+1.")
    print("Or the server is currently crunching the submitted files, please wait some time before retrying.")
  } else if (status == 423) {
    print("ERR: Too many submissions")
  } else {
    content <- httr::content(response)
    print("ERR: Server returned: " + toString(status))
    print("Ouch! It seems that we were not expecting this kind of result from the server, if the probleme persist, contact a cruncher.")
    print(paste("Message:", content$message, sep=" "))
  }
  
  # DEVELOPER WARNING:
  # THE API ERROR CODE WILL BE HANDLER DIFFERENTLY IN A NEAR FUTURE!
  # PLEASE STAY UPDATED BY JOINING THE DISCORD (https://discord.gg/veAtzsYn3M) AND READING THE NEWSLETTER EMAIL
}
```

