library(tidymodels)
library(embed)
library(vroom)
library(bonsai)
library(lightgbm)

train <- vroom("./train.csv")
test <- vroom("./test.csv")
view(train)


train$type <- as.factor(train$type)


my_recipe <- recipe(type ~ ., data=train) %>%
  step_mutate(color = as.factor(color)) %>% 
  step_normalize(all_numeric_predictors()) 

prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)

boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
set_engine("lightgbm") %>%
  set_mode("classification")

GGG_workflow_BOOST <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_model)

tuning_grid_BOOST <- grid_regular(tree_depth(),
                                 trees(),
                                 learn_rate(),
                                 levels = 5)

folds_BOOST <- vfold_cv(train, v = 2, repeats=1)

CV_results_BOOST <- GGG_workflow_BOOST %>% 
  tune_grid(resamples = folds_BOOST,
            grid = tuning_grid_BOOST,
            metrics = metric_set(accuracy),
             control = control_grid(verbose = TRUE))

 bestTune <- CV_results_BOOST %>%
  select_best()

final_wf_BOOST <-
  GGG_workflow_BOOST %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

final_wf_BOOST %>%
  predict(new_data = train, type = "class")

GGG_pred_BOOST <- predict(final_wf_BOOST,
                         new_data = test,
                         type = "class")

kaggle_submission <- GGG_pred_BOOST %>% 
  bind_cols(., test) %>% 
  rename(type = .pred_class) %>% 
  select(id, type)

vroom_write(x = kaggle_submission, file = "./GGG_BOOST.csv" , delim = ",")

