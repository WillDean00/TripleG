library(tidymodels)
library(embed)
library(vroom)
library(kernlab)
library(themis)

train <- vroom("./train.csv")
test <- vroom("./test.csv")
view(train)


train$type <- as.factor(train$type)

my_recipe <- recipe(type ~ ., data = train) %>%
  step_mutate(id, features = id) %>%
  step_mutate(color = as.factor(color)) %>% 
  step_normalize(all_numeric_predictors()) 

prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)

vmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% 
  set_mode("classification") %>%
  set_engine("kernlab")

GGG_workflow_SVMS <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(vmRadial)

tuning_grid_SVMS <- grid_regular(rbf_sigma(),
                                 cost(),
                                 levels = 3)

folds_SVMS <- vfold_cv(train, v = 3, repeats=3)

CV_results_SVMS <- GGG_workflow_SVMS %>% 
  tune_grid(resamples = folds_SVMS,
            grid = tuning_grid_SVMS,
            metrics = metric_set(accuracy))

bestTune <- CV_results_SVMS %>%
  select_best()

final_wf_SVMS <-
  GGG_workflow_SVMS %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

final_wf_SVMS %>%
  predict(new_data = train, type = "class")

GGG_pred_SVMS <- predict(final_wf_SVMS,
                            new_data = test,
                            type = "class")

kaggle_submission <- GGG_pred_SVMS %>% 
  bind_cols(., test) %>% 
  rename(type = .pred_class) %>% 
  select(id, type)

vroom_write(x = kaggle_submission, file = "./GGG_SVMS.csv" , delim = ",")


library(discrim)
library(naivebayes)

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") 

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

tuning_grid_bayes <- grid_regular(Laplace(),
                                  smoothness(),
                                  levels = 3)

folds_nb <- vfold_cv(train, v = 3, repeats=3)

CV_results_nb <- nb_wf %>% 
  tune_grid(resamples = folds_nb,
            grid = tuning_grid_bayes,
            metrics = metric_set(accuracy))

bestTune <- CV_results_nb %>%
  select_best()

final_wf_nb <-
  nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

final_wf_nb %>%
  predict(new_data = train, type="class")

GGG_pred_nb <- predict(final_wf_nb,
                          new_data = test,
                          type = "class")

kaggle_submission <- GGG_pred_nb %>% 
  bind_cols(., test) %>% 
  rename(type = .pred_class) %>% 
  select(id, type)

vroom_write(x = kaggle_submission, file = "./GGG_NB.csv" , delim = ",")
