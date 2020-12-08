# mids-w207-final-project

- Team Name: Clear Cut Solution 
- Kaggle Project: https://www.kaggle.com/c/forest-cover-type-prediction
- Class Spreadsheet: [Google Sheets Link](https://docs.google.com/spreadsheets/d/17Tett3QC_26hajUqKjYoVULeok2bLiaLSu3M7Y8WTso/edit?skip_itp2_check=true#gid=0)
- Final Presentation Slide Deck: [Google Slides Link](https://docs.google.com/presentation/d/1dMx_PfZBMRnDXwnIyE50zNRD70N-VKzRA0ZsEW-s_Po/edit?usp=sharing)

## Primary Files:

1. **exploratory_data_analysis.ipynb**: Jupyter notebook with a detailed analysis of the training data
2. **feature_engineering.py**:  Python library containing all transformations 
3. **models.py**: Python library containing all models and configurations 
4. **clear_cut_solution.ipynb**:  Jupyter notebook with descriptions, solutions and test results

## Repo Map

* `README.md`
  * Project introduction, file structure, environment instructions 
* `exploratory_data_analysis.ipynb`
  * Distributions, visualizations, sanity checks, correlation etc.  
* `clear_cut_solution.ipynb`
  * Formal project implementation with feature engineering, training, evaluation and testing
* `feature_engineering.py` and `models.py`
  * Libraries of functions for feature engineering and models Consumed in clear_cut_solution.py and also contains experimental code not included in final project. 
* `./data`
  * Notebook diagrams, training data, testing data
* `./submissions`
  * Test output files (csv) to be uploaded on Kaggle
* `./backups`
  * Html, markdown, and python versions of the clear_cut_solution notebook
* `./comp_setup`
  * Details of custom container creation 

  
## Computing Environment
  
Work was conducted in the `kmartcontainers/207final` container ([Dockerhub link](https://hub.docker.com/repository/docker/kmartcontainers/207final)). It is a custom container which adds the `xgboost` library to the `jupyter/tensorflow-notebook` docker container as put together by the jupyter development team. Details of how to set up the container to run on your machine or GCP as well as details of the container creation are in the `comp_setup/ComputeSetup.md` file.

