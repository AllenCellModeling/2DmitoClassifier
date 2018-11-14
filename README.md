# MitoClassifier V1.0

This branch contains the files used for applying the X Y Z models and choosing the outcome by minimum entropy. The source
for training the X Y Z models will be committed in a separate branch. 

## Project Files:
**LoadPTModelsToDb.py** &rarr; Script to load trained models into MLDB (modeling database).

**IngestData.py** &rarr; Load the data from the Image Processing Pipeline (data_jobs_out.csv) into MLDB.

**MinEntropy.py** &rarr; Pull data from MLDB, apply 3 models, take min entropy, populate MLDB with MitosisLabelPredicted.

**CM_maker.py** &rarr; Pull data from MLDB and plot the confusion matrix for MitosisLabel vs MitosisLabelPredicted.

**src/model_analysis.py** &rarr; Library to create confusion matrix.

**src/train_model.py** &rarr; Library containing train_model() function.

**src/plot_images.py** &rarr; Library for plotting images using PIL.

## Running MinEntropy.py

Launch docker image rorydm/pytorch_extras:jupyter with flags -v /allen/aics:/allen/aics 
(so the filepaths match the database paths), assume it launches with name fancy_pants.
Once the image is running datasetdatabase must be pip installed.
Finally to execute the code:
``` 
TTY 1: (or use rory's bash script)
> docker run --rm -ti --ipc=host --runtime=nvidia -e "PASSWORD=jup_passwd" \
        -p 9199:9999 \
        -p 9106:6006 \
        -v /allen/aics:/allen/aics \
        rorydm/pytorch_extras:jupyter \
        bash -c "jupyter lab --allow-root --NotebookApp.iopub_data_rate_limit=10000000"
```
This creates a docker process which we'll say got the generated name 'fancy_pants'.
```
TTY 2:
> docker exec -it fancy_pants /bin/bash
#root> pip install git+https://github.com/AllenCellModeling/aics_modeling_db.git
#root> python MinEntropy.py
```
