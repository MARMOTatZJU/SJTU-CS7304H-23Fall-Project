# SJTU CS7304H Statistical Learning 23 Fall Project


## Preparing Data

```
rsync -av path/to/your/kaggle.json ~/.kaggle/kaggle.json
kaggle competitions download -c sjtu-cs7304h-statistical-learning-23-fall-project
```

Reference: https://github.com/Kaggle/kaggle-api

Please place the data under `./data`.


## Usage

### Model Training and Selection

```
python test_model_selection.py
```

The model, cross-validation result, best parameter settings, and the submission csv will be output under `./model_selection_results/${THE_FORMATED_DATETIME]}/`.

### Model Ensemble

```
python test_ensemble.py
```

The ensemble reuslt will be placed at `./ensemble_submissions.csv`.


## Tips

### Clear processes

```
# Kill all processes matching the string "myProcessName":
ps -ef | grep 'myProcessName' | grep -v grep | awk '{print $2}' | xargs -r kill -9
```

Source: https://stackoverflow.com/questions/8987037/how-to-kill-all-processes-with-a-given-partial-name

### infinite checkpoints

For model selection

```
while true;do time CUDA_VISIBLE_DEVICES=6 python test_model_selection.py;done
```
