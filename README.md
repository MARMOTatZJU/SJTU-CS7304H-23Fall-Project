# SJTU CS7304H Statistical Learning 23 Fall Project


Project repository for the course *SJTU CS7304H Statistical Learning 23 Fall*.

**DISCLAIMER**: THE AUTHOR WOULD RECOMMEND TO USE THIS REPOSITORY ONLI IN CASE OF GETTING BLOCKED BY TECHNICAL/IMPLEMENTATION ISSUES.


## Preparing Data

Either download the data through Kaggle API:

```
# download Kaggle API

rsync -av path/to/your/kaggle.json ~/.kaggle/kaggle.json
# Reference: https://github.com/Kaggle/kaggle-api
kaggle competitions download -c sjtu-cs7304h-statistical-learning-23-fall-project
```

, or manually download the data and place the files under `./data`.


## Environment Setup

```
pip install -r requirements.txt [-i https://pypi.tuna.tsinghua.edu.cn/simple]
```

For those using CUDA 11, install compatible Torch-related packages through the following command:

```
pip install --force-reinstall torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```



## Usage

### Train a Single Model

Train a given model (`${MODEL_NAME}`) on the full training set, export the model at `./${MODEL_NAME}/classifier.pkl`, and then prepare the submission csv `./${MODEL_NAME}/submission.csv`.

```
# svm with hard margin
python test_model.py --model="svm-h"

# svm with soft margin
python test_model.py --model="svm-s"

# mlp
python test_model.py --model="mlp"

# mlp with hyper-parameter found in model-selection phase
python test_model.py --model=mlp-with-model-selection
```


### Model Selection

Currenlty hardcoded for MLP.

```
python test_model_selection.py
```

The model and thecross-validation result (`model_selector.pkl`), best parameter settings (`best_param.json`), and the submission csv (`submission.csv`) will be exported under `./model_selection_results/${THE_FORMATED_DATETIME]}/`.


### Model Ensemble

```
# ensemble by averaging on proability
python test_ensemble.py  --ensemble_method "avg_on_proba"

# ensemble by averaging on logit
python test_ensemble.py  --ensemble_method "avg_on_logit"

# ensemble by voting
python test_ensemble.py  --ensemble_method "voting"
```

The ensemble reuslt will be placed at `./ensemble_submissions.csv`.


## Tips

### Clear processes

```
# Kill all processes matching the string "myProcessName":
ps -ef | grep 'myProcessName' | grep -v grep | awk '{print $2}' | xargs -r kill -9
```

Source: https://stackoverflow.com/questions/8987037/how-to-kill-all-processes-with-a-given-partial-name

### Infinite checkpoints

Extremely useful during model ensemble:

```
while true;do time CUDA_VISIBLE_DEVICES=6 python test_model_selection.py;done
```
