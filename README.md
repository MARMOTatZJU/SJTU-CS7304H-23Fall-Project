# SJTU CS7304H Statistical Learning 23 Fall Project


# Preparing Data

```
rsync -av path/to/your/kaggle.json ~/.kaggle/kaggle.json
kaggle competitions download -c sjtu-cs7304h-statistical-learning-23-fall-project
```

Reference: https://github.com/Kaggle/kaggle-api


# Tips

## Clear processes

```
# Kill all processes matching the string "myProcessName":
ps -ef | grep 'myProcessName' | grep -v grep | awk '{print $2}' | xargs -r kill -9
```

Source: https://stackoverflow.com/questions/8987037/how-to-kill-all-processes-with-a-given-partial-name
