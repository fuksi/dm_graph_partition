## What

Source code for programming project. This includes reference code ```spectra_lib.py``` and execution ```main.py```

## Result
Pre-run result is available in ```result``` folder
- result with unnorm_ prefix is for algorithm 2 (according to lecture slides)
- the rest are for algorithm 3
- ```summary.txt``` summarizes phi objective for all datasets

## Execution / Re-producing the result
Pipenv flow:
- ```pipenv installl``` to restore the package
- ```pipenv run python main.py``` to run the program and reproduce the result (this takes time)

Non-pipenv flow:
- Use pip to restore packaged defined in ```Pipfile```
- ```python main.py``` to run the program

## main.py

A wrapper method for the whole project. This method will, for each dataset:
- Find clusters using algorithm 2, 3
- Output to output file in ```result``` folder
- Evaluate the objective phi, and output to console and ```summary.txt``` in ```result``` folder

## spectral_lib.py
Partial methods used for the spectral algorithm, please see docstrings in file.


