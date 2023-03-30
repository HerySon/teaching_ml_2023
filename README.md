# Project description

This is the final project for my machine learning class

## Content directories

- data: store here your small additional datasets (<50Mo)
- models : store your trained models
- tests: store here your code dedicated to small tests 
- notebooks: store here your notebooks dedicated to analyses
- scripts: store here your code structured as python packages for automation of your workflow
- results : store here your results (pictures, model performance, ... )

## Steps to start working on the project

- Create a new virtual environment and activate it :

with pyenv virtualenv:
```bash
	pyenv virtualenv clustering_OFF && pyenv activate clustering_OFF
```
or conda:
```bash
	conda clustering_OFF && conda activate clustering_OFF
```

- Install required librairies :
```bash
  	pip install -r requirements
```

While you work on the project, don't forget to update the`requirements.txt`

