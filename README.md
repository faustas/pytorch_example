## Prerequisites

You should have installed:

- python3
- venv (Virtual environment tool)
- pip

## Creating a new virtual Python environment

```shell
python3 -m venv env
```

## Activate the environment

```shell
source env/bin/activate
```

## Check which python environment is used

```shell
which python
```

## Leave the python environment

```shell
deactivate
```

## Create the requirements.txt file

```shell script
pip freeze > requirements.txt
``` 

# Install for the project

Activate the environment and install the packages.
```shell script
pip install -r requirements.txt
```
