# How to run
## Download pre-saved files and test dataset
- Go to [this drive folder](https://drive.google.com/drive/folders/1lztvyzh1L4jsmzaiADzAeFWNpdILECKS?usp=sharing) and download the `pre-saved` folder and the `test.csv`. Put them in the root of the project
## Install python with the same version
- Recommend installing python with [pyenv](https://github.com/pyenv/pyenv)
- Install python 3.7.13: `pyenv install 3.7.13`
- Use python 3.7.13 globally: `pyenv global 3.7.13`
## Install modules from Pipfile
- Install [pipenv](https://pypi.org/project/pipenv/)
- Install packages from Pipfile with `pipenv install`
## Run `uvicorn main:app`
- Run with virtual env: `pipenv run uvicorn main:app`
