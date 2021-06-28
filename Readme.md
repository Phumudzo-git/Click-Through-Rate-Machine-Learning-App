## Getting Started
1. In your Windows Terminal, create a working directory to save the scripts/files from this repo as follows:
```
$ mkdir <work-diectory>
$ cd <work-directory>
```
Upload the scripts/files and save them inside the created directory

## Then create a virtual environment in windows systems as follows:

```
$ virtualenv -p python3 .
```
Activate the virtual environment with the following command,

```
$.\Scripts\activate
```
Then run the requirements.txt file to install the required packages as follows:
```
$ pip install requirements.txt
```
To run the pythom ml model, the following command is used,
```
$ python model.py
```
Or, for better visualization of the model data preparation, model training and evaluation, the jupyter notebook model.ipynb should be preferred. First run the following comand to install the Jupyter IDE,

```
$ pip install jupyter notebook
```
To run the App server and access the frontend of the App, the following command is used,
```
$ python server.py
```
To access the frontend you go to the http localhost url given after running the server.py as shown below,

Restarting with stat
 * Debugger is active!
 * Debugger PIN: 262-454-158
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

On the frontend, you are required to fill the field entries, after completion enter the field button Submit,
then your field entries are sent to a machine learning model for prediction. The model will either predict, Click or No Click, and the results of the model will be returned (presented) on the frontend.

