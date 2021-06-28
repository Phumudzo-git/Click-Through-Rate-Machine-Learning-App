#import mode.classify # Importing the python file containing the Machine Learning model
from flask import Flask, request, render_template,jsonify # Importing Python flask libraries
import pickle
import numpy as np
filename = 'random_forest.sav'
rfc_model = pickle.load(open(filename, 'rb'))

# Initializing the flask class and indicating the templates directory
app = Flask(__name__,template_folder="templates")

# The default route will be set as 'home'
@app.route('/home')
def home():
    return render_template('home.html') # Rendering the home.html

class_mapping = {0: 'No click', 1: 'Click'}

def classify(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p):
    arr = np.array([a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p]) # Convert to numpy array
    arr = arr.astype(np.float64) # Change the data type to float
    query = arr.reshape(1, -1) # Reshape the array
    prediction = class_mapping[rfc_model.predict(query)[0]] # Retrieve from dictionary
    return prediction # Return the prediction

# Route 'classify' accepts GET request
@app.route('/classify',methods=['POST','GET'])
def classify_type():
    try:
        ui_component_position = request.args.get('uicomp') # Get parameters for ui component position
        device_type = request.args.get('dvt') # Get parameters for device type
        device_model = request.args.get('dm') # get parameters for device model
        site_domain = request.args.get('st') # get parameters for site domain
        app_id = request.args.get("ai") # get parameters for app id
        app_domain = request.args.get("ad") # get parameters for app domain
        app_category = request.args.get("ac") # app category
        device_conn_type = request.args.get('dct') # Get parameters for device conn type
        f0 = request.args.get('f0') # Get parameters for f0
        f1 = request.args.get('f1')
        f2 = request.args.get('f2')
        f3 = request.args.get('f3')
        f4 = request.args.get('f4')
        f5 = request.args.get('f5')
        f6 = request.args.get('f6')
        f7 = request.args.get('f7')

        # Get the output from the classification model
        results = classify(ui_component_position, device_type, device_model, site_domain, app_id, 
                            app_domain, app_category,device_conn_type, f0, f1, f2, f3, f4, f5, f6, f7)

        # Render the output in new HTML page
        return render_template('output.html', results=results)
    except:
        return 'Error'

# Run the Flask server
if(__name__=='__main__'):
    app.run(debug=True)        