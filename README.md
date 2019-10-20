# Lymphmeter

Model training:
•	We initially used machine-learning models, but the result was not so desirable. Code is included in "ML.py". In the end, we chose to use deep learning model.
•	Run “pip install –r requirements.txt” in terminal to set up the environment 
•	Run “python generate_model.py” to save deep learning model in "model.h5". This contains all the weights given to each symptom. This is the output generated from the code we have written in Keras.
•	"index.html" is our website page. 


Server (Nginx) Setup for MacOS:
•	Install Nginx. We recommend using homebrew: "brew install nginx"
•	Change the path in "nginx.conf" to link to our "index.html". Default place of "nginx.conf" on Mac after installing with brew is "/usr/local/etc/nginx/nginx.conf"
•	Give the command in terminal: "sudo nginx", to start nginx
•	Start our flask app by running "python DeepLearningModel.py" in project folder
•	Go to “localhost:8080” (if “localhost:8080” doesn’t work, modify nginx.conf and try “localhost:80”) in browser to see the webpage.
•	Fill in the form and click submit to see the result.
