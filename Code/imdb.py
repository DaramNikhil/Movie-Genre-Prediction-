import numpy as np
from flask import Flask,render_template,url_for,request
import pickle
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
classification= pickle.load(open("imdb.pkl","rb"))
model = pickle.load(open('genre.pkl','rb'))


app = Flask(__name__,template_folder='/storage/emulated/0/')

@app.route('/',methods=["POST","GET"])
def home_page():
    if request.method=="GET":
        return render_template('index.html')
    else:
        plot_ = request.form['story']
        story = [plot_]
        classifier = classification.fit_transform(story).toarray()
        prediction = model.predict(classifier)
        return render_template('genre.html', prediction_by= prediction)
        
                
    
    

if __name__ == '__main__':
    app.run(debug = True)