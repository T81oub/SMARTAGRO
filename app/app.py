
from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd

app=Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')
if __name__ == "__main__":
    app.run(debug=True)
