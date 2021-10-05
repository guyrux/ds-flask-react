import os

from flask import Flask, request, render_template
import pandas as pd
import pickle
from flask_cors import CORS

app = Flask(__name__) 
CORS(app)

model = pickle.load(open("./model/example_knn.pkl", "rb"))


@app.route("/")
def use_template():
    return render_template("index.html")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    input_1 = request.form["1"]
    input_2 = request.form["2"]
    input_3 = request.form["3"]
    input_4 = request.form["4"]
    input_5 = request.form["5"]
    input_6 = request.form["6"]
    input_7 = request.form["7"]
    input_8 = request.form["8"]

    df_setup = pd.DataFrame(
        [
            pd.Series(
                [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8]
            )
        ]
    )

    y_predict = model.predict_proba(df_setup)

    output = "{0:.{1}f}".format(y_predict[0][1], 2)
    output = str(float(output) * 100) + "%"

    if output > str(0.5):
        return render_template(
            "result.html",
            pred=f"You have the following chance of having diabetes: {output}",
        )
    else:
        return render_template(
            "result.html", pred=f"You have low chance of having diabetes: {output}"
        )


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=80)
    app.run(debug=True)
    # port = int(os.environ.get("PORT", 5000))
    # app.run(host='0.0.0.0', port=port)
