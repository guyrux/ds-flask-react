from flask import Flask

app = Flask(__name__)


@app.route("/")
def greeting():
    return "Flask é incrível"
