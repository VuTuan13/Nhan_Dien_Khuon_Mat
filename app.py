from flask import Flask, render_template, request, redirect
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/add', methods=['POST'])
def add_person():
    name = request.form.get('name')
    subprocess.run(["python", "scripts/new_person.py"], input=name.encode())
    return redirect("/")

@app.route('/recognize', methods=['GET'])
def recognize():
    subprocess.run(["python", "scripts/recognize.py"])
    return redirect("/")

if __name__ == '__main__':
    app.run(debug=True)
