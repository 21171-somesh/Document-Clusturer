from flask import Flask, request, render_template, url_for, redirect
from flask_restful import Api, reqparse, Resource
from predict import pred

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('query')

@app.route('/', methods = ['GET'])
def main():
	return render_template("main.html")

@app.route('/about', methods = ['POST'])
def about_post():
	s = request.form['title']
	return redirect(url_for('results', data = s), code = 302)

@app.route('/results', methods = ['GET'])
def results():
	s = request.args['data']
	ans = pred(s)
	return render_template("result.html", data = ans)

class ApiFunction(Resource):
	def post(self):
		s = parser.parse_args()['query']
		ans = pred(s)
		print(ans, s)
		return 'Prediction: ' + str(ans)
	def get(self):
		s = parser.parse_args()['query']
		ans = pred(s)
		print(ans, s)
		return 'Prediction: ' + str(ans)

api.add_resource(ApiFunction, '/api')

if __name__ == "__main__":
	app.run(debug=True)