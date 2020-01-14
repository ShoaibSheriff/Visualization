from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)

from flask import jsonify
from processing import Data_Processer


csv_file = r"E:\CSE 564\Assign 2\flask\data\diet.csv"

dp = Data_Processer(sample_rate = 0.5, no_of_columns = 10)
dp.get_task_1(csv_file)
dp.get_task_2(csv_file)

@app.route('/')
def default():
    return jsonify({"error": "Blank path passed"})

@app.route('/task_1', methods=['GET'])
def task_1():
    global csv_file
    global dp

    data = dp.get_task_1(csv_file)
    return jsonify({"samples": data[0], "elbow": data[1], "elbow_index": data[2], "startified_samples": data[3]})

@app.route('/task_2', methods=['GET'])
def task_2():
    global csv_file
    global dp

    data = dp.get_task_2(csv_file)
    return jsonify({"dimensions": data[0], "scree_all": data[1], "scree_strat" : data[2], "cumu_all" : data[3], "cum_strat" : data[4], "top_attributes": data[5]})

@app.route('/task_3', methods=['GET'])
def task_3():
    global csv_file
    global dp

    data = dp.get_task_3(csv_file)
    return jsonify({"data_projections": data[0], "euclidean_2d": data[1], "correlation_2d": data[2], "scatter_matrix_top_3": data[3]})


@app.route('/task_3_0', methods=['GET'])
def task_3_0():
    global csv_file
    global dp

    data = dp.get_task_3_0(csv_file)
    return jsonify({"data_projections": data[0], "euclidean_2d": data[1], "correlation_2d": data[2], "scatter_matrix_top_3": data[3]})


if __name__ == '__main__':
    app.run()