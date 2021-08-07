from imports import *
from flask import *
from flask_restful import *
import werkzeug
from flask_restful import reqparse
from gevent.pywsgi import WSGIServer
from gevent import monkey
hp = Help_Funcs()
app = Flask(__name__)
api = Api(app)
files = reqparse.RequestParser()
files.add_argument(
    "file",
    type=werkzeug.datastructures.FileStorage,
    location="files",
    help='Pass a img pls send a post request or get or put request. How to do is next. *|* files = {"file": open(f"./API/test_data/{path}", "rb")} \n result = requests.post("http://127.0.0.1:5000/test/", files=files)',
)


def get_intrest(idx, files):
    data = files.parse_args()
    file = data["file"]
    file.save(f"./out/{idx}.png")
    img = cv2.imread(f"./out/{idx}.png")
    img = cv2.resize(img, dsize=(84, 84))
    model = torch.load("./trained_models/model-0.pt")
    preds = hp.get_multiple_preds([f"./out/{idx}.png"], model)
    labels = {0: "no", 1: "yes"}
    try:
        print(labels[preds[0][0][0]])
    except:
        preds = {}
    return {"preds": preds, "labels": labels}


class Test(Resource):
    def __init__(self):
        super(Test, self).__init__()
        self.idx = -1

    def post(self):
        self.idx += 1
        results = get_intrest(idx=self.idx, files=files)
        return results

    def get(self):
        self.idx += 1
        results = get_intrest(idx=self.idx, files=files)
        return results

    def put(self):
        self.idx += 1
        results = get_intrest(idx=self.idx, files=files)
        return results


api.add_resource(Test, "/")


if __name__ == "__main__":
    app.run(debug=True, threaded=True, host="192.168.1.9", port=5000)

