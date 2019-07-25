from flask import Flask, jsonify
from flask_restful import Resource, Api
from match_konspect import MatchKonspekt


app = Flask(__name__)
api = Api(app)


class Konspekt(Resource):
    def get(self, mdt):
        mk = MatchKonspekt()
        category, subcategory, description = mk.find_category(mdt)
        result = {"category": category, "subcategory": subcategory, "description": description}
        return jsonify(result)


api.add_resource(Konspekt, '/konspekt/<path:mdt>')

if __name__ == '__main__':
    app.run()
