from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from Parcial2019 import Ejercicio


#db_connect = create_engine('sqlite:////chinook.db') #La ruta depende de donde tengas almacenada la base de datos
app = Flask(__name__)
api = Api(app)


class Calendar(Resource):
    def get(self, calendar):
        #conn = db_connect.connect()
        #query = conn.execute("select * from employees where EmployeeId =%d " % int(employee_id))
        #result = {'data': [dict(zip(tuple(query.keys()), i)) for i in query.cursor]}
        result = Ejercicio.resolucion(calendar)
        return jsonify(result)

api.add_resource(Calendar, '/Calendar/<calendar>')  # Route_3

if __name__ == '__main__':
     app.run(port='5000')