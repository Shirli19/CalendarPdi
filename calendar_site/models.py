from django.db import models

class Calendar(models.Model):
    #atributos
    """
    def get(self, calendar):
        #conn = db_connect.connect()
        #query = conn.execute("select * from employees where EmployeeId =%d " % int(employee_id))
        #result = {'data': [dict(zip(tuple(query.keys()), i)) for i in query.cursor]}
        result = Ejercicio.resolucion(calendar)
        return jsonify(result)
    """