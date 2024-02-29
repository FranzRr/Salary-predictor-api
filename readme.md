## Salary Predictor API

API creada en Python, que predice el salario de una persona, basado en:
- Año en el que se quiere predecir
- Nombre del trabajo
- Experiencia de trabajo
- Modalidad (presencial, híbrido, remoto)

Stack utilizado: Flask, onnxruntime, pandas, sickit_learn, numpy.

Esta API recibe un modelo pre-entrenado creado desde cero con TensorFlow y Keras, una red neuronal que resuelve el problema de regresión de predecir el salario.

## Ejecución en local
Es recomendable utilizar un Virtual Environment. Luego ejecutar:
``` bash
git clone https://github.com/FranzRr/Salary-predictor-api.git
cd Salary-predictor-api
pip install -r requirements.txt
py index.py
```

## Ejecución para producción
Se utilizará gunicorn (green unicorn) como puerta de enlace del servidor web Python, en este caso la puerta de enlace es 0.0.0.0 en el puerto 10000 utilizando el archivo index con una instancia de Flask llamada app.
``` bash
git clone https://github.com/FranzRr/Salary-predictor-api.git
cd Salary-predictor-api
pip install -r requirements.txt
pip install gunicorn
gunicorn 0.0.0.0:10000 index:app
```

