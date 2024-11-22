import streamlit as st # type: ignore
import pandas as pd # type: ignore
import joblib
import numpy as np


# Título de la aplicación
st.title("Predicción de precios de viviendas")
st.header("Ambito: Barcelona")
st.write("Ingrese las características de las viviendas")

# Front 
# 1-Metros cuadrados: float
m2 = float(st.text_input("Metros Cuadrados:", value="100"))

# 2-distrito: list
df_distritos = pd.read_csv('distritos.csv')
opciones_distrito = df_distritos['distritos'].tolist()
seleccion_distrito = st.selectbox('Distrito:', opciones_distrito)
indice_distrito = opciones_distrito.index(seleccion_distrito)
distrito = df_distritos.loc[indice_distrito, 'precio']

# 3-barrio: list
df_disbar = pd.read_csv('distrito_barrio.csv')
df_barrios_dis = df_disbar[df_disbar['distrito'] == seleccion_distrito]
df_barrios_dis = df_barrios_dis.reset_index(drop=True)
#st.write(df_barrios_dis)

#df_barrios = pd.read_csv('barrios.csv')
opciones_barrio = df_barrios_dis['barrio'].tolist()
seleccion_barrio = st.selectbox('Barrio:', opciones_barrio)
indice_barrio = opciones_barrio.index(seleccion_barrio)
barrio = df_barrios_dis.loc[indice_barrio, 'precio']
#st.write(barrio)

# 4-Tipo de vivienda: list
df_vivienda = pd.read_csv('tipo_vivienda.csv')
opciones_vivienda = df_vivienda['tipo'].tolist()
seleccion_vivienda = st.selectbox('Tipo de vivienda:', opciones_vivienda)
indice_vivienda = opciones_vivienda.index(seleccion_vivienda)
tipo_vivienda = df_vivienda.loc[indice_vivienda, 'valor']

# 5-numero de habitaciones: int
num_habitaciones = int(st.slider('Número de habitaciones :',0 , 8, 1))

# 6-numero de banos: int
num_banos = int(st.slider('Número de baños :',1 , 8, 1))

# 7-planta: list
df_planta = pd.read_csv('planta.csv')
opciones_planta = df_planta['planta'].tolist()
seleccion_planta = st.selectbox('Planta:', opciones_planta)
indice_planta = opciones_planta.index(seleccion_planta)
planta = df_planta.loc[indice_planta, 'valor']

# 8-terraza: bol
options_terraza = ["No", "Si"]
terraza_select = st.selectbox("Tiene terraza:", options_terraza)
terraza_index = options_terraza.index(terraza_select)
terraza = int(terraza_index)

# 9-balcon: bol
options_balcon2 = ["No", "Si"]
balcon_select = st.selectbox("Tiene balcón:", options_balcon2)
balcon_index = options_balcon2.index(balcon_select)
balcon = int(balcon_index)

# 10-ascensor: bol
options_ascensor = ["No", "Si"]
ascensor_select = st.selectbox("Tiene ascensor:", options_ascensor)
ascensor_index = options_ascensor.index(ascensor_select)
ascensor = int(ascensor_index)

# 11-estado_inmmueble: list
df_estado = pd.read_csv('estado_inmueble.csv')
opciones_estado = df_estado['estado'].tolist()
seleccion_estado = st.selectbox('Estado del inmueble:', opciones_estado)
indice_estado = opciones_estado.index(seleccion_estado)
estado = df_estado.loc[indice_estado, 'valor']

## Mostrar la opción seleccionada
#st.write(f'Has seleccionado tipo: {seleccion_tipo}')

# Creamos el array de entrada
X_list =    [m2,
             float(distrito),
             float(barrio),
             int(barrio),
             num_habitaciones,
             num_banos,
             int(planta),
             terraza,
             balcon,
             ascensor,
             int(estado)
              ]

#X = np.array([float(elemento) for elemento in X_list])
X = np.array(X_list, dtype=np.float64)
X = X.reshape(1,-1)

# Botón para ejecutar el modelo
if st.button("Predecir"):
    if len(X) > 0:
        
        # Cargar el modelo y los parámetros de normalización guardados
        #scaler = joblib.load('scaler.pkl')
        model = joblib.load('modelo_random_forest_joblib.pkl')
        
        # Mostrar las primeras filas del DataFrame cargado
        #st.write("Datos cargados:")
        #st.write(X)
        
        #data_scaled = scaler.transform(X)
        
        # Realizar predicciones con el modelo XGBoost
        predicciones = model.predict(X)
        
        # Mostrar las predicciones
        st.write("Predicciones de precio (Euros):")
        #st.write(predicciones)
        corrector = 0.10
        predicciones_bottom = predicciones * (1 - 2*corrector)
        precio_medio = predicciones * (1 - corrector)
        predicciones_top = predicciones
        df = pd.DataFrame({'Precio mínimo':np.round(predicciones_bottom,2),
                           'Precio esperado':np.round(precio_medio,2),
                           'Precio máximo':np.round(predicciones_top,2)})
        st.write(df)
        