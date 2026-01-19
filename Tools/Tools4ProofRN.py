# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 10:56:10 2026

@author: Jesus Coss
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import model_from_json
from datetime import timedelta

import statsmodels.formula.api as smf

class ProofRedNeuronal():
    def __init__(self,df):
        # Obtener el directorio de trabajo actual (ruta principal del proyecto).
        self.ruta_principal = os.getcwd()

        # Construir la ruta al archivo 
        self.json_path = os.path.join(self.ruta_principal, "Models", "modelo_arquitectura.json")
        self.json_Net = os.path.join(self.ruta_principal, "Files", "caracNet.json")
        self.weights_path = os.path.join(self.ruta_principal, "Models", "modelo_pesos.weights.h5")
        self.scarler_path = os.path.join(self.ruta_principal, "Models", 'ScalerNet.pkl')
        self.BuenasCarac_path = os.path.join(self.ruta_principal, "Files", "BuenaCaracteristicas.json")
        
        self.df= df
        self.Get_info()
        self.Data4Forecasting()
        self.NewClientsPredNet()
        

    
    def Get_info(self):
        with open(self.BuenasCarac_path, 'r') as f:
            # 2. Cargar el contenido del archivo JSON
            self.BC_json = json.load(f)
            
        with open(self.json_Net, 'r') as f:
            self.datos_carac = json.load(f)
            
        with open(self.json_path, 'r') as json_file:
            self.loaded_model_json = json_file.read()
        
            
    def Data4Forecasting(self):
        self.df['TIPO_CLASE'] = np.where(
        self.df['CLASE_SERVICIO'].astype(str).str.contains('DOS PISOS', case=False, na=False),
        'DOS',
        'UNO'
        )
        self.df["HORA_SALIDA_CORRIDA"] = pd.to_datetime(self.df["HORA_SALIDA_CORRIDA"])
        self.df['FECHA_CORRIDA'] = pd.to_datetime(self.df['FECHA_CORRIDA'])
        df_total = pd.DataFrame(columns=self.datos_carac["FrameN.columns"])
        
        df_total['Origen-Destino'] = self.df['ORIGEN'].astype(str) + '-' + self.df['DESTINO'].astype(str)
        df_total['DiaSemana_Corrida']=self.df['FECHA_CORRIDA'].dt.dayofweek
        df_total['Hora_Corrida']=self.df['HORA_SALIDA_CORRIDA'].dt.hour
        df_total[['NUM_ASIENTO','HORAS_ANTICIPACION']]=self.df[['NUM_ASIENTO','HORAS_ANTICIPACION']].copy()
        df_total['%_dif_TBT_Venta']=self.datos_carac['%_dif_TBT_Venta']
        df_total['Mes_Corrida']=self.df['FECHA_CORRIDA'].dt.month
        df_total['Anio_Corrida']=self.df['FECHA_CORRIDA'].dt.year
        df_total['Buen_Dia'] = self.df['FECHA_CORRIDA'].dt.dayofweek.isin(self.BC_json["DiaBueno"]).astype(int)
        df_total['Buena_Hora'] = self.df['HORA_SALIDA_CORRIDA'].dt.hour.isin(self.BC_json["HoraBuena"]).astype(int)
        df_total['Buen_Mes'] = self.df['FECHA_CORRIDA'].dt.month.isin(self.BC_json["MesBueno"]).astype(int)
        
        
        
        DictAsientosBuenos=self.BC_json["AsientosBuenos"]
        # 1. Convertir la columna 'CAPACIDAD_ASIENTOS_TRAMO' de df a string (si las llaves del diccionario son strings).
        # Esto se hace en df, que es donde se encuentra la columna de capacidad.
        self.df['CAPACIDAD_ASIENTOS_TRAMO'] = self.df['CAPACIDAD_ASIENTOS_TRAMO'].astype(str)
        
        # 2. Mapear la lista de asientos buenos a una nueva columna temporal en df_total
        # Usamos la columna de df para buscar las listas en DictAsientosBuenos.
        df_total['LISTA_ASIENTOS_BUENOS'] = self.df['CAPACIDAD_ASIENTOS_TRAMO'].map(DictAsientosBuenos)
        
        # 3. Aplicar la lógica para crear la columna 'Buen_Asiento' en df_total
        # Esto verifica si el 'NUM_ASIENTO' (que asumimos está en df_total) está en la lista mapeada.
        df_total['Buen_Asiento'] = df_total.apply(
            # Si el valor de la celda es una lista, realiza la comprobación. Si no, es 0.
            lambda row: int(row['NUM_ASIENTO'] in row['LISTA_ASIENTOS_BUENOS'])
            if isinstance(row['LISTA_ASIENTOS_BUENOS'], list) else 0,
            axis=1
        )
        # 4. Eliminar la columna temporal de df_total
        df_total = df_total.drop(columns=['LISTA_ASIENTOS_BUENOS'])
        
        
        # Crea un nuevo DataFrame con las variables dummy (codificación one-hot)
        df_dummies = pd.get_dummies(
            self.df['TIPO_CLIENTE'],
            prefix='TIPO_CLIENTE', # Prefijo para las nuevas columnas (ej: TIPO_CLIENTE_A)
            drop_first=False        # Elimina la primera categoría para evitar multicolinealidad
        ).astype(int)
    
        df_total[df_dummies.columns]= df_dummies[df_dummies.columns].copy()
        # Crea un nuevo DataFrame con las variables dummy (codificación one-hot)
        df_dummies1 = pd.get_dummies(
            self.df['TIPO_CLASE'],
            prefix='PISO', 
            drop_first=False
        ).astype(int)
    
        df_total[df_dummies1.columns]= df_dummies1[df_dummies1.columns].copy()
        # Une las nuevas columnas dummy al DataFrame original
        #df_total['TARIFA']= df['TARIFA_BASE_TRAMO']
        df_total['VENTA']=self.df['VENTA'].copy()
        df_total=df_total.fillna(0)
        
        self.df_T=df_total.copy()
    
    def GetPredictingForm(self):
        X1 = self.df_T.drop('VENTA', axis=1)
        
        X_final=pd.DataFrame(columns=self.datos_carac["X_processed.columns"])
        
        categorical_features= 'Origen-Destino'
        df_ohe = pd.get_dummies(X1[categorical_features]).astype(int)
        
        # Columnas numéricas que necesitan Estandarización
        # Excluimos las binarias/dummies que ya están bien escaladas (0 o 1)
        numeric_features = [
            'DiaSemana_Corrida', 'Hora_Corrida', 'NUM_ASIENTO', 
            'HORAS_ANTICIPACION', '%_dif_TBT_Venta', 'Mes_Corrida','Anio_Corrida'
        ]
        
        # Columnas binarias (se dejan pasar sin transformación)
        binary_features = [col for col in X1.columns if col not in [categorical_features] + numeric_features]
        
        indice_correcto = X1[numeric_features].index # o df_ohe.index
        
        
        scaler_cargado = joblib.load(self.scarler_path) # Carga el objeto guardado
    
        # 2. Convierte el array escalado (NumPy) a DataFrame, ASIGNANDO el índice correcto
        X_escalado_array = scaler_cargado.transform(X1[numeric_features])
        X_escalado = pd.DataFrame(X_escalado_array, 
                                  index=indice_correcto, # <-- ¡CLAVE!
                                  columns=numeric_features)
        
        X_processed1= pd.concat([df_ohe, X_escalado,X1[binary_features]], axis=1)
        
        X_final[X_processed1.columns]= X_processed1[X_processed1.columns].copy()

        self.X_final=X_final.fillna(0)
    

    def GetValues(self):
        # 'model' es tu red neuronal entrenada
        # 'X_test' son tus features de prueba (escalados y codificados)
        Y_pred = self.loaded_model.predict(self.X_final, verbose=0)
        
        Y_R_real = self.df_T['VENTA']
                    
        
        # Calcular el MAE real
        mae_real = mean_absolute_error(Y_R_real, Y_pred)
        
        #print(f"\nEl Error Absoluto Medio (MAE) final es de: {mae_real:,.2f} [Moneda]")
        return Y_pred
    
    def NewClientsPredNet(self):
        self.loaded_model = model_from_json(self.loaded_model_json)

        # Cargar los pesos entrenados desde HDF5
        self.loaded_model.load_weights(self.weights_path)
        
        # Compilar el modelo cargado (necesario antes de hacer predicciones)
        self.loaded_model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
        
        
        X_final= self.GetPredictingForm()
        self.PrecioDin=self.GetValues()
        
        return 