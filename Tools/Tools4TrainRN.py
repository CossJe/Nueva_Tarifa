# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 11:31:35 2025

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

class RedNeuronal():
         
    def __init__(self,Df):
        # Obtener el directorio de trabajo actual (ruta principal del proyecto).
        self.ruta_principal = os.getcwd()

        # Construir la ruta al archivo 
        self.json_path = os.path.join(self.ruta_principal, "Models", "modelo_arquitectura.json")
        self.json_Net = os.path.join(self.ruta_principal, "Files", "caracNet.json")
        self.weights_path = os.path.join(self.ruta_principal, "Models", "modelo_pesos.weights.h5")
        
        self.BuenasCarac_path = os.path.join(self.ruta_principal, "Files", "BuenaCaracteristicas.json")
        
        self.Df= Df
        self.Get_info()
        self.Get_data4RN()
        self.GetTrainingForm()
        self.TrainingNet()
        self.FinistTraining()
    
    def Get_info(self):
        with open(self.BuenasCarac_path, 'r') as f:
            # 2. Cargar el contenido del archivo JSON
            self.BC_json = json.load(f)
            
    def Get_data4RN(self):
        df_total= pd.DataFrame()
        df_total['Origen-Destino'] = self.Df['ORIGEN'].astype(str) + '-' + self.Df['DESTINO'].astype(str)
        df_total['DiaSemana_Corrida']=self.Df['FECHA_CORRIDA'].dt.dayofweek
        df_total['Hora_Corrida']=self.Df['HORA_SALIDA_CORRIDA'].dt.hour
        df_total[['NUM_ASIENTO','HORAS_ANTICIPACION','%_dif_TBT_Venta']]=self.Df[['NUM_ASIENTO','HORAS_ANTICIPACION','%_dif_TBT_Venta']].copy()
        df_total['Mes_Corrida']=self.Df['FECHA_CORRIDA'].dt.month
        df_total['Anio_Corrida']=self.Df['FECHA_CORRIDA'].dt.year
        df_total['Buen_Dia'] = self.Df['FECHA_CORRIDA'].dt.dayofweek.isin(self.BC_json["DiaBueno"]).astype(int)
        df_total['Buena_Hora'] = self.Df['HORA_SALIDA_CORRIDA'].dt.hour.isin(self.BC_json["HoraBuena"]).astype(int)
        df_total['Buen_Mes'] = self.Df['FECHA_CORRIDA'].dt.month.isin(self.BC_json["MesBueno"]).astype(int)
        
        DictAsientosBuenos=self.BC_json["AsientosBuenos"]
        # 1. Convertir la columna 'CAPACIDAD_ASIENTOS_TRAMO' de df a string (si las llaves del diccionario son strings).
        # Esto se hace en df, que es donde se encuentra la columna de capacidad.
        self.Df['CAPACIDAD_ASIENTOS_TRAMO'] = self.Df['CAPACIDAD_ASIENTOS_TRAMO'].astype(str)
        
        # 2. Mapear la lista de asientos buenos a una nueva columna temporal en self.Df_total
        # Usamos la columna de df para buscar las listas en DictAsientosBuenos.
        df_total['LISTA_ASIENTOS_BUENOS'] = self.Df['CAPACIDAD_ASIENTOS_TRAMO'].map(DictAsientosBuenos)
        
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
            self.Df['TIPO_CLIENTE'],
            prefix='TIPO_CLIENTE', # Prefijo para las nuevas columnas (ej: TIPO_CLIENTE_A)
            drop_first=False        # Elimina la primera categoría para evitar multicolinealidad
        ).astype(int)
        
        # Crea un nuevo DataFrame con las variables dummy (codificación one-hot)
        df_dummies1 = pd.get_dummies(
            self.Df['TIPO_CLASE'],
            prefix='PISO', 
            drop_first=False
        ).astype(int)
        
        # Une las nuevas columnas dummy al DataFrame original
        df_total = pd.concat([df_total, df_dummies,df_dummies1], axis=1)
        #df_total['TARIFA']=df['TARIFA_BASE_TRAMO'].copy()
        df_total['VENTA']=self.Df['VENTA'].copy()
        
        self.dff=df_total.copy()
        
    def GetTrainingForm(self):
        # Definir la variable objetivo (Y)
        self.Y = self.dff['VENTA']
        
        # Eliminar la variable VENTA del dataframe de features (X)
        X = self.dff.drop('VENTA', axis=1) 
        
        categorical_features= 'Origen-Destino'
        df_ohe = pd.get_dummies(X[categorical_features]).astype(int)
        
        # Columnas numéricas que necesitan Estandarización
        # Excluimos las binarias/dummies que ya están bien escaladas (0 o 1)
        numeric_features = [
            'DiaSemana_Corrida', 'Hora_Corrida', 'NUM_ASIENTO', 
            'HORAS_ANTICIPACION', '%_dif_TBT_Venta', 'Mes_Corrida','Anio_Corrida',
        ]
        
        # Columnas binarias (se dejan pasar sin transformación)
        binary_features = [col for col in X.columns if col not in [categorical_features] + numeric_features]
        
        indice_correcto = X[numeric_features].index # o df_ohe.index
        
        scaler = StandardScaler()
    
        
        # 2. Convierte el array escalado (NumPy) a DataFrame, ASIGNANDO el índice correcto
        X_escalado_array = scaler.fit_transform(X[numeric_features])
        X_escalado = pd.DataFrame(X_escalado_array, 
                                  index=indice_correcto, # <-- ¡CLAVE!
                                  columns=numeric_features)
        
        config_path = os.path.join(self.ruta_principal, "Models", 'ScalerNet.pkl')
        joblib.dump(scaler,config_path )
        
        self.X_processed= pd.concat([df_ohe, X_escalado,X[binary_features]], axis=1)
        
    def TrainingNet(self):
        # División del 80% para entrenamiento y 20% para prueba
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X_processed, 
            self.Y, 
            test_size=0.2, 
            random_state=42 # Para asegurar resultados reproducibles
        )
        
        input_feature_count = X_train.shape[1] 
        
        
        # --- 1. Definir el número de features de entrada ---
        # (Esto debe ser el número de columnas de tu X_train después del OHE y estandarización)
        input_shape = X_train.shape[1] 
        
        # --- 2. CONSTRUCCIÓN DEL MODELO ---
        model = Sequential([
            # Capa Oculta 1
            Dense(128, activation='relu', input_shape=(input_shape,)),
            
            # Capa Oculta 2 (Regularización para evitar overfitting)
            # Aquí puedes añadir 'Dropout' si notas que el modelo se sobreajusta
            Dense(64, activation='relu'), 
            
            # Capa de Salida: 1 neurona y activación lineal para regresión
            Dense(1, activation='linear') 
        ])
        
        # --- 3. COMPILACIÓN DEL MODELO ---
        model.compile(
            optimizer='adam',
            loss='mse',           # Función de pérdida: Error Cuadrático Medio
            metrics=['mae', 'mse']  # Métricas a monitorear: MAE y MSE
        )
        
        # --- 4. ENTRENAMIENTO (Ejemplo) ---
        history = model.fit(
            X_train, 
            Y_train,  # ¡Usamos la variable VENTA transformada con logaritmo!
            epochs=1, 
            batch_size=32, 
            validation_split=0.2, # Usamos el 20% para validación interna
            verbose=0
        )
        
        Y_pred = model.predict(X_test, verbose=0)
        
        # Calcular el MAE real
        mae_real = mean_absolute_error(Y_test, Y_pred)
        
        print(f"\nEl Error Absoluto Medio (MAE) final es de: {mae_real:,.2f} [Moneda]")
        
        #  Calcular el Error Cuadrático Medio (MSE)
        mse_real = mean_squared_error(Y_test, Y_pred)
        
        # Calcular la Raíz del Error Cuadrático Medio (RMSE)
        #    La RMSE es simplemente la raíz cuadrada del MSE
        rmse_real = np.sqrt(mse_real)
        
        print(f"\nLa Raíz del Error Cuadrático Medio (RMSE) final es de: {rmse_real:,.2f} [Moneda]")
        
        self.model= model
        return 
    
    def FinistTraining(self):
        model_json = self.model.to_json()
        with open(self.json_path, "w") as json_file:
            json_file.write(model_json)
        
        self.model.save_weights(self.weights_path)
        
        carac = {
            "X_processed.columns": list(self.X_processed.columns),
            "%_dif_TBT_Venta": float(self.dff['%_dif_TBT_Venta'].mean()),
            "FrameN.columns": list(self.dff.columns)
        }
        
        
        with open(self.json_Net, "w", encoding="utf-8") as f:
            json.dump(carac, f, ensure_ascii=False, indent=4)