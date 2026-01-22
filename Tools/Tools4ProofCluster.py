# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 11:39:35 2026

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
from xgboost import XGBClassifier 

class PredictorClusters:
    def __init__(self,Frame):
        self.ruta_principal = os.getcwd()
        self.NomArch = "modelo_xgboost_clientes.json"
        self.config_path = os.path.join(self.ruta_principal, "Models", self.NomArch)
        self.columnas = [
    'FECHA_OPERACION', 'HORA_OPERACION', 'NOMBRE_DIA_OPERACION', 'AREA_VENTA',
    'CLASE_SERVICIO', 'ORIGEN', 'DESTINO', 'CV_CORRIDA', 'FECHA_CORRIDA',
    'HORA_SALIDA_CORRIDA', 'TIPO_PASAJERO', 'TIPO_CLIENTE', 'OCUPACION_TRAMO',
    'TARJETA', 'EFECTIVO', 'TIPO_OPERACION', 'VENTA_ANTICIPADA', 'NOMBRE_PASAJERO',
    'PORCENT_PROMO', 'DESC_DESCUENTO', 'BOLETOS_CANCEL', 'BOLETOS_TOTALES',
    'BOLETOS_VEND', 'VENTA_TOTAL', 'VENTA', 'ORIGEN_CORRIDA', 'DESTINO_CORRIDA',
    'CAPACIDAD_ASIENTOS_TRAMO', 'KMS_TRAMO', 'EMPRESA', 'TIPO_CORRIDA',
    'TIPO_BUS', 'HORA_SALIDA_ORIGEN_CORRIDA', 'NOMBRE_DIA_CORRIDA', 'TARIFA_BASE_TRAMO',
    'EMAIL', 'PAGO_METODO', 'HORA_SALIDA_CORRIDA_', 'TOTAL_BOLETOS',
    'TOTAL_VENTA', 'DIAS_ANTICIPACION', 'HORAS_ANTICIPACION','ORIGEN_DESTINO']
        
        self.csv_path = os.path.join(self.ruta_principal, "Files", "DB_ClientesAgrupados.csv")
        self.DF_K_means = pd.read_csv(self.csv_path)
        self.Frame=Frame
        self.Cargar_Modelo()
        self.Preparar_datos()
        self.Predecir()
        self.Get_values()
        
    def Cargar_Modelo(self):
        self.modelo_cargado = XGBClassifier()
        self.modelo_cargado.load_model(self.config_path)
        
        
    def Preparar_datos(self):
        columnas_existentes = [col for col in self.columnas if col in self.Frame.columns]
        df_original = self.Frame[columnas_existentes].copy()
        # Merge para datos que ya estan en la base K_means
        df_In_Kmeans = pd.merge(
            df_original,
            self.DF_K_means[['EMAIL', 'Cluster']],
            on='EMAIL',
            how='inner'
        )
        
        # 1. Unimos usando 'left' para mantener todo lo de la izquierda (df_original)
        df_solo_original = pd.merge(
            df_original,
            self.DF_K_means[['EMAIL']], # Solo necesitamos el EMAIL para comparar
            on='EMAIL',
            how='left',
            indicator=True
        )
        
        # 2. Filtramos los que NO encontraron pareja en la tabla de clusters
        # (Los que tienen 'left_only' en la columna indicadora)
        df_NotIn_Kmeans = df_solo_original[df_solo_original['_merge'] == 'left_only']
        
        # 3. Opcional: Eliminamos la columna '_merge' para que quede limpio
        df_NotIn_Kmeans = df_NotIn_Kmeans.drop(columns=['_merge'])
        
        columnas_util = ['EMAIL', 'PORCENT_PROMO', 'DESC_DESCUENTO', 'VENTA',
                        'DIAS_ANTICIPACION', 'EMPRESA', 'PAGO_METODO', 'ORIGEN_DESTINO']
        
        self.df_nuevos = df_NotIn_Kmeans[columnas_util]
        self.df_In_Kmeans= df_In_Kmeans.copy()
        self.df_NotIn_Kmeans= df_NotIn_Kmeans.copy()
        
    def Predecir(self): 
        
        datos = self.df_nuevos.copy()
        print(f"Datos recibidos: {datos.shape[0]} filas, {datos.shape[1]} columnas")
        
        # One-hot encoding
        datos = pd.get_dummies(
            datos,
            columns=['DESC_DESCUENTO', 'EMPRESA', 'PAGO_METODO', 'ORIGEN_DESTINO'],
            prefix=['DESC', 'EMP', 'PAGO', 'RUTA'],
            dtype='int8'
        )
        
        # Guardar EMAILs
        emails = datos['EMAIL'].copy()
        print(f"Clientes a predecir: {emails.nunique()} emails únicos")
        
        # Preparar datos para modelo
        datos_para_modelo = datos.drop('EMAIL', axis=1)
        print(f"Columnas para el modelo: {datos_para_modelo.shape[1]} features")
        
        # Alinear columnas con el modelo
        if hasattr(self.modelo_cargado, 'feature_names_in_'):
            columnas_esperadas = self.modelo_cargado.feature_names_in_
            print(f"Modelo espera {len(columnas_esperadas)} características")
            datos_para_modelo = datos_para_modelo.reindex(
                columns=columnas_esperadas,
                fill_value=0
            )
            print("Columnas alineadas con el modelo entrenado")
        
        # Realizar predicciones
        predicciones = self.modelo_cargado.predict(datos_para_modelo)
        print(f"Predicciones de cluster realizadas")
        
        # Construir resultados
        resultados = pd.DataFrame({'EMAIL': emails, 'Cluster': predicciones})
        
        # Agregar columnas originales
        for col in datos.columns:
            if col != 'EMAIL':
                resultados[col] = datos[col].values

        self.df_NotIn_Kmeans['Cluster']=resultados['Cluster'].copy()

    
    def Get_values(self):
        self.df_final = pd.concat([self.df_In_Kmeans, self.df_NotIn_Kmeans], axis=0).reset_index(drop=True)
        # 1. Define las equivalencias
        mapeo_clusters = {
            0: 5,
            2: 7,
            3: 13,
            1: 3,
            4: 0
        }
        
        # 2. Aplica el mapeo a toda la columna
        # Esto creará una nueva columna (o sobreescribirá una) con los valores que buscas
        self.df_final['Aumento'] = self.df_final['Cluster'].map(mapeo_clusters)
        return self.df_final
        
        