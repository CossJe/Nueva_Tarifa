# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 17:27:00 2026

@author: Jesus Coss
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json

class Elasticity():
    def __init__(self,df):
        # Obtener el directorio de trabajo actual (ruta principal del proyecto).
        self.ruta_principal = os.getcwd()

        # Construir la ruta al archivo 
        self.json_path = os.path.join(self.ruta_principal, "Files", "Elasticidades.json")
        
        self.df= df
        self.Get_Elas()
        
    def Get_Elas(self):

        # CORRECCIÓN DE TIPOS (Evitar el TypeError de Categorical)
        # Convertimos a string para poder concatenar y crear la llave de trayecto
        self.df['ORIGEN_DESTINO'] = self.df['ORIGEN'].astype(str) + '-' + self.df['DESTINO'].astype(str)
        
        # AGRUPACIÓN
        # Consolidamos ventas y boletos por trayecto y fecha
        df_agrupado = self.df.groupby(['ORIGEN_DESTINO', 'FECHA_CORRIDA', 'CV_CORRIDA']).agg({
            'INGRESO_TEORICO_TRAMO': 'mean',
            'INGRESO_TRANSP': 'mean',
            'CAPACIDAD_ASIENTOS_TRAMO': 'mean',
            'OCUPACION_TRAMO': 'mean'
        }).reset_index().copy()
        
        a= df_agrupado.groupby(['ORIGEN_DESTINO','FECHA_CORRIDA']).agg({
        'INGRESO_TEORICO_TRAMO': 'sum',
        'INGRESO_TRANSP': 'sum',
        'CAPACIDAD_ASIENTOS_TRAMO': 'sum',
        'OCUPACION_TRAMO': 'sum'}).reset_index().copy()
        
        # 3. PREPARACIÓN CRONOLÓGICA
        # Aseguramos que las fechas sean objetos datetime y ordenamos
        a['FECHA_CORRIDA'] = pd.to_datetime(a['FECHA_CORRIDA'])
        a = a.sort_values(by=['ORIGEN_DESTINO', 'FECHA_CORRIDA'])
        a['Day']=a['FECHA_CORRIDA'].dt.day_name()
        
        # CÁLCULO DE VARIABLES DERIVADAS
        # Calculamos el precio promedio por boleto para cada día
        a['%P']=(a['INGRESO_TEORICO_TRAMO']-a['INGRESO_TRANSP'])/a['INGRESO_TRANSP']
        a['%D']= (a['CAPACIDAD_ASIENTOS_TRAMO']-a['OCUPACION_TRAMO'])/a['OCUPACION_TRAMO']
        # CÁLCULO DE VARIACIONES PORCENTUALES (%Δ)
        # Usamos groupby para que el cambio porcentual no se mezcle entre trayectos distintos
        # Esto equivale a tu fórmula: (Siguiente - Anterior) / Anterior
        #a['PCT_CHANGE_DEMANDA'] = a.groupby('ORIGEN_DESTINO')['BOLETOS_VEND'].pct_change()
        #a['PCT_CHANGE_PRECIO'] = a.groupby('ORIGEN_DESTINO')['PRECIO_PROM'].pct_change()
        
        # CÁLCULO DE ELASTICIDAD PRECIO DE LA DEMANDA
        # E = % Cambio en Cantidad / % Cambio en Precio
        #a['ELASTICIDAD'] = a['PCT_CHANGE_DEMANDA'] / a['PCT_CHANGE_PRECIO']
        a['ELASTICIDAD'] = a['%D'] / a['%P']
        # 7. LIMPIEZA DE RESULTADOS
        # Reemplazamos infinitos (cuando el precio no cambió) por NaN
        a['ELASTICIDADES'] = a['ELASTICIDAD'].replace([np.inf, -np.inf], np.nan)
        
        self.Df=a
        