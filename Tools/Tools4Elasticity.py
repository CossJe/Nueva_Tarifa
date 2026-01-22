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
import gc  # Garbage Collector para liberar memoria

class Elasticity():
    def __init__(self, df):
        self.ruta_principal = os.getcwd()
        self.json_path = os.path.join(self.ruta_principal, "Files", "Elasticidades.json")
        
        # Trabajamos sobre una referencia para no duplicar el DF original de entrada
        self.df = df
        self.Get_Elas()
        
    def Get_Elas(self):
        # 1. OPTIMIZACIÓN DE TIPOS Y MEMORIA
        # Convertimos a string y luego a categoría para ahorrar RAM en el agrupamiento
        self.df['ORIGEN_DESTINO'] = (self.df['ORIGEN'].astype(str) + '-' + 
                                     self.df['DESTINO'].astype(str)).astype('category')
        
        # 2. PRIMERA AGRUPACIÓN (Optimizada con as_index=False y sin .copy())
        df_agrupado = self.df.groupby(['ORIGEN_DESTINO', 'FECHA_CORRIDA', 'CV_CORRIDA'], 
                                      as_index=False, observed=True).agg({
            'INGRESO_TEORICO_TRAMO': 'mean',
            'INGRESO_TRANSP': 'mean',
            'CAPACIDAD_ASIENTOS_TRAMO': 'mean',
            'OCUPACION_TRAMO': 'mean'
        })
        
        df_agrupado['INGRESO_TEORICO']= (df_agrupado['INGRESO_TEORICO_TRAMO']/df_agrupado['OCUPACION_TRAMO'])*df_agrupado['CAPACIDAD_ASIENTOS_TRAMO']              
        # Liberamos memoria del DataFrame original si ya no se usará
        # self.df = None 
        # gc.collect()

        # 3. SEGUNDA AGRUPACIÓN
        # Eliminamos .copy() ya que agg() ya genera un objeto nuevo
        a = df_agrupado.groupby(['ORIGEN_DESTINO','FECHA_CORRIDA'], 
                                as_index=False, observed=True).agg({
            'INGRESO_TEORICO': 'sum',
            'INGRESO_TRANSP': 'sum',
            'CAPACIDAD_ASIENTOS_TRAMO': 'sum',
            'OCUPACION_TRAMO': 'sum'
        })
        
        # Liberar el dataframe intermedio
        del df_agrupado
        gc.collect()
        
        # 4. PREPARACIÓN CRONOLÓGICA
        a['FECHA_CORRIDA'] = pd.to_datetime(a['FECHA_CORRIDA'])
        # inplace=True evita crear una copia adicional en memoria al ordenar
        a.sort_values(by=['ORIGEN_DESTINO', 'FECHA_CORRIDA'], inplace=True)
        a['Day'] = a['FECHA_CORRIDA'].dt.day_name().astype('category') # Categoría es más ligera que string
        
        # 5. CÁLCULO DE VARIABLES DERIVADAS (Lógica original)
        # Realizamos las operaciones directamente (vectorizadas)
        a['%P'] = ( a['INGRESO_TRANSP']-a['INGRESO_TEORICO']) / a['INGRESO_TEORICO']
        a['%D'] = ( a['OCUPACION_TRAMO']-a['CAPACIDAD_ASIENTOS_TRAMO']) / a['CAPACIDAD_ASIENTOS_TRAMO']
        
        # 6. ELASTICIDAD
        a['ELASTICIDAD'] = a['%D'] / a['%P']
        
        # 7. LIMPIEZA DE RESULTADOS
        # Usamos inplace para no generar otro DF
        a['ELASTICIDAD'].replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Renombramos o asignamos según tu lógica
        a.rename(columns={'ELASTICIDAD': 'ELASTICIDADES'}, inplace=True)
        
        self.Df = a