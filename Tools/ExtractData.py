# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 11:33:34 2025

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

from src.dynamic_pricing_data_loader import cargar_y_preparar_datos


class GetDataFromETL():
    def __init__(self,train):
        # Obtener el directorio de trabajo actual (ruta principal del proyecto).
        self.ruta_principal = os.getcwd()
        self.train=train
        # Construir la ruta al archivo de configuración "config/config.json".
        self.config_path = os.path.join(self.ruta_principal, "config", "config.json")
        if self.train:
            self.Rutas_path = os.path.join(self.ruta_principal, "Files", "Rutas_continuasETN.csv")
        else:
            self.Rutas_path = os.path.join(self.ruta_principal, "Files", "Rutas_continuasETN_Muestra.csv")
        self.ArreglarConfig()
        self.Frame=cargar_y_preparar_datos(self.config_path, self.ruta_principal)
        self.ArreglarData()
        self.days=101
    
    def ArreglarConfig(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.Rutas_continuasETN = pd.read_csv(self.Rutas_path)
        # 2. Actualizar la sección "Ruta" con los datos del DataFrame
        # Usamos tolist() para que el formato sea compatible con JSON
        config["Ruta"]["ORIG"] = self.Rutas_continuasETN["ORIGEN"].tolist()
        config["Ruta"]["DEST"] = self.Rutas_continuasETN["DESTINO"].tolist()
        
        # 3. Volver a guardar el archivo (sobreescribir)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        print("El archivo JSON ha sido actualizado con las nuevas rutas.")
        
    def ArreglarData(self):
        self.Frame['ORIGEN_DESTINO'] = self.Frame['ORIGEN'].astype(str) + '-' + self.Frame['DESTINO'].astype(str)
        self.Rutas_continuasETN['ORIGEN_DESTINO'] = self.Rutas_continuasETN['ORIGEN'].astype(str) + '-' + self.Rutas_continuasETN['DESTINO'].astype(str)
        OriDes=list(self.Rutas_continuasETN['ORIGEN_DESTINO'])
        self.Frame = self.Frame[self.Frame['ORIGEN_DESTINO'].isin(OriDes)].reset_index(drop=True)
        
        
    def D4_Train_NN(self):
        df= self.Frame[['NOMBRE_PASAJERO','BOLETOS_VEND',"FECHA_CORRIDA", "HORA_SALIDA_CORRIDA", "CLASE_SERVICIO", 'IVA_TARIFA_BASE_TRAMO',
    "PAX_SUBEN", "TARIFA_BASE_TRAMO",'FECHA_OPERACION', 'HORA_OPERACION','VENTA','DISPONIBILIDAD_TRAMO',
    'HORAS_ANTICIPACION','ORIGEN', 'DESTINO','TIPO_CLIENTE','NUM_ASIENTO','CAPACIDAD_ASIENTOS_TRAMO','TIPO_PASAJERO'
                  ]].copy()
        
        df = df[df['VENTA'] > 0]
        df=df[df['BOLETOS_VEND']>0]
        df=df.drop('BOLETOS_VEND',axis=1)
        df= df[df['TIPO_PASAJERO'] == 'AD']
        
        df['FECHA_OPERACION'] = pd.to_datetime(df['FECHA_OPERACION'])
        dia_anterior = df['FECHA_OPERACION'].max() - timedelta(days=self.days)
        df = df[df['FECHA_OPERACION'] <= dia_anterior].copy()  
        df['FECHA_CORRIDA'] = pd.to_datetime(df['FECHA_CORRIDA'])

        df["HORA_SALIDA_CORRIDA"] = pd.to_datetime(df["HORA_SALIDA_CORRIDA"])
        
        df['TBT']= df['TARIFA_BASE_TRAMO']#-df['IVA_TARIFA_BASE_TRAMO']
        df['%_dif_TBT_Venta']= (df['TBT']-df['VENTA'])/df['TBT']
        df['TIPO_CLASE'] = np.where(
        df['CLASE_SERVICIO'].astype(str).str.contains('DOS PISOS', case=False, na=False),
            'DOS','UNO')
        
        return df 
    
    def D4_Proof_NN(self):
        df= self.Frame[['NOMBRE_PASAJERO','BOLETOS_VEND',"FECHA_CORRIDA", "HORA_SALIDA_CORRIDA", "CLASE_SERVICIO", 'IVA_TARIFA_BASE_TRAMO',
    "PAX_SUBEN", "TARIFA_BASE_TRAMO",'FECHA_OPERACION', 'HORA_OPERACION','VENTA','DISPONIBILIDAD_TRAMO',
    'HORAS_ANTICIPACION','ORIGEN', 'DESTINO','TIPO_CLIENTE','NUM_ASIENTO','CAPACIDAD_ASIENTOS_TRAMO','TIPO_PASAJERO'
                  ]].copy()
        
        df = df[df['VENTA'] > 0]
        df=df[df['BOLETOS_VEND']>0]
        df=df.drop('BOLETOS_VEND',axis=1)
        #df= df[df['TIPO_PASAJERO'] == 'AD']
        
        df['FECHA_OPERACION'] = pd.to_datetime(df['FECHA_OPERACION'])
        dia_anterior = df['FECHA_OPERACION'].max()
        fecha_inicio = df['FECHA_OPERACION'].max() - timedelta(days=self.days-1)
        
        df = df[
        (df['FECHA_OPERACION'] >= fecha_inicio) & 
        (df['FECHA_OPERACION'] <= dia_anterior)
        ].copy()  
        
        df['FECHA_CORRIDA'] = pd.to_datetime(df['FECHA_CORRIDA'])

        df["HORA_SALIDA_CORRIDA"] = pd.to_datetime(df["HORA_SALIDA_CORRIDA"])
        
        df['TBT']= df['TARIFA_BASE_TRAMO']#-df['IVA_TARIFA_BASE_TRAMO']
        df['%_dif_TBT_Venta']= (df['TBT']-df['VENTA'])/df['TBT']
        df['TIPO_CLASE'] = np.where(
        df['CLASE_SERVICIO'].astype(str).str.contains('DOS PISOS', case=False, na=False),
            'DOS','UNO')
        
        return df 
    
    def D4_ClusterNS(self):
        Df= self.Frame.copy()
        # Filtrar registros con VENTA_TOTAL > 0 (elimina ventas nulas o negativas).
        Df=Df[Df[ 'VENTA_TOTAL']>0]
        Df=Df[Df[ 'BOLETOS_VEND']>0]
        Df['HORAS_ANTICIPACION']=Df['HORAS_ANTICIPACION'].abs()
        Df['DIAS_ANTICIPACION']=Df['DIAS_ANTICIPACION'].abs()
        Df['FECHA_OPERACION'] = pd.to_datetime(Df['FECHA_OPERACION'])
        
        dia_anterior = Df['FECHA_OPERACION'].max() - timedelta(days=self.days)
        Df = Df[Df['FECHA_OPERACION'] <= dia_anterior].copy()  
        # Rellenar valores faltantes en PORCENT_PROMO con 0.
        Df['PORCENT_PROMO'] = Df['PORCENT_PROMO'].fillna(0)
        
        # Construir un DataFrame 'known' con nombres y emails conocidos:
        # - eliminar filas sin EMAIL,
        # - eliminar duplicados por NOMBRE_PASAJERO,
        # - quedarnos solo con ['NOMBRE_PASAJERO', 'EMAIL'].
        known = (
            Df
            .dropna(subset=['EMAIL'])
            .drop_duplicates(subset=['NOMBRE_PASAJERO'])
            [['NOMBRE_PASAJERO','EMAIL']]
        )
        
        # Hacer un merge para anotar cada fila con el EMAIL conocido (si existe).
        # Se crea temporalmente la columna 'EMAIL_KNOWN' para contener el email mapeado.
        Df = Df.merge(
            known,
            on='NOMBRE_PASAJERO',
            how='left',
            suffixes=('','_KNOWN')
        )
        
        # Rellenar la columna EMAIL con EMAIL_KNOWN cuando EMAIL original esté vacío.
        # Esto prioriza el email original; si es NaN, toma el conocido.
        Df['EMAIL'] = Df['EMAIL'].fillna(Df['EMAIL_KNOWN'])
        
        # Eliminar la columna auxiliar EMAIL_KNOWN que ya no se necesita.
        Df.drop(columns=['EMAIL_KNOWN'], inplace=True)
        
        # Dominio que se usará para generar emails genéricos (placeholder).
        dominio = 'ejemplo.com'
        
        # Obtener la lista de nombres únicos sin email asignado.
        nombres_sin_email = Df.loc[Df['EMAIL'].isna(), 'NOMBRE_PASAJERO'].unique()
        
        # Crear un diccionario mapping nombre -> email genérico (ej: "juan.perez@ejemplo.com").
        # Nota: se hace una transformación simple (lower + reemplazar espacios por puntos).
        generic_map = {
            nombre: f"{nombre.lower().replace(' ','.')}@{dominio}"
            for nombre in nombres_sin_email
        }
        
        # Asegurar que la columna EMAIL es de tipo object (cadena).
        Df['EMAIL'] = Df['EMAIL'].astype('object')
        
        # Rellenar los emails faltantes usando el mapping generado.
        # Se usa map sobre NOMBRE_PASAJERO y fillna para no sobrescribir emails existentes.
        Df['EMAIL'] = Df['EMAIL'].fillna(Df['NOMBRE_PASAJERO'].map(generic_map))
        
        # Corrección de categorías:
        # Si un adulto tiene promoción > 0, renombrar su descuento como "PROMOCION ESPECIAL"
        Df.loc[(Df['PORCENT_PROMO'] > 0) & (Df['DESC_DESCUENTO'] == 'ADULTO'), 'DESC_DESCUENTO'] = 'PROMOCION ESPECIAL'
        
        # Si tiene 0% de promoción pero estaba marcado como "PROMOCION ESPECIAL", devolverlo a "ADULTO"
        Df.loc[(Df['PORCENT_PROMO'] == 0) & (Df['DESC_DESCUENTO'] == 'PROMOCION ESPECIAL'), 'DESC_DESCUENTO'] = 'ADULTO'
        
        # Hacer nueva columna
        Df['ORIGEN_DESTINO'] = Df['ORIGEN'].astype(str) + '-' + Df['DESTINO'].astype(str)
        
        # Devolver el DataFrame final procesado.
        return Df
    
    def D4_ClusteringSuper(self):
        Df= self.Frame.copy()
        # Filtrar registros con VENTA_TOTAL > 0 (elimina ventas nulas o negativas).
        Df=Df[Df[ 'VENTA_TOTAL']>0]
        Df=Df[Df[ 'BOLETOS_VEND']>0]
        Df['HORAS_ANTICIPACION']=Df['HORAS_ANTICIPACION'].abs()
        Df['DIAS_ANTICIPACION']=Df['DIAS_ANTICIPACION'].abs()
        Df['FECHA_OPERACION'] = pd.to_datetime(Df['FECHA_OPERACION'])
        
        dia_anterior = Df['FECHA_OPERACION'].max()
        fecha_inicio = Df['FECHA_OPERACION'].max() - timedelta(days=self.days-1)
        
        Df = Df[
        (Df['FECHA_OPERACION'] >= fecha_inicio) & 
        (Df['FECHA_OPERACION'] <= dia_anterior)
        ].copy()  
        

        # Rellenar valores faltantes en PORCENT_PROMO con 0.
        Df['PORCENT_PROMO'] = Df['PORCENT_PROMO'].fillna(0)
        
        # Construir un DataFrame 'known' con nombres y emails conocidos:
        # - eliminar filas sin EMAIL,
        # - eliminar duplicados por NOMBRE_PASAJERO,
        # - quedarnos solo con ['NOMBRE_PASAJERO', 'EMAIL'].
        known = (
            Df
            .dropna(subset=['EMAIL'])
            .drop_duplicates(subset=['NOMBRE_PASAJERO'])
            [['NOMBRE_PASAJERO','EMAIL']]
        )
        
        # Hacer un merge para anotar cada fila con el EMAIL conocido (si existe).
        # Se crea temporalmente la columna 'EMAIL_KNOWN' para contener el email mapeado.
        Df = Df.merge(
            known,
            on='NOMBRE_PASAJERO',
            how='left',
            suffixes=('','_KNOWN')
        )
        
        # Rellenar la columna EMAIL con EMAIL_KNOWN cuando EMAIL original esté vacío.
        # Esto prioriza el email original; si es NaN, toma el conocido.
        Df['EMAIL'] = Df['EMAIL'].fillna(Df['EMAIL_KNOWN'])
        
        # Eliminar la columna auxiliar EMAIL_KNOWN que ya no se necesita.
        Df.drop(columns=['EMAIL_KNOWN'], inplace=True)
        
        # Dominio que se usará para generar emails genéricos (placeholder).
        dominio = 'ejemplo.com'
        
        # Obtener la lista de nombres únicos sin email asignado.
        nombres_sin_email = Df.loc[Df['EMAIL'].isna(), 'NOMBRE_PASAJERO'].unique()
        
        # Crear un diccionario mapping nombre -> email genérico (ej: "juan.perez@ejemplo.com").
        # Nota: se hace una transformación simple (lower + reemplazar espacios por puntos).
        generic_map = {
            nombre: f"{nombre.lower().replace(' ','.')}@{dominio}"
            for nombre in nombres_sin_email
        }
        
        # Asegurar que la columna EMAIL es de tipo object (cadena).
        Df['EMAIL'] = Df['EMAIL'].astype('object')
        
        # Rellenar los emails faltantes usando el mapping generado.
        # Se usa map sobre NOMBRE_PASAJERO y fillna para no sobrescribir emails existentes.
        Df['EMAIL'] = Df['EMAIL'].fillna(Df['NOMBRE_PASAJERO'].map(generic_map))
        
        # Corrección de categorías:
        # Si un adulto tiene promoción > 0, renombrar su descuento como "PROMOCION ESPECIAL"
        Df.loc[(Df['PORCENT_PROMO'] > 0) & (Df['DESC_DESCUENTO'] == 'ADULTO'), 'DESC_DESCUENTO'] = 'PROMOCION ESPECIAL'
        
        # Si tiene 0% de promoción pero estaba marcado como "PROMOCION ESPECIAL", devolverlo a "ADULTO"
        Df.loc[(Df['PORCENT_PROMO'] == 0) & (Df['DESC_DESCUENTO'] == 'PROMOCION ESPECIAL'), 'DESC_DESCUENTO'] = 'ADULTO'
        
        # Hacer nueva columna
        Df['ORIGEN_DESTINO'] = Df['ORIGEN'].astype(str) + '-' + Df['DESTINO'].astype(str)
        
        # Devolver el DataFrame final procesado.
        return Df
    
    def D4_Elasticity(self):
        df= self.Frame[['PAX_SUBEN','DISPONIBILIDAD_TRAMO','TARIFA_BASE_TRAMO',
                        "FECHA_CORRIDA", "HORA_SALIDA_CORRIDA", 
                        'INGRESO_TEORICO_TRAMO', 'INGRESO_TRANSP','OCUPACION_TRAMO',
                        'ORIGEN', 'DESTINO','TIPO_PASAJERO','CAPACIDAD_ASIENTOS_TRAMO',
                        'CV_CORRIDA']].copy()
        
        # Aseguramos que la columna sea tipo datetime
        df['FECHA_CORRIDA'] = pd.to_datetime(df['FECHA_CORRIDA'])
        
        # Obtenemos el año más grande presente en los datos
        anio_maximo = df['FECHA_CORRIDA'].dt.year.max()
        
        # Filtramos: tomamos todos los datos cuyo año sea menor al año máximo
        # Esto excluye el año actual (el más grande) y deja todo lo anterior
        df = df[df['FECHA_CORRIDA'].dt.year == anio_maximo-1].copy()
        
        return df