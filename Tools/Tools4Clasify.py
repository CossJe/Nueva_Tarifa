# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 16:16:56 2026

@author: Jesus Coss
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
import os
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

class K_means:
    
    def __init__(self):
        
        self.ruta_principal = os.getcwd()

        # Construir la ruta al archivo 
        self.json_path = os.path.join(self.ruta_principal, "Files", "DB_ClientesAgrupados.csv")
        
        self.columnas = ['FECHA_OPERACION','HORA_OPERACION', 'NOMBRE_DIA_OPERACION', 'AREA_VENTA',
               'CLASE_SERVICIO', 'ORIGEN', 'DESTINO', 'CV_CORRIDA', 'FECHA_CORRIDA',
               'HORA_SALIDA_CORRIDA', 'TIPO_PASAJERO', 'TIPO_CLIENTE','OCUPACION_TRAMO','TARJETA', 'EFECTIVO',
               'TIPO_OPERACION', 'VENTA_ANTICIPADA', 'NOMBRE_PASAJERO', 
               'PORCENT_PROMO', 'DESC_DESCUENTO', 'BOLETOS_CANCEL', 'BOLETOS_TOTALES',
               'BOLETOS_VEND','VENTA_TOTAL', 'VENTA', 'ORIGEN_CORRIDA', 'DESTINO_CORRIDA',
               'CAPACIDAD_ASIENTOS_TRAMO', 'KMS_TRAMO', 'EMPRESA', 'TIPO_CORRIDA',
               'TIPO_BUS', 'HORA_SALIDA_ORIGEN_CORRIDA','NOMBRE_DIA_CORRIDA', 'TARIFA_BASE_TRAMO',
               'EMAIL', 'PAGO_METODO', 'HORA_SALIDA_CORRIDA_', 'TOTAL_BOLETOS',
               'TOTAL_VENTA','DIAS_ANTICIPACION', 'HORAS_ANTICIPACION']
    
    def complete_data4cluster(self, Frame):
        # 1. Obtener los datos procesados
        # Df = self.modifying_data(Frame, columnas)
        Df = Frame[self.columnas]

        # 2. Crear un DataFrame vacío
        df_correo = pd.DataFrame()

        # Variable clave de agrupación
        atributo = 'EMAIL'

        print("=" * 60)
        print(f"Calculando métricas para {Df[atributo].nunique()} clientes")
        print("=" * 60)

        # 3. Calcular métricas agregadas
        if 'BOLETOS_VEND' in Df.columns:
            df_correo['SBol_Vend'] = Df.groupby(atributo)['BOLETOS_VEND'].sum()
            print("✓ SBol_Vend: Total de boletos vendidos")
        
        if 'VENTA' in Df.columns:
            df_correo['Prom_Pagado'] = Df.groupby(atributo)['VENTA'].mean()
            df_correo['Sum_Pagado'] = Df.groupby(atributo)['VENTA'].sum()
            print("✓ Prom_Pagado: Ticket promedio de compra")
            print("✓ Sum_Pagado: Monto total gastado")
        
        if 'PORCENT_PROMO' in Df.columns:
            df_correo['%_Promo'] = Df.groupby(atributo)['PORCENT_PROMO'].mean()
            print("✓ %_Promo: Promedio de % de promoción usado")
        
        if 'HORAS_ANTICIPACION' in Df.columns:
            df_correo['Prom_Horas_Ant'] = Df.groupby(atributo)['HORAS_ANTICIPACION'].mean()
            print("✓ Prom_Horas_Ant: Anticipación promedio de compra (horas)")
        
        # Contar viajes por cliente
        df_correo['total_viajes_cliente'] = Df.groupby(atributo)[atributo].count()
        print("✓ total_viajes_cliente: Total de viajes por cliente")
        
        # Métricas basadas en fecha de corrida
        if 'FECHA_CORRIDA' in Df.columns:
            primer_viaje_cliente = Df.groupby(atributo)['FECHA_CORRIDA'].min()
            ultimo_viaje_cliente = Df.groupby(atributo)['FECHA_CORRIDA'].max()
            
            df_correo['periodo_dias_cliente'] = (ultimo_viaje_cliente - primer_viaje_cliente).dt.days + 1
            df_correo['dias_entre_viajes'] = df_correo['periodo_dias_cliente'] / df_correo['total_viajes_cliente'].clip(lower=1)
            print("✓ periodo_dias_cliente: Días de actividad del cliente")
            print("✓ dias_entre_viajes: Promedio de días entre viajes")
        
        if 'FECHA_OPERACION' in Df.columns:
            Df['FECHA_OPERACION'] = pd.to_datetime(Df['FECHA_OPERACION'])
        
        # Proporción de compras con venta anticipada
        if 'VENTA_ANTICIPADA' in Df.columns:
            prop_ct = pd.crosstab(
                index=Df[atributo],
                columns=Df['VENTA_ANTICIPADA'],
                normalize='index'
            )
            
            if 'SI' in prop_ct.columns:
                df_correo['Venta_Ant'] = prop_ct['SI'].fillna(0)
                print("✓ Venta_Ant: Proporción de ventas anticipadas")
        
        # Calcular "Recencia"
        if 'FECHA_OPERACION' in Df.columns:
            df_max = Df.groupby(atributo)['FECHA_OPERACION'].max().reset_index(name='FECHA_MAX')
            fecha_max_global = Df['FECHA_OPERACION'].max()
            df_max['Recencia'] = (fecha_max_global - df_max['FECHA_MAX']).dt.days
            df_correo = df_correo.merge(df_max[['EMAIL', 'Recencia']], on='EMAIL', how='left')
            print("✓ Recencia: Días desde última compra")
        
        # Obtener la moda de variables categóricas
        columnas_categoricas = [
            ('PAGO_METODO', 'Tipo_pago'),
            ('DESC_DESCUENTO', 'Tipo_desc'),
            ('EMPRESA', 'Tipo_empresa'),
            ('ORIGEN_DESTINO', 'Tipo_destino')
        ]
        
        df_modas = df_correo.reset_index()[['EMAIL']].copy()
        dummies_para_concatenar = []
        
        for col_original, col_nueva in columnas_categoricas:
            if col_original in Df.columns:
                try:
                    df_temp = Df.groupby(atributo)[col_original].apply(
                        lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
                    ).reset_index(name=col_nueva)
                    
                    df_modas = df_modas.merge(df_temp[['EMAIL', col_nueva]], on='EMAIL', how='left')
                    
                    # Crear dummies si hay más de una categoría
                    if df_modas[col_nueva].nunique() > 1:
                        df_dummies = pd.get_dummies(df_modas[col_nueva], 
                                                   prefix=col_nueva.split('_')[0], 
                                                   drop_first=True).astype(int)
                        dummies_para_concatenar.append(df_dummies)
                        print(f"✓ Dummies {col_nueva.split('_')[0]} creadas ({df_modas[col_nueva].nunique()} categorías)")
                
                except Exception as e:
                    print(f"⚠️  Error procesando {col_original}: {e}")
        
        # Concatenar dummies
        if dummies_para_concatenar:
            df_correo = pd.concat([df_correo] + dummies_para_concatenar, axis=1)
        
        print("\n" + "=" * 60)
        print(f"✅ PROCESO COMPLETADO")
        print(f"   Métricas creadas: {len(df_correo.columns)}")
        print(f"   Clientes procesados: {len(df_correo)}")
        print("=" * 60)
        
        return df_correo
    
    def calcular_fik_promedio(self, Frame, fecha_col='FECHA_CORRIDA', 
                              ruta_col='ORIGEN_DESTINO', venta_col='VENTA', 
                              km_col='KMS_TRAMO', email_col='EMAIL'):
        """Función interna calcular_fik_promedio"""
        #Df = self.modifying_data(Frame, self.columnas)
        Df = Frame[self.columnas]
        df = Df[[fecha_col, ruta_col, venta_col, km_col, email_col]]
        
        columnas_necesarias = [fecha_col, ruta_col, venta_col, km_col, email_col]
        for col in columnas_necesarias:
            if col not in df.columns:
                raise ValueError(f"Columna '{col}' no encontrada en el DataFrame")
        
        try:
            # Calcular FIK diario por ruta
            fik_diario = (
                df.groupby([fecha_col, ruta_col])
                .apply(lambda grupo: pd.Series({
                    'FIK_DIARIO': grupo[venta_col].sum() / grupo[km_col].iloc[0]
                }))
                .reset_index()
            )
            
            if (df[km_col] == 0).any():
                print("Advertencia: Hay valores de kilómetros iguales a 0")
            
            # Unión FIK diario con datos originales
            df_con_fik = df.merge(
                fik_diario,
                on=[fecha_col, ruta_col],
                how='left'
            )
            
            # Calculo estadísticas por cliente y ruta
            resultado = (
                df_con_fik
                .groupby([email_col, ruta_col])
                .agg(
                    TOTAL_VIAJES=(fecha_col, 'count'),
                    FIK_PROMEDIO=('FIK_DIARIO', 'mean'),
                    VENTA_TOTAL=(venta_col, 'sum'),
                    ULTIMA_FECHA=(fecha_col, 'max'),
                    PRIMERA_FECHA=(fecha_col, 'min')
                )
                .reset_index()
                .round(4)  
            )
            
            resultado = resultado.sort_values(
                by=[email_col, 'FIK_PROMEDIO'], 
                ascending=[True, False]
            )
            
            print(f"   • Clientes únicos: {resultado[email_col].nunique()}")
            print(f"   • Rutas únicas: {resultado[ruta_col].nunique()}")
            print(f"   • FIK promedio general: {resultado['FIK_PROMEDIO'].mean():.2f}")
            
            return resultado[['EMAIL','TOTAL_VIAJES','ORIGEN_DESTINO','FIK_PROMEDIO']]
            
        except Exception as e:
            print(f"Error en el cálculo de FIK: {str(e)}")
            raise
    
    def procesar_con_fik(self, Frame, fecha_col='FECHA_CORRIDA',
                         ruta_col='ORIGEN_DESTINO', venta_col='VENTA',
                         km_col='KMS_TRAMO', email_col='EMAIL'):
        df_correo = self._complete_data4cluster(Frame)

        # Calcular FIK
        resultado_fik = self.calcular_fik_promedio(
            Frame, self.columnas,
            fecha_col=fecha_col,
            ruta_col=ruta_col,
            venta_col=venta_col,
            km_col=km_col,
            email_col=email_col
        )
        
        # Seleccionar columnas necesarias
        columnas_fik = ['EMAIL', 'TOTAL_VIAJES', 'ORIGEN_DESTINO', 'FIK_PROMEDIO']
        
        # Hacer merge
        df_correo = pd.merge(
            df_correo,
            resultado_fik[columnas_fik],
            on=email_col,
            how='left'
        )
        
        # Transformaciones adicionales
        df_correo['Destinos_unicos'] = df_correo.groupby(email_col)['ORIGEN_DESTINO'].transform('nunique')
        df_correo = df_correo.loc[df_correo.groupby(email_col)['TOTAL_VIAJES'].idxmax()]
        
        # Filtrar si la columna existe
        if 'SBol_Vend' in df_correo.columns:
            df_correo = df_correo[df_correo['SBol_Vend'] > 1]

        return df_correo
    
    def data_lista_para_transformar(self, Frame, aplicar_con_fik=False, **fik_params):
        
        # Aplicar FIK si se solicita
        if aplicar_con_fik:
            # Valores por defecto para FIK
            defaults = {
                'fecha_col': 'FECHA_CORRIDA',
                'ruta_col': 'ORIGEN_DESTINO',
                'venta_col': 'VENTA',
                'km_col': 'KMS_TRAMO',
                'email_col': 'EMAIL'
            }
            
            params = {**defaults, **fik_params}
            
            # Aplica procesamiento FIK
            df_correo = self.procesar_con_fik(Frame, **params)

        else:
            df_correo = self.complete_data4cluster(Frame)
            
            # Filtrar si la columna existe
            if 'SBol_Vend' in df_correo.columns:
                df_correo = df_correo[df_correo['SBol_Vend'] > 1]    
        
        return df_correo
    
    def transformar_datos(self, Frame, umbral_skew=1.5, aplicar_con_fik=False, **fik_params):
        
        # Procesar datos
        df = self.data_lista_para_transformar(
            Frame, 
            aplicar_con_fik=aplicar_con_fik,
            **fik_params
        )
        
        print(f"✅ Datos procesados: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        print("🔄 Aplicando transformaciones para reducir asimetría...")
        
        # Crear copia del DataFrame
        df_transformado = df.copy()
        
        # Diccionario para guardar información de transformaciones
        info_transformaciones = {}
        
        # Determinar columnas a transformar (solo numéricas)
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"📈 Columnas numéricas a evaluar: {len(columnas_numericas)}")
        
        # Función interna para transformar una sola columna
        def aplicar_transformacion_columna(serie):
            """Aplica la mejor transformación a una serie basada en su asimetría"""
            # Calcular asimetría original
            skew_original = serie.skew()
            
            # Si no necesita transformación
            if abs(skew_original) < umbral_skew:
                return serie, "ninguna", skew_original
            
            # Lista para almacenar resultados de transformaciones
            transformaciones_posibles = []
            
            # 1. Transformación log1p (si todos los valores son >= 0)
            if (serie >= 0).all():
                serie_log1p = np.log1p(serie)
                skew_log1p = serie_log1p.skew()
                transformaciones_posibles.append(("log1p", serie_log1p, skew_log1p))
            
            # 2. Transformación log (si todos los valores son > 0)
            if (serie > 0).all():
                serie_log = np.log(serie)
                skew_log = serie_log.skew()
                transformaciones_posibles.append(("log", serie_log, skew_log))
            
            # 3. Transformación raíz cuadrada (si todos los valores son >= 0)
            if serie.min() >= 0:
                serie_sqrt = np.sqrt(serie)
                skew_sqrt = serie_sqrt.skew()
                transformaciones_posibles.append(("sqrt", serie_sqrt, skew_sqrt))
            
            # 4. Transformación Box-Cox
            try:
                from scipy import stats
                if (serie > 0).all():
                    serie_boxcox_array, _ = stats.boxcox(serie)
                    serie_boxcox = pd.Series(serie_boxcox_array, index=serie.index)
                    skew_boxcox = serie_boxcox.skew()
                    transformaciones_posibles.append(("boxcox", serie_boxcox, skew_boxcox))
            except ImportError:
                pass
            
            # Si hay transformaciones posibles, elegir la mejor
            if transformaciones_posibles:
                mejor_transformacion = min(transformaciones_posibles, key=lambda x: abs(x[2]))
                metodo, serie_transformada, skew_despues = mejor_transformacion
                return serie_transformada, metodo, skew_original
            else:
                return serie, "ninguna", skew_original
        
        # Contadores para el resumen
        transformadas = 0
        no_transformadas = 0
        
        # Aplicar transformación a cada columna numérica
        for col in columnas_numericas:
            serie_original = df[col]
            
            if serie_original.notna().sum() < 2:
                no_transformadas += 1
                continue
            
            serie_transformada, metodo, skew_antes = aplicar_transformacion_columna(serie_original)
            df_transformado[col] = serie_transformada
            
            if metodo != "ninguna":
                skew_despues = serie_transformada.skew()
                info_transformaciones[col] = {
                    'metodo': metodo,
                    'skew_antes': skew_antes,
                    'skew_despues': skew_despues,
                    'mejoria_skew': abs(skew_antes) - abs(skew_despues),
                    'mejoria_porcentaje': ((abs(skew_antes) - abs(skew_despues)) / abs(skew_antes) * 100) 
                                          if skew_antes != 0 else 0
                }
                transformadas += 1
            else:
                no_transformadas += 1
        
        # Mostrar resumen
        print(f"\n📋 RESUMEN DE TRANSFORMACIONES:")
        print(f"   ✅ Columnas transformadas: {transformadas}")
        print(f"   ⏭️  Columnas sin transformar: {no_transformadas}")
        print(f"   📊 Umbral de skew usado: {umbral_skew}")
        
        if info_transformaciones:
            print(f"\n🔧 Transformaciones aplicadas:")
            for col, info in info_transformaciones.items():
                print(f"   • {col}: {info['metodo']} "
                      f"({info['skew_antes']:.2f} → {info['skew_despues']:.2f})")
        
        return df_transformado
    
    def clustering_k_means_completo(self, Frame, umbral_skew=1.5, optimal_k=0, 
                                    aplicar_fik=False, **fik_params):
        
        print("="*60)
        print("INICIANDO PROCESO COMPLETO DE CLUSTERING")
        print("="*60)
        
        # 1. Obtener df_correo
        print("\n📊 PASO 1: Obteniendo datos procesados...")
        df_correo = self.data_lista_para_transformar(
            Frame, 
            aplicar_con_fik=aplicar_fik,
            **fik_params
        )
        
        print(f"✅ Datos procesados: {df_correo.shape[0]} filas, {df_correo.shape[1]} columnas")
        
        # Guardar una copia original
        df_correo_original = df_correo.copy()
        
        # 2. Transformar datos
        print("\n📊 PASO 2: Transformando datos...")
        df_transformado = self.transformar_datos(
            Frame,  umbral_skew,
            aplicar_con_fik=aplicar_fik,
            **fik_params
        )
        
        # 3. Verificar columnas necesarias
        columnas_requeridas = ['EMAIL', 'SBol_Vend']
        for col in columnas_requeridas:
            if col not in df_transformado.columns:
                raise ValueError(f"Columna requerida '{col}' no encontrada.")
        
        # 4. Filtrar clientes con más de 1 boleto
        print("\n🎯 PASO 3: Filtrando clientes con más de 1 boleto...")
        registros_antes = len(df_transformado)
        dfMoreThanOne = df_transformado[df_transformado['SBol_Vend'] > 0].copy()
        registros_despues = len(dfMoreThanOne)
        
        print(f"   Clientes filtrados: {registros_antes} → {registros_despues} (SBol_Vend > 0)")
        
        if len(dfMoreThanOne) == 0:
            raise ValueError("No hay clientes con SBol_Vend > 0 después del filtrado")
        
        # 5. Preparar variables para clustering
        print("\n🔢 PASO 4: Preparando variables para clustering...")
        
        # Ubicar la posición de la columna EMAIL
        try:
            col_index = list(dfMoreThanOne.columns).index('EMAIL')
        except ValueError:
            col_index = 0
        
        # Tomar solo variables numéricas
        X = dfMoreThanOne[dfMoreThanOne.columns[col_index+1:]]
        X = X.select_dtypes(include=[np.number])
        
        if X.shape[1] == 0:
            raise ValueError("No hay variables numéricas para clustering")
        
        print(f"   Variables para clustering: {X.shape[1]} columnas numéricas")
        
        # 6. Escalado Min-Max
        print("📏 PASO 5: Escalando variables (Min-Max)...")
        minmax_scaler = MinMaxScaler()
        X_escalado = minmax_scaler.fit_transform(X)
        X_escalado = pd.DataFrame(X_escalado, columns=X.columns)
        
        # 7. Encontrar K óptimo
        if optimal_k == 0:
            print("🔍 PASO 6: Buscando número óptimo de clusters...")
            max_silhouette_score = -1
            K_range = range(2, min(11, len(X_escalado)))
            
            for k in K_range:
                kmeans_model = KMeans(n_clusters=k, n_init='auto', random_state=42)
                kmeans_model.fit(X_escalado)
                
                if len(np.unique(kmeans_model.labels_)) > 1:
                    score = silhouette_score(X_escalado, kmeans_model.labels_)
                    
                    if score > max_silhouette_score:
                        max_silhouette_score = score
                        optimal_k = k
            
            print(f"   K óptimo encontrado: {optimal_k} (silhouette: {max_silhouette_score:.3f})")
        else:
            print(f"🔍 PASO 6: Usando K proporcionado: {optimal_k}")
        
        # 8. Entrenar modelo KMeans
        print(f"🎯 PASO 7: Entrenando KMeans con K={optimal_k}...")
        modelo_entrenado = KMeans(n_clusters=optimal_k, n_init='auto', random_state=42)
        modelo_entrenado.fit(X_escalado)
        
        # 9. Asignar clusters
        print("🏷️  PASO 8: Asignando clusters a clientes...")
        dfMoreThanOne['Cluster'] = modelo_entrenado.labels_
        
        Modelo_cluster = dfMoreThanOne[['EMAIL', 'Cluster']]
        dfMoreThanOne = pd.merge(df_correo_original, Modelo_cluster, on='EMAIL', how='left')
        
        # 10. Mostrar resultados
        print("\n📊 RESULTADOS FINALES:")
        print("="*40)
        
        # Distribución de clusters
        distribucion = dfMoreThanOne['Cluster'].value_counts().sort_index()
        print(f"\nDistribución de clusters ({len(distribucion)} clusters):")
        for cluster, count in distribucion.items():
            porcentaje = (count / len(dfMoreThanOne)) * 100
            print(f"  Cluster {cluster}: {count:4d} clientes ({porcentaje:5.1f}%)")
        
        print("="*60)
        print(f"✅ Pipeline completado exitosamente")
        print(f"   • Clientes clusterizados: {len(dfMoreThanOne)}")
        print(f"   • Número de clusters: {optimal_k}")
        
        return dfMoreThanOne
    
    def fit(self, Frame, umbral_skew=1.5, optimal_k=5, aplicar_fik=False, **fik_params):
        
        if self.columnas is None:
            self.columnas = Frame.columns.tolist()
        
        print("INICIANDO CLUSTERING CON CLASE K_MEANS")
        print("="*60)
        
        # Ejecutar completo
        resultado = self.clustering_k_means_completo(
            Frame=Frame,
            umbral_skew=umbral_skew,
            optimal_k=optimal_k,
            aplicar_fik=aplicar_fik,
            **fik_params
        )
        
        resultado.to_csv(self.json_path)
        return resultado

"""
# Función de conveniencia para usar directamente
def ejecutar_kmeans(Frame, columnas=None, **kwargs):
    kmeans = K_means()
    return kmeans.fit(Frame, columnas, **kwargs)
"""

