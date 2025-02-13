# %% [markdown]
# ## IMPORTAR LIBRERIAS

# %%
import pandas as pd
import numpy as np
import os
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Cargar archivos

# %%
### Características Equipos

equipos = pd.read_csv('c:/repo_remoto/etapa2/data/Caracteristicas_Equipos.csv')
equipos_df = pd.DataFrame(equipos)

# %%
### Historicos Ordenes

ordenes = pd.read_csv('c:/repo_remoto/etapa2/data/Historicos_Ordenes.csv')
ordenes_df = pd.DataFrame(ordenes)

# %%
### Registros Condiciones

condiciones = pd.read_csv('c:/repo_remoto/etapa2/data/Registros_Condiciones.csv')
condiciones_df = pd.DataFrame(condiciones)

# %% [markdown]
# ### LIMPIEZA Y PERFILADO DE CARACTERISTICAS EQUIPOS

# %%
# Mostrar las primeras filas de cada dataset
equipos_df.head()

# %%
# Revisar info del df
equipos_df.info()

# %%
# Datos vacios
print('Datos vacios en Caracteristicas_Equipos.csv\n',equipos_df.isna().sum())

# %%
# Extremos
z_scores = (equipos_df-equipos_df.mean(numeric_only=True)) / \
    equipos_df.std(numeric_only=True)
z_scores_abs = z_scores.apply(np.abs)
print(tabulate(z_scores_abs, headers='keys'))

# %%
umbral = 3

out_mask = ~z_scores[z_scores_abs > umbral].isna()
print('\nOutliers per column:\n')
print(out_mask.sum())

# %%
# Mostrar todas las filas duplicadas
print('\nSumatorio duplicados en sales', equipos_df.duplicated().sum())
equipos_df[equipos_df.duplicated(keep=False)]

# %%
# Elimino filas duplicadas

equipos_df_clean = equipos_df.drop_duplicates(keep='first')
print('\nSumatorio duplicados en equipos_df:', equipos_df_clean.duplicated().sum())


# %%
equipos_df_clean

# %%
equipos_df_clean.info()

# %%
# Cardinalidad

# Calcular el sumatorio de valores para cada columna
sum_value_equipos = equipos_df_clean.count()

# Calcular el sumatorio de valores únicos para cada columna
unique_sum_equipos = equipos_df_clean.nunique()

# Crear un nuevo DataFrame para mostrar ambos sumatorios
result_df_equipos = pd.DataFrame({
    'Sumatorio de valores': sum_value_equipos,
    'Sumatorio de valores únicos': unique_sum_equipos
})

print(result_df_equipos)

# %%
# Columnas a 'category'

equipos_df_clean[['Tipo_Equipo', 'Fabricante', 'Modelo']] = equipos_df_clean[[
    'Tipo_Equipo', 'Fabricante', 'Modelo']].astype('category')

equipos_df_clean.info()

# %%
equipos_df_clean.describe()

# %%
equipos_df_clean.describe(include='category')

# %%
# Existen equipos con Potencia menor de 0

print("\nEquipos con potencia <0 kW:", len(equipos_df_clean[equipos_df_clean['Potencia_kW'] < 0]))
equipos_df_clean[equipos_df_clean['Potencia_kW'] < 0]

# %%
# Busco filas que tengan el mismo Tipo_Equipo + Fabricante + Modelo que la fila con Potencia_kW <0 y además muestro al final su sumatorio

# Filtrar filas con Costo_Mantenimiento vacío
filas_pot = equipos_df_clean[equipos_df_clean['Potencia_kW'] < 0]

# Inicializar contador de coincidencias
pot_coincidencias = 0

# Buscar coincidencias en todo el DataFrame
for index, row in filas_pot.iterrows():
    coincidencias4 = equipos_df_clean[(equipos_df_clean['Tipo_Equipo'] == row['Tipo_Equipo']) &
                                      (equipos_df_clean['Fabricante'] == row['Fabricante']) & 
                               (equipos_df_clean['Modelo'] == row['Modelo'])]
    if len(coincidencias4) > 1:
        print(tabulate(coincidencias4, headers='keys'))
        pot_coincidencias += 1

# Mostrar el sumatorio de las filas NaN que han coincidido
print(f'\nSumatorio de filas NaN que han coincidido: {pot_coincidencias}')

# %%
# Reemplazo el valor de Potencia_kW < 0 por la media de las fils con mismo Tipo_Equipo + Fabricante + Modelo

# Filtrar filas con Potencia_kW < 0
filas_pot_negativas = equipos_df_clean[equipos_df_clean['Potencia_kW'] < 0]

# Reemplazar valores negativos por la media de sus valores filtrados
for index, row in filas_pot_negativas.iterrows():
    # Filtrar filas con el mismo Tipo_Equipo, Fabricante y Modelo
    coincidencias = equipos_df_clean[(equipos_df_clean['Tipo_Equipo'] == row['Tipo_Equipo']) &
                                     (equipos_df_clean['Fabricante'] == row['Fabricante']) &
                                     (equipos_df_clean['Modelo'] == row['Modelo']) &
                                     (equipos_df_clean['Potencia_kW'] >= 0)]
    if not coincidencias.empty:
        # Calcular la media de Potencia_kW para las coincidencias
        media_potencia = coincidencias['Potencia_kW'].mean()
        # Reemplazar el valor negativo con la media calculada
        equipos_df_clean.at[index, 'Potencia_kW'] = media_potencia

# Mostrar el sumatorio de las filas con Potencia_kW <0
print(f'\nSumatorio de filas con Potencia_kW < 0 kW: {len(equipos_df_clean[equipos_df_clean["Potencia_kW"] < 0])}')

# %%
# Value_counts te muestra cuantos valores hay de cada valor único en la columna selecionada

print(equipos_df_clean[['Tipo_Equipo', 'Fabricante', 'Modelo']].value_counts())

# %%
# Frecuencias en equipos

for col in equipos_df_clean.columns:
    print('\n- Frecuencias para "{0}"'.format(col), '\n')
    print(equipos_df_clean[col].value_counts())

# %%
# Correlación

corr_equipos = equipos_df_clean.corr('pearson', numeric_only=True)
corr_equipos

# %%
# Correlación

corr_equipos[(np.abs(corr_equipos) >= 0.7) & (np.abs(corr_equipos) >= 0.7)]

# %%
# Sesgo

skw_equipos = equipos_df_clean.skew(numeric_only=True)
skw_equipos

# %%
# Sesgo

skw_equipos[np.abs(skw_equipos) > 2]

# %%
# Kurtosis

kurt_equipos = equipos_df_clean.kurt(numeric_only=True)
kurt_equipos

# %%
# Kurtosis

kurt_equipos[np.abs(kurt_equipos) > 3]

# %%
# Evaluar la unicidad de posibles claves primarias

# Crear una lista para almacenar los resultados
unike_keys_equipos = []

# Iterar sobre cada columna del DataFrame
for column in equipos_df_clean.columns:
    # Verificar si la columna tiene valores únicos
    if equipos_df_clean[column].is_unique:
        unike_keys_equipos.append(column)

# Mostrar las posibles claves primarias
print("Posibles claves primarias en equipos_df_clean:")
print(unike_keys_equipos)

# %%
equipos_df_clean.describe()

# %%
equipos_df_clean.describe(include='category')

# %% [markdown]
# ### LIMPIEZA Y PERFILADO DE HISTORICOS ORDENES

# %%
# Mostrar las primeras filas de cada dataset
ordenes_df.head()

# %%
# Revisar info del df
ordenes_df.info()

# %%
# Datos vacios
print('Datos vacios en Historicos_Ordenes.csv\n',ordenes_df.isna().sum())

# %%
# Extremos
z_scores = (ordenes_df-ordenes_df.mean(numeric_only=True)) / \
    ordenes_df.std(numeric_only=True)
z_scores_abs = z_scores.apply(np.abs)
print(tabulate(z_scores_abs, headers='keys'))

# %%
umbral = 3

out_mask = ~z_scores[z_scores_abs > umbral].isna()
print('\nOutliers per column:\n')
print(out_mask.sum())

# %%
# Busco filas que tengan el mismo ID_Equipo + Duracion_Horas que la fila con Costo_Mantenimiento vacío y además muestro al final su sumatorio

# Filtrar filas con Costo_Mantenimiento vacío
filas_vacias = ordenes_df[ordenes_df['Costo_Mantenimiento'].isna()]

# Inicializar contador de coincidencias
nan_coincidencias = 0

# Buscar coincidencias en todo el DataFrame
for index, row in filas_vacias.iterrows():
    coincidencias1 = ordenes_df[(ordenes_df['ID_Equipo'] == row['ID_Equipo']) & 
                               (ordenes_df['Duracion_Horas'] == row['Duracion_Horas'])]
    if len(coincidencias1) > 1:
        print(tabulate(coincidencias1, headers='keys'))
        nan_coincidencias += 1

# Mostrar el sumatorio de las filas NaN que han coincidido
print(f'\nSumatorio de filas NaN que han coincidido: {nan_coincidencias}')

# %%
# Primero sustituyo el valor de Costo_Mantenimiento vacío por el valor de la línea que coincida con ID_Equipo + Duración_Horas + Ubicación

ordenes_df_clean = pd.DataFrame(ordenes_df)

# Filtrar filas con Costo_Mantenimiento vacío
filas_vacias = ordenes_df_clean[ordenes_df_clean['Costo_Mantenimiento'].isna()]

# Buscar coincidencias en todo el DataFrame y reemplazar valores vacíos
for index, row in filas_vacias.iterrows():
    coincidencias2 = ordenes_df_clean[(ordenes_df_clean['ID_Equipo'] == row['ID_Equipo']) & 
                                (ordenes_df_clean['Duracion_Horas'] == row['Duracion_Horas']) &
                                (ordenes_df_clean['Ubicacion'] == row['Ubicacion']) & 
                                (ordenes_df_clean['Costo_Mantenimiento'].notna())]
    if not coincidencias2.empty:
        # Reemplazar el valor vacío en ordenes_df_clean con el valor de la primera coincidencia encontrada
        ordenes_df_clean.at[index, 'Costo_Mantenimiento'] = coincidencias2.iloc[0]['Costo_Mantenimiento']


# %%
# Segundo sustituyo el valor de Costo_Mantenimiento vacío por el valor de la línea que coincida ID_Equipo + Duración_Horas

# Filtrar filas con Costo_Mantenimiento vacío
filas_vacias = ordenes_df_clean[ordenes_df_clean['Costo_Mantenimiento'].isna()]

# Buscar coincidencias en todo el DataFrame y reemplazar valores vacíos
for index, row in filas_vacias.iterrows():
    coincidencias3 = ordenes_df_clean[(ordenes_df_clean['ID_Equipo'] == row['ID_Equipo']) & 
                               (ordenes_df_clean['Duracion_Horas'] == row['Duracion_Horas']) &
                               (ordenes_df_clean['Costo_Mantenimiento'].notna())]
    if not coincidencias3.empty:
        # Reemplazar el valor vacío con el valor de la primera coincidencia encontrada
        ordenes_df_clean.at[index, 'Costo_Mantenimiento'] = coincidencias3.iloc[0]['Costo_Mantenimiento']



# %%
# Vuelvo a revisar los datos vacíos donde ya he sustituidos los anteriores reduciendo a 33 los valores vacíos
print('Datos vacios en Historicos_Ordenes.csv\n',ordenes_df_clean.isna().sum())

# %%
# Sustituyo valores vacíos por su media

# Calcular la media de la columna Costo_Mantenimiento
mean_costo_mantenimiento = ordenes_df_clean['Costo_Mantenimiento'].mean()

# Sustituir los valores vacíos por la media
ordenes_df_clean['Costo_Mantenimiento'].fillna(mean_costo_mantenimiento, inplace=True)

# Vuelvo a revisar los datos vacíos donde ya he sustituidos los anteriores reduciendo en 17 los valores vacíos
print('Datos vacios en Historicos_Ordenes.csv\n',ordenes_df_clean.isna().sum())

# %%
# Vuelvo a revisar info del df
ordenes_df_clean.info()

# %%
# Mostrar todas las filas duplicadas
print('\nSumatorio duplicados en ordenes', ordenes_df_clean.duplicated().sum())
ordenes_df_clean[ordenes_df_clean.duplicated(keep=False)]

# %%
# Elimino filas duplicadas una vez las he analizado

ordenes_df_cleaned = ordenes_df_clean.drop_duplicates(keep='first')
print('\nSumatorio duplicados en ordenes_df_clean:', ordenes_df_cleaned.duplicated().sum())

# %%
# Vuelvo a revisar info del df
ordenes_df_cleaned.info()

# %%
# Cardinalidad

# Calcular el sumatorio de valores para cada columna
sum_values_ordenes = ordenes_df_cleaned.count()

# Calcular el sumatorio de valores únicos para cada columna
unique_sum_ordenes = ordenes_df_cleaned.nunique()

# Crear un nuevo DataFrame para mostrar ambos sumatorios
result_df_ordenes = pd.DataFrame({
    'Sumatorio de valores': sum_values_ordenes,
    'Sumatorio de valores únicos': unique_sum_ordenes
})

print(result_df_ordenes)

# %%
# Columnas object a 'category'

ordenes_df_cleaned[['Tipo_Mantenimiento', 'Ubicacion']] = ordenes_df_cleaned[[
    'Tipo_Mantenimiento', 'Ubicacion']].astype('category')

# Convertir fechas a a dateTime
ordenes_df_cleaned["Fecha"] = pd.to_datetime(ordenes_df_cleaned["Fecha"], errors='coerce')

ordenes_df_cleaned.info()

# %%
ordenes_df_cleaned.describe()

# %%
ordenes_df_cleaned.describe(include='category')

# %%
# Value_counts te muestra cuantos valores hay de cada valor único en la columna selecionada

print(ordenes_df_cleaned[['Tipo_Mantenimiento', 'Ubicacion']].value_counts())

# %%
# Frecuencias en ordenes

for col in ordenes_df_cleaned.columns:
    print('\n- Frecuencias para "{0}"'.format(col), '\n')
    print(ordenes_df_cleaned[col].value_counts())

# %%
# Correlación

corr_ordenes = ordenes_df_cleaned.corr('pearson', numeric_only=True)
corr_ordenes

# %%
# Correlación

corr_ordenes[(np.abs(corr_ordenes) >= 0.7) & (np.abs(corr_ordenes) >= 0.7)]

# %%
# Sesgo

skw_ordenes = ordenes_df_cleaned.skew(numeric_only=True)
skw_ordenes

# %%
# Sesgo

skw_ordenes[np.abs(skw_ordenes) > 2]

# %%
# Kurtosis

kurt_ordenes = ordenes_df_cleaned.kurt(numeric_only=True)
kurt_ordenes

# %%
# Kurtosis

kurt_ordenes[np.abs(kurt_ordenes) > 3]

# %%
# Evaluar la unicidad de posibles claves primarias

# Crear una lista para almacenar los resultados
unike_keys_ordenes = []

# Iterar sobre cada columna del DataFrame
for column in ordenes_df_cleaned.columns:
    # Verificar si la columna tiene valores únicos
    if ordenes_df_cleaned[column].is_unique:
        unike_keys_ordenes.append(column)

# Mostrar las posibles claves primarias
print("Posibles claves primarias en ordenes_df_cleaned:")
print(unike_keys_ordenes)

# %% [markdown]
# ### LIMPIEZA Y PERFILADO DE REGISTROS CONDICIONES

# %%
# Mostrar las primeras filas de cada dataset
condiciones_df.head()

# %%
# Revisar info del df
condiciones_df.info()

# %%
# Datos vacios
print('Datos vacios en Caracteristicas_Equipos.csv\n',condiciones_df.isna().sum())

# %%
# Relleno los valores vacíos de Horas_Operativas con la media de su columna una vez analizados sus valores

condiciones_df_fill = pd.DataFrame(condiciones_df)

# Calcular la media solo de las columna Horas_Operativas
mean_values_horas = condiciones_df['Horas_Operativas'].mean()

# Rellenar los valores vacíos con la media de su columna
condiciones_df_fill['Horas_Operativas'].fillna(mean_values_horas, inplace=True)

# Verificar que no queden valores vacíos
print('Datos vacios en condiciones_df_fill\n', condiciones_df_fill.isna().sum())


# %%
# Extremos
z_scores = (condiciones_df_fill - condiciones_df_fill.mean(numeric_only=True)) / \
    condiciones_df_fill.std(numeric_only=True)
z_scores_abs = z_scores.apply(np.abs)
print(tabulate(z_scores_abs, headers='keys'))

# %%
umbral = 3

out_mask = ~z_scores[z_scores_abs > umbral].isna()
print('\nOutliers per column:\n')
print(out_mask.sum())

# %%
# Filtrar las filas con outliers
outliers = condiciones_df_fill[(z_scores_abs > umbral).any(axis=1)]

# Mostrar las filas con outliers
print(outliers)

# %%
# Remplazo los outliers con la media de su columna después de analizar sus características

condiciones_df_clean = pd.DataFrame(condiciones_df_fill)

# Calcular la media de la columna Temperatura_C
mean_temperatura = condiciones_df_clean['Temperatura_C'].mean()

# Reemplazar los outliers por la media
condiciones_df_clean.loc[(z_scores_abs['Temperatura_C'] > umbral), 'Temperatura_C'] = mean_temperatura

# Verificar que los outliers han sido reemplazados
print(condiciones_df_clean[(z_scores_abs['Temperatura_C'] > umbral)])

# %%
# Mostrar todas las filas duplicadas
print('\nSumatorio duplicados en Registros_Condiciones', condiciones_df_clean.duplicated().sum())
condiciones_df_clean[condiciones_df_clean.duplicated(keep=False)]

# %%
condiciones_df_clean.info()

# %%
# Cardinalidad

# Calcular el sumatorio de valores para cada columna
sum_value_condiciones = condiciones_df_clean.count()

# Calcular el sumatorio de valores únicos para cada columna
unique_sum_condiciones = condiciones_df_clean.nunique()

# Crear un nuevo DataFrame para mostrar ambos sumatorios
result_df_condicones = pd.DataFrame({
    'Sumatorio de valores': sum_value_condiciones,
    'Sumatorio de valores únicos': unique_sum_condiciones
})

print(result_df_condicones)

# %%
# Convertir fechas a a dateTime

condiciones_df_clean["Fecha"] = pd.to_datetime(condiciones_df_clean["Fecha"], errors='coerce')

condiciones_df_clean.info()

# %%
condiciones_df_clean.describe()

# %%
# Frecuencias en condiciones

for col in condiciones_df_clean.columns:
    print('\n- Frecuencias para "{0}"'.format(col), '\n')
    print(condiciones_df_clean[col].value_counts())

# %%
# Correlación

corr_condiciones = condiciones_df_clean.corr('pearson', numeric_only=True)
corr_condiciones

# %%
# Correlación

corr_condiciones[(np.abs(corr_condiciones) >= 0.7) & (np.abs(corr_condiciones) >= 0.7)]

# %%
# Sesgo

skw_condiciones = condiciones_df_clean.skew(numeric_only=True)
skw_condiciones

# %%
# Sesgo

skw_condiciones[np.abs(skw_condiciones) > 2]

# %%
# Kurtosis

kurt_condiones = condiciones_df_clean.kurt(numeric_only=True)
kurt_condiones

# %%
# Kurtosis

kurt_condiones[np.abs(kurt_condiones) > 3]

# %%
# Evaluar la unicidad de posibles claves primarias

# Crear una lista para almacenar los resultados
unike_keys_condiciones = []

# Iterar sobre cada columna del DataFrame
for column in condiciones_df_clean.columns:
    # Verificar si la columna tiene valores únicos
    if condiciones_df_clean[column].is_unique:
        unike_keys_condiciones.append(column)

# Mostrar las posibles claves primarias
print("Posibles claves primarias en condiciones_df_clean:")
print(unike_keys_condiciones)

# %%
condiciones_df_clean.describe()

# %%
# Obtener la ruta del directorio actual
current_directory = os.getcwd()

print("La ruta del directorio actual es:", current_directory)

# %%
# Extraer los DataFrame limpios a un archivo CSV

equipos_df_clean.to_csv('c:/repo_remoto/etapa2/data/Caracteristicas_Equipos_limpio.csv', index=False)
ordenes_df_cleaned.to_csv('c:/repo_remoto/etapa2/data/Historicos_Ordenes_limpio.csv', index=False)
condiciones_df_clean.to_csv('c:/repo_remoto/etapa2/data/Registros_Condiciones_limpio.csv', index=False)

print("El DataFrame equipos_df_clean, ordenes_df_cleaned y condiciones_df_clean se ha extraído a *.csv")


