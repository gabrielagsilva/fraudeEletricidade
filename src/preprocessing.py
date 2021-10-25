import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler


no_transformer = ["consumo_medio_mensal", "temperatura_maxima", "temperatura_minima"]
ordinal_transformer = ["local_medidor", "tipo_medidor", "numero_fases"]
X_transformer = ColumnTransformer(
    transformers=[
        ('col_keep', 'passthrough', no_transformer),                 # manter colunas sem aplicar transformacao
        ("ordinal", OrdinalEncoder(dtype=int), ordinal_transformer)  # transformar atributos nominais em ordinais
    ]
)
pipeline = Pipeline([
    ('categorical', X_transformer),        # primeira etapa: transformar atributos nominais em ordinais
    ('standardscaler', StandardScaler()),  # padronização dos dados
])

# processar dataset
df = pd.read_csv(open("data/dataset_subestacao.csv"), header=0)
X, y = df.iloc[:, 1:-1], df['classe_cliente']
dataset = pd.DataFrame(pipeline.fit_transform(X), columns=no_transformer+ordinal_transformer)
dataset.insert(0, 'id_subestacao', df['id_subestacao'])
dataset['classe_cliente'] = df['classe_cliente'] == "fraudador"
dataset.to_csv("data/dataset_subestacao_t.csv", index=None)

# salvar pipeline
with open('data/preprocessamento.pipeline', 'wb') as f:
    pickle.dump(pipeline, f)
