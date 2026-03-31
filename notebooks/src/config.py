from pathlib import Path


PASTA_PROJETO = Path(__file__).resolve().parents[2]

PASTA_DADOS = PASTA_PROJETO / "dados"

DADOS_ORIGINAIS = PASTA_DADOS / "ml_project1_data.csv"
DADOS_LIMPOS = PASTA_DADOS / "ml_project1_data_cleaned.parquet"
DADOS_DUMMIES = PASTA_DADOS / "dummies.parquet"
DADOS_CORR = PASTA_DADOS / "correlation.parquet"
DADOS_CLUSTERED = PASTA_DADOS / "clustered.parquet"
DADOS_CLUSTERED_PCA = PASTA_DADOS / "clustered_pca.parquet"


PASTA_MODELOS = PASTA_PROJETO / "modelos"
MODELO_FINAL = PASTA_MODELOS / "modelo.joblib"


PASTA_RELATORIOS = PASTA_PROJETO / "relatorios"
RELATORIO = PASTA_RELATORIOS / "eda_ifood.html"


PASTA_IMAGENS = PASTA_RELATORIOS / "imagens"
