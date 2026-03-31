# importações

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter, PercentFormatter
from matplotlib.colors import ListedColormap

from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix,
    PredictionErrorDisplay, 
    silhouette_score
)


from .models import RANDOM_STATE

sns.set_theme(palette="bright")

PALETTE = "coolwarm"
SCATTER_ALPHA = 0.2

# Main

def plot_coeficientes(df_coefs, tituto="Coeficientes"):

    """
        Gera um gráfico de barras horizontais para visualizar coeficientes de um modelo.

        Esta função cria uma visualização rápida da importância ou magnitude de coeficientes,
        incluindo uma linha vertical no zero para facilitar a identificação de valores
        positivos e negativos.

        Args:
            df_coefs (pd.DataFrame): DataFrame contendo os coeficientes. Geralmente 
                o índice contém o nome das variáveis e uma coluna com os valores.
            titulo (str, opcional): O título que será exibido no topo do gráfico. 
                O padrão é "Coeficientes".

        Returns:
            None: A função apenas exibe o gráfico na tela usando plt.show().

        Exemplo:
            >>> coefs = pd.DataFrame({'coef': [1.2, -0.5]}, index=['Var1', 'Var2'])
            >>> plot_coeficientes(coefs, "Importância das Variáveis")
        """



    
    df_coefs.plot.barh()

    plt.title(tituto)
    plt.axvline(x=0, color=".5")

    plt.xlabel("Coeficientes")
    plt.gca().get_legend().remove()

    plt.show()


def plot_residuos(y_true, y_pred):

    """
    Gera um painel de diagnóstico visual para avaliar os resíduos de um modelo de regressão.

    O painel consiste em três visualizações:
    1. Histograma dos resíduos com estimativa de densidade de kernel (KDE).
    2. Gráfico de Resíduos vs. Valores Preditos (para verificar linearidade e variância).
    3. Gráfico de Valores Reais vs. Valores Preditos (para verificar a precisão do ajuste).

    Args:
        y_true (np.ndarray ou pd.Series): Valores reais (alvo) observados nos dados.
        y_pred (np.ndarray ou pd.Series): Valores previstos pelo modelo de regressão.

    Returns:
        None: A função renderiza o painel de gráficos diretamente via plt.show().

    Observações esperadas:
        - O gráfico de resíduos ideal deve mostrar os erros distribuídos aleatoriamente 
          em torno de zero, sem padrões de funil ou curvas.
        - No gráfico de 'Actual vs Predicted', os pontos devem se aproximar da linha diagonal.
    """

    residuos = y_true - y_pred

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    sns.histplot(residuos, kde=True, ax=axs[0])

    error_display_01 = PredictionErrorDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, kind="residual_vs_predicted", ax=axs[1]
    )

    error_display_02 = PredictionErrorDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, kind="actual_vs_predicted", ax=axs[2]
    )

    plt.tight_layout()

    plt.show()


def plot_residuos_estimador(estimator, X, y, eng_formatter=False, fracao_amostra=0.25):

    """
        Gera diagnósticos de resíduos diretamente a partir de um estimador e dados de teste.

        Esta função utiliza a API do Scikit-Learn para extrair predições do modelo e 
        construir um painel visual com a distribuição dos erros e gráficos de dispersão,
        permitindo o controle da densidade de pontos e formatação numérica.

        Args:
            estimator (BaseEstimator): Modelo de regressão treinado do Scikit-Learn.
            X (array-like ou pd.DataFrame): Matriz de características (features) para predição.
            y (array-like ou pd.Series): Vetor de valores reais (target).
            eng_formatter (bool, opcional): Se True, aplica o EngFormatter aos eixos, 
                útil para valores muito grandes ou pequenos (ex: 1k, 1M). Padrão é False.
            fracao_amostra (float, opcional): Fração dos dados (0.0 a 1.0) a ser exibida 
                nos gráficos de dispersão para evitar sobreposição (overplotting). 
                Padrão é 0.25.

        Returns:
            None: A função exibe o painel de gráficos via plt.show().

        Observação:
            - A função utiliza variáveis globais como `RANDOM_STATE` para reprodutibilidade 
            da subamostra e `SCATTER_ALPHA` para transparência dos pontos.
            - O primeiro gráfico exibe a distribuição (Histograma/KDE) dos resíduos.
            - O segundo e terceiro gráficos são gerados via `PredictionErrorDisplay`.
        """

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    error_display_01 = PredictionErrorDisplay.from_estimator(
        estimator,
        X,
        y,
        kind="residual_vs_predicted",
        ax=axs[1],
        random_state=RANDOM_STATE,
        scatter_kwargs={"alpha": SCATTER_ALPHA},
        subsample=fracao_amostra,
    )

    error_display_02 = PredictionErrorDisplay.from_estimator(
        estimator,
        X,
        y,
        kind="actual_vs_predicted",
        ax=axs[2],
        random_state=RANDOM_STATE,
        scatter_kwargs={"alpha": SCATTER_ALPHA},
        subsample=fracao_amostra,
    )

    residuos = error_display_01.y_true - error_display_01.y_pred

    sns.histplot(residuos, kde=True, ax=axs[0])

    if eng_formatter:
        for ax in axs:
            ax.yaxis.set_major_formatter(EngFormatter())
            ax.xaxis.set_major_formatter(EngFormatter())

    plt.tight_layout()

    plt.show()


def plot_comparar_metricas_modelos(df_resultados):
    
    """
        Gera um painel comparativo de métricas de desempenho para diferentes modelos de regressão.

        Args:
            df_resultados (pd.DataFrame): DataFrame contendo os resultados da validação. 
                Deve conter obrigatoriamente as colunas: 
                'model', 'time_seconds', 'test_r2', 'test_neg_mean_absolute_error' 
                e 'test_neg_root_mean_squared_error'.

        Returns:
            None: A função renderiza o painel de gráficos diretamente via plt.show().

        Obsevação:
            - O parâmetro `showmeans=True` adiciona um marcador para a média aritmética, 
            permitindo comparar a média com a mediana (linha central do box).
            - `sharex=True` garante que os nomes dos modelos no eixo X fiquem alinhados 
            entre os gráficos superiores e inferiores.
            - As métricas de erro (MAE/RMSE) costumam vir como valores negativos no 
            Scikit-Learn (para fins de otimização); a função as exibe conforme estão no DataFrame.
        """
    

    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

    comparar_metricas = [
        "time_seconds",
        "test_r2",
        "test_neg_mean_absolute_error",
        "test_neg_root_mean_squared_error",
    ]

    nomes_metricas = [
        "Tempo (s)",
        "R²",
        "MAE",
        "RMSE",
    ]

    for ax, metrica, nome in zip(axs.flatten(), comparar_metricas, nomes_metricas):
        sns.boxplot(
            x="model",
            y=metrica,
            data=df_resultados,
            ax=ax,
            showmeans=True,
        )
        ax.set_title(nome)
        ax.set_ylabel(nome)
        ax.tick_params(axis="x", rotation=90)

    plt.tight_layout()

    plt.show()


def plot_pair(
        dataframe,
        colunas_analise,
        coluna_hue=None,

        legend_on=True, 
        legend_title='Legenda',
        legends_labels=None,
        legends_cols=2,
        
        corner=True,
        alpha=0.5,
        **kwargs
):
    
    """
        Cria uma matriz de gráficos de dispersão (Pair Plot) customizada para análise multivariada.

        Args:
            dataframe (pd.DataFrame): O conjunto de dados principal.
            colunas_analise (list[str]): Lista com os nomes das colunas numéricas para analisar.
            coluna_hue (str, opcional): Coluna categórica para colorir os pontos (segmentação).
            legend_on (bool, opcional): Se True, renderiza uma legenda centralizada no topo. 
                Padrão é True.
            legend_title (str, opcional): Título da legenda. Padrão é 'Legenda'.
            legends_labels (dict, opcional): Dicionário para renomear as categorias da `coluna_hue`.
                Ex: {0: 'Não', 1: 'Sim'}. Também define a ordem de exibição.
            legends_cols (int, opcional): Número de colunas na caixa de legenda. Padrão é 2.
            corner (bool, opcional): Se True, plota apenas o triângulo inferior da matriz, 
                reduzindo a redundância. Padrão é True.
            alpha (float, opcional): Nível de transparência dos pontos (0 a 1). Padrão é 0.5.
            **kwargs: Argumentos adicionais repassados diretamente ao `sns.pairplot`.

        Returns:
            None: A função exibe o gráfico e libera a memória do DataFrame temporário.

        Observação:
            - A função utiliza `diag_kind="kde"` por padrão para as diagonais.
            - O layout é ajustado automaticamente com `tight_layout` e um ajuste no topo 
            para acomodar a legenda customizada.
    """


    colunas_filtro = list(set(colunas_analise) | {coluna_hue}) if coluna_hue else colunas_analise
    colunas_filtro.sort()

    if legends_labels and coluna_hue:
        df_temp = dataframe[colunas_filtro].copy() # é mais eficiente copiar só as colunas necessárias
        df_temp[coluna_hue] = df_temp[coluna_hue].map(legends_labels)

        hue_order = list(legends_labels.values()) # força a manter a ordem do dicionário
    else:
        df_temp = dataframe[colunas_filtro]
        hue_order = None

    p = sns.pairplot(
        df_temp,
        
        hue=coluna_hue,
        hue_order=hue_order,

        diag_kind="kde",
        plot_kws={"alpha":alpha},

        corner=corner,

        **kwargs
    )

    if legend_on:
        p._legend.remove() 
        p.add_legend(
            title=legend_title, 
            ncol=legends_cols, loc='upper center', 
            bbox_to_anchor=(0.5, 1.02),
            fontsize=12, 
        )
        p._legend.get_title().set_fontsize(14)
        p._legend._legend_box.align = "center"

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)

    plt.show()

    del df_temp


def graficos_elbow_silhouette(X, random_state=42, invervalo_de_k=(2, 11)):

    """
        Executa a análise de otimização de K para agrupamento usando os métodos Elbow e Silhouette.

        A função treina múltiplos modelos KMeans para diferentes valores de K e gera dois
        gráficos comparativos para auxiliar na escolha do número ideal de clusters:
        1. Elbow Method (Inércia): Busca o "ponto de cotovelo" onde a queda da variância diminui.
        2. Silhouette Method: Mede a qualidade do agrupamento (quão similar um objeto é ao 
        seu próprio cluster em comparação a outros).

        Args:
            X (array-like ou pd.DataFrame): Dados de entrada (características) para o clustering.
            random_state (int, opcional): Semente para garantir que os resultados do KMeans 
                sejam reprodutíveis. Padrão é 42.
            invervalo_de_k (tuple, opcional): Tupla (mínimo, máximo) definindo o intervalo de 
                K a ser testado. Padrão é (2, 11).

        Returns:
            None: A função exibe o painel de gráficos via plt.show().

        Observações esperadas:
            - O método Elbow foca na minimização da inércia (soma dos quadrados dentro do cluster).
            - O método Silhouette busca o valor mais próximo de 1.0 (indica clusters bem definidos).
            - Valores de K muito altos podem causar overfitting e perda de interpretabilidade.
        """

    fig, axs = plt.subplots(nrows=1, ncols=2,
                            figsize=(15, 5),
                            tight_layout=True)

    elbow = {}
    silhouette = []


    k_range = range(*invervalo_de_k)

    for i in k_range:

        kmeans = KMeans(n_clusters=i,
                        random_state=random_state,
                        n_init=10)
        kmeans.fit(X)

        elbow[i] = kmeans.inertia_
        labels = kmeans.labels_
        silhouette.append(silhouette_score(X, labels))

    sns.lineplot(x=list(elbow.keys()),
                    y=list(elbow.values()),
                    ax=axs[0])

    axs[0].set_xlabel("K")
    axs[0].set_ylabel("Inertia")
    axs[0].set_title("Elbow Method")


    sns.lineplot(x=list(k_range),
                    y=list(silhouette),
                    ax=axs[1])

    axs[1].set_xlabel("K")
    axs[1].set_ylabel("Silhouette score")
    axs[1].set_title("Silhouette Method")


    plt.show()



def visualizar_clusters(
    dataframe, 
    colunas, 
    centroids, 
    mapeamento=None, 
    coluna_clusters='cluster',
    mostrar_centroids=True, 
    mostrar_pontos=True,
    fig_title="Segmentação de Clientes: Personas Identificadas",
    leg_title="Perfil de clientes"
):
    """
    Gera um gráfico de dispersão 2D para visualização de clusters e centroides.

    Args:
        dataframe (pd.DataFrame): Base de dados contendo as coordenadas e os rótulos.
        colunas (list): Lista com os nomes das duas colunas (X e Y) para plotagem.
        centroids (np.ndarray): Array ou lista com as coordenadas dos centroides.
        mapeamento (dict, optional): Dicionário {ID: "Nome"} para renomear os rótulos na legenda.
        coluna_clusters (str): Nome da coluna que contém os IDs numéricos dos clusters.
        mostrar_centroids (bool): Se True, destaca o centro de cada grupo com um círculo e ID.
        mostrar_pontos (bool): Se True, desenha os pontos individuais do dataset.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Definição da paleta de cores baseada na quantidade de clusters únicos
    ids_unicos = sorted(dataframe[coluna_clusters].unique())
    n_clusters = len(ids_unicos)
    paleta = plt.cm.tab10.colors[:n_clusters]

    if mostrar_pontos:
        for i, cluster_id in enumerate(ids_unicos):
            # Filtra os dados do cluster atual
            subset = dataframe[dataframe[coluna_clusters] == cluster_id]
            # Busca o nome no mapeamento; se não houver, usa o ID
            label_nome = mapeamento.get(cluster_id, f"Cluster {cluster_id}") if mapeamento else cluster_id
            
            ax.scatter(
                subset[colunas[0]], 
                subset[colunas[1]], 
                color=paleta[i], 
                label=label_nome, 
                edgecolors='w', 
                linewidth=0.5
            )

    if mostrar_centroids:
        for i, c in enumerate(centroids):
            # Desenha o círculo do centroide
            ax.scatter(*c, s=500, alpha=0.5, color=paleta[i] if i < len(paleta) else 'gray')
            # Adiciona o número do ID no centro
            ax.text(*c, f'{i}', fontsize=15, weight='bold', 
                    horizontalalignment='center', verticalalignment='center')

    # Ajustes estéticos
    ax.set_xlabel(colunas[0])
    ax.set_ylabel(colunas[1])
    ax.set_title(
        fig_title, 
        x=0.5, y=1.11,
        weight='bold'
    )
    
    # Posiciona a legenda fora do gráfico para não obstruir os dados
    ax.legend(
        title=leg_title, 
        bbox_to_anchor=(0.5, 1.1), 
        ncols=3,
        loc='upper center'
    )
    
    plt.tight_layout()
    plt.show()


def plot_col_percent(
        dataframe,
        colunas,
        col_clusters='clusters',

        rows_cols=(2, 3), 
        figsize=(20, 10),

        fig_title="Proporção de dados em cada Cluster",

        palette='tab10'
):
    
    """
     Args:
        dataframe (pd.DataFrame): O conjunto de dados contendo os clusters e as variáveis.
        colunas (list): Lista de strings com os nomes das colunas categóricas a serem analisadas.
        col_clusters (str, optional): Nome da coluna que identifica os clusters. Padrão é 'clusters'.
        rows_cols (tuple, optional): Formato da grade (linhas, colunas). Padrão é (2, 3).
        figsize (tuple, optional): Dimensões da figura (largura, altura). Padrão é (20, 10).
        fig_title (str, optional): Título principal da imagem. Padrão é "Proporção de dados em cada Cluster".
        palette (str, optional): Paleta de cores do Seaborn para as categorias. Padrão é 'tab10'.

    Returns:
        None: Exibe o gráfico utilizando plt.show().
    """

    fig, axs = plt.subplots(ncols=rows_cols[1], nrows=rows_cols[0],
                            figsize=figsize,
                            sharey=True)


    if not isinstance(axs, np.ndarray):
        axs = np.array(axs)

    for ax, c in zip(axs.flatten(), colunas):

        h = sns.histplot(
            data=dataframe,
            x='clusters',
            hue=c,

            multiple='fill',
            stat="percent",
            discrete=True,
            shrink=0.8,

            palette=palette,

            ax=ax,
        )

        h.label_outer()
        h.set_xticks(dataframe[col_clusters].unique())
        h.yaxis.set_major_formatter(PercentFormatter(1))

        for bar in h.containers:
            h.bar_label(
                bar, 
                label_type="center", 
                labels=[f"{b.get_height():.1%}" for b in bar],
                color="white", weight="bold", fontsize=12
            )


        legend = h.get_legend()
        labels = [text.get_text() for text in legend.get_texts()] 
        legend.remove()

        h.legend(
            handles=legend.legend_handles, 
            labels=labels,
            ncol=3,
            loc='upper center', 
            title=c, 
            bbox_to_anchor=(0.5, 1.3),
            fontsize=12
        )

        # for p in h.patches:
        #     p.set_linewidth(0)
        # h.set_facecolor('none')

    fig.suptitle(
        fig_title,
        fontsize=16, weight="bold", 
        y=1.02
    )
    plt.subplots_adjust(
        hspace=0.4,
        wspace=0.15
    )

    plt.show()



def plot_col_clusters_percent(
        dataframe,
        colunas,
        col_clusters='clusters',

        rows_cols=(2, 3), 
        figsize=(20, 10),

        fig_title="Proporção de cada Cluster por categorias",

        palette='tab10'
):
    """
    Args:
        dataframe (pd.DataFrame): DataFrame contendo os dados e a coluna de clusters.
        colunas (list): Lista de colunas para o eixo X (categorias/variáveis).
        col_clusters (str, optional): Nome da coluna de clusters (usada no 'hue'). Padrão é 'clusters'.
        rows_cols (tuple, optional): Organização dos subplots (linhas, colunas). Padrão é (2, 3).
        figsize (tuple, optional): Tamanho da imagem. Padrão é (20, 10).
        fig_title (str, optional): Título superior do gráfico. Padrão é "Proporção de cada Cluster por categorias".
        palette (str, optional): Esquema de cores para os clusters. Padrão é 'tab10'.

    Returns:
        None: Exibe o gráfico formatado com legendas unificadas no topo.
    """

    fig, axs = plt.subplots(ncols=rows_cols[1], nrows=rows_cols[0],
                            figsize=figsize,
                            sharey=True)


    if not isinstance(axs, np.ndarray):
        axs = np.array(axs)

    for ax, c in zip(axs.flatten(), colunas):

        h = sns.histplot(
            data=dataframe,
            x=c,
            hue=col_clusters,

            multiple='fill',
            stat="percent",
            discrete=True,
            shrink=0.8,

            palette=palette,

            ax=ax,
        )

        if dataframe[c].dtype != "object":
            h.set_xticks(dataframe[c].unique())

        h.yaxis.set_major_formatter(PercentFormatter(1))

        for bar in h.containers:
            h.bar_label(
                bar, 
                label_type="center", 
                labels=[f"{b.get_height():.1%}" for b in bar],
                color="white", weight="bold", fontsize=12
            )


        legend = h.get_legend()
        legend.remove()
        
        labels = [text.get_text() for text in legend.get_texts()] 

    fig.legend(
        handles=legend.legend_handles, 
        labels=labels,
        ncol=3,
        loc='upper center', 
        title=col_clusters, 
        bbox_to_anchor=(0.5, 1),
        fontsize=12
    )

    fig.suptitle(
        fig_title,
        fontsize=16, weight="bold", 
        y=1.02
    )
    plt.subplots_adjust(
        hspace=0.25,
        wspace=0.15
    )
    plt.show()

# Observação técnica para o gráfico abaixo:
## Para a versão 1.8+ do scikit-learn já existe uma função própria para mapear os thresholds.
## Ela se chama confusion_matrix_at_thresholds do sklearn.metrics


def visualizar_threshold_metrics(pipeline, X_test, y_test, n_thresholds=100):
    """
    Calcula e plota TN, FP, FN e TP para diferentes limites de decisão (thresholds).
    
    Args:
        pipeline: Pipeline do scikit-learn já treinado.
        X_test: Dados de teste.
        y_test: Rótulos reais de teste.
        n_thresholds: Quantidade de pontos no eixo X (precisão do gráfico).
    """

    y_score = pipeline.predict_proba(X_test)[:, 1] # Obter probabilidades da classe positiva
    
    thresholds = np.linspace(0, 1, n_thresholds)
    tns, fps, fns, tps = [], [], [], []

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)
        tps.append(tp)


    plt.figure(figsize=(10, 6))
    
    plt.plot(thresholds, tns, label="True Negatives (TNs)", color='tab:blue')
    plt.plot(thresholds, fps, label="False Positives (FPs)", color='tab:orange')
    plt.plot(thresholds, fns, label="False Negatives (FNs)", color='tab:green')
    plt.plot(thresholds, tps, label="True Positives (TPs)", color='tab:red')

    plt.xlabel("Thresholds")
    plt.ylabel("Count")
    plt.title("TNs, FPs, FNs and TPs vs Thresholds")

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), shadow=True, ncol=1)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.show()