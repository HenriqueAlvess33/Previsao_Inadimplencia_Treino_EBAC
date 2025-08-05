# Importa as bibliotecas necessárias para análise de dados, visualização e modelagem estatística
import pandas as pd  # Manipulação de dados em DataFrames
import numpy as np  # Operações numéricas e arrays
import matplotlib.pyplot as plt  # Visualização de dados (gráficos)
import seaborn as sns  # Visualização de dados estatísticos (gráficos avançados)
from IPython.display import (
    display,
    HTML,
)  # Exibição de tabelas e HTML no Jupyter Notebook
import statsmodels.formula.api as smf  # Modelagem estatística (regressão, etc) usando fórmulas
import statsmodels.api as sm  # Modelagem estatística (funções gerais)
from scipy.stats import t
from statsmodels.graphics.regressionplots import influence_plot
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.tools.tools import add_constant

from sklearn import metrics
from scipy.stats import ks_2samp

# Carrega a base de dados credit_scoring.ftr em um DataFrame
# O arquivo deve estar no caminho especificado. O formato .ftr é eficiente para leitura/escrita de grandes volumes de dados.
df_original = pd.read_feather("./Conjunto de dados/credit_scoring.ftr")

# -------- BLOCO DAS FUNÇÕES -------- #


def atualizar_metadados(df):
    # Cria um DataFrame com os tipos de dados das colunas
    metadados = pd.DataFrame(df.dtypes, columns=["dtype"])

    # Adiciona uma coluna com o número de valores ausentes (missings) por variável
    metadados["n_missings"] = df.isna().sum()

    # Adiciona uma coluna com o número de valores únicos por variável
    metadados["Valores únicos"] = df.nunique()

    # Remove as variáveis 'data_ref' e 'index' do DataFrame de metadados, pois não são explicativas
    metadados = metadados.drop(["data_ref", "index"], axis=0)

    # Retorna o DataFrame de metadados
    return metadados


def poder_de_predicao(df, modelo):
    # Cria uma cópia do DataFrame para evitar alterações no original
    df2 = df.copy()

    # Gera os scores (probabilidades preditas) usando o modelo fornecido
    df2["score"] = modelo.predict(df2)

    # Calcula a acurácia usando um threshold fixo (.068) para classificar como 'mau'
    acc = metrics.accuracy_score(df2.mau, df2.score > 0.068)

    # Calcula a curva ROC e a área sob a curva (AUC)
    fpr, tpr, thresholds = metrics.roc_curve(df2.mau, df2.score)
    auc = metrics.auc(fpr, tpr)

    # Calcula o Gini (2*AUC - 1)
    gini = 2 * auc - 1

    # Calcula o KS (Kolmogorov-Smirnov) entre scores dos maus e não maus
    ks = ks_2samp(
        df2.loc[df2.mau == 1, "score"], df2.loc[df2.mau != 1, "score"]
    ).statistic

    # Exibe as métricas formatadas
    print(
        "Acurácia: {0:.1%} \nauc: {1:.1%} \nGINI: {2:.1%}\nKS: {3:.1%}".format(
            acc, auc, gini, ks
        )
    )


# Função para criar dataframe VIF e realizar a filtragem de variáveis que possuírem um valor acima do estipulado nos parâmetros
def vif_filter(X, limite=10):
    """
    Filtra variáveis com alto fator de inflação de variância (VIF) de um DataFrame.

    Parâmetros:
    X (pd.DataFrame): DataFrame contendo apenas variáveis numéricas independentes.
    limite (float): Valor limite de VIF para remoção de variáveis (padrão=10).

    Retorna:
    X_filtrado (pd.DataFrame): DataFrame com variáveis remanescentes após filtragem.
    removed_features (list): Lista de tuplas (variável, VIF) removidas.
    remaining_vif (pd.DataFrame): DataFrame com VIF das variáveis remanescentes.
    """
    X_filtrado = X.copy()  # Cria uma cópia do DataFrame original
    removed_features = []  # Lista para armazenar variáveis removidas
    vif_scores = {}  # Dicionário para armazenar os VIFs

    while True:
        # Adiciona constante para cálculo do VIF
        X_with_const = add_constant(X_filtrado)

        # Calcula o VIF para cada variável (incluindo a constante)
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_with_const.columns
        vif_data["VIF"] = [
            vif(X_with_const.values, i) for i in range(X_with_const.shape[1])
        ]

        # Remove a constante da análise
        vif_data = vif_data[vif_data["feature"] != "const"]

        # Salva os VIFs calculados
        for _, row in vif_data.iterrows():
            vif_scores[row["feature"]] = row["VIF"]

        # Verifica o maior VIF
        max_vif = vif_data["VIF"].max()
        if max_vif <= limite:
            break  # Sai do loop se todos os VIFs estiverem abaixo do limite

        # Remove a variável com maior VIF
        feature_to_remove = vif_data.loc[vif_data["VIF"].idxmax(), "feature"]
        removed_features.append((feature_to_remove, max_vif))
        X_filtrado = X_filtrado.drop(columns=[feature_to_remove])

        # Se todas as variáveis forem removidas, lança erro
        if X_filtrado.shape[1] == 0:
            raise ValueError(
                "Todas as variáveis foram removidas - limite pode estar muito baixo"
            )

    # Monta DataFrame final com VIF das variáveis remanescentes
    remaining_vif = vif_data[vif_data["feature"].isin(X_filtrado.columns)]
    remaining_vif = remaining_vif.sort_values("VIF", ascending=False)

    return X_filtrado, removed_features, remaining_vif


def proporcao_de_categorias(dataframe, var, comparar=False):
    # Se comparar for False, calcula a frequência e percentual de uma variável em um dataframe
    if comparar == False:
        contagem = pd.crosstab(index=dataframe[var], columns="Frequência")
        contagem.sort_values(by="Frequência", ascending=False, inplace=True)
        contagem["pct_freq"] = (
            contagem["Frequência"] / contagem["Frequência"].sum()
        ) * 100
        contagem.sort_values(by="Frequência", ascending=False, inplace=True)

        # Exibe a tabela formatada no notebook
        display(
            HTML(
                contagem.to_html(index=True, border=1, justify="center", bold_rows=True)
            )
        )
        return contagem

    # Se comparar for True e o dataframe for uma lista, compara a distribuição da variável entre vários dataframes
    if comparar and isinstance(dataframe, list):
        resultados = []
        lista_html = []
        for df in dataframe:
            tab = pd.crosstab(index=df[var], columns="Frequência")
            tab.sort_values(by="Frequência", ascending=False, inplace=True)
            tab["pct_freq"] = (tab["Frequência"] / tab["Frequência"].sum()) * 100
            tab.sort_values(by="Frequência", ascending=False, inplace=True)

            resultados.append(tab)
            lista_html.append(
                tab.to_html(index=True, border=1, justify="center", bold_rows=True)
            )

        # Exibe as tabelas lado a lado para comparação
        return display(HTML("<br><br>".join(lista_html)))


def biv_discreta(var, df):

    if type(var) != list:
        # Cria uma coluna 'bom' como o complemento de 'mau' (1-mau), para facilitar os cálculos
        df["bom"] = 1 - df.mau

        # Agrupa o DataFrame pela variável categórica de interesse
        g = df.groupby(item)

        # Monta um DataFrame com estatísticas bivariadas para cada categoria da variável
        biv = pd.DataFrame(
            {
                "qt_bom": g["bom"].sum(),  # Quantidade de bons em cada categoria
                "qt_mau": g["mau"].sum(),  # Quantidade de maus em cada categoria
                "mau": g["mau"].mean(),  # Taxa de maus em cada categoria
                item: g["mau"].mean().index,  # Nome da categoria
                "cont": g[item].count(),  # Contagem de registros em cada categoria
            }
        )

        # Calcula o erro padrão da taxa de maus para cada categoria
        biv["ep"] = (biv.mau * (1 - biv.mau) / biv.cont) ** 0.5

        # Calcula o intervalo de confiança superior e inferior para a taxa de maus
        biv["mau_sup"] = biv.mau + t.ppf([0.975], biv.cont - 1) * biv.ep
        biv["mau_inf"] = biv.mau + t.ppf([0.025], biv.cont - 1) * biv.ep

        # Calcula o logit (log das odds) da taxa de maus e seus limites
        biv["logit"] = np.log(biv.mau / (1 - biv.mau))
        biv["logit_sup"] = np.log(biv.mau_sup / (1 - biv.mau_sup))
        biv["logit_inf"] = np.log(biv.mau_inf / (1 - biv.mau_inf))

        # Calcula o Weight of Evidence (WOE) para cada categoria
        tx_mau_geral = df.mau.mean()
        woe_geral = np.log(tx_mau_geral / (1 - tx_mau_geral))
        biv["woe"] = biv.logit - woe_geral
        biv["woe_sup"] = biv.logit_sup - woe_geral
        biv["woe_inf"] = biv.logit_inf - woe_geral

        categorias = {cat: i for i, cat in enumerate(biv[item].unique())}
        x_vals = biv[var].map(categorias)

        # Plota o WOE e seus limites para cada categoria
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        ax[0].plot(x_vals, biv.woe, ":bo", label="woe")
        ax[0].plot(x_vals, biv.woe_sup, "o:r", label="limite superior")
        ax[0].plot(x_vals, biv.woe_inf, "o:r", label="limite inferior")

        # Ajusta os limites e rótulos do gráfico
        num_cat = biv.shape[0]
        ax[0].set_xlim([-0.3, num_cat - 0.7])
        ax[0].set_ylabel("Weight of Evidence")
        ax[0].legend(bbox_to_anchor=(0.83, 1.17), ncol=3)
        ax[0].set_xticks(list(categorias.values()))
        ax[0].set_xticklabels(list(categorias.keys()), rotation=15)

        # Plota a contagem de registros por categoria
        ax[1] = biv.cont.plot.bar()

        return biv

    if type(var) == list:
        lista_html = []
        lista = var

        for item in lista:
            # Cria uma coluna 'bom' como o complemento de 'mau' (1-mau), para facilitar os cálculos
            df["bom"] = 1 - df.mau

            # Agrupa o DataFrame pela variável categórica de interesse
            g = df.groupby(item)

            # Monta um DataFrame com estatísticas bivariadas para cada categoria da variável
            biv = pd.DataFrame(
                {
                    "qt_bom": g["bom"].sum(),  # Quantidade de bons em cada categoria
                    "qt_mau": g["mau"].sum(),  # Quantidade de maus em cada categoria
                    "mau": g["mau"].mean(),  # Taxa de maus em cada categoria
                    item: g["mau"].mean().index,  # Nome da categoria
                    "cont": g[item].count(),  # Contagem de registros em cada categoria
                }
            )

            # Calcula o erro padrão da taxa de maus para cada categoria
            biv["ep"] = (biv.mau * (1 - biv.mau) / biv.cont) ** 0.5

            # Calcula o intervalo de confiança superior e inferior para a taxa de maus
            biv["mau_sup"] = biv.mau + t.ppf([0.975], biv.cont - 1) * biv.ep
            biv["mau_inf"] = biv.mau + t.ppf([0.025], biv.cont - 1) * biv.ep

            # Calcula o logit (log das odds) da taxa de maus e seus limites
            biv["logit"] = np.log(biv.mau / (1 - biv.mau))
            biv["logit_sup"] = np.log(biv.mau_sup / (1 - biv.mau_sup))
            biv["logit_inf"] = np.log(biv.mau_inf / (1 - biv.mau_inf))

            # Calcula o Weight of Evidence (WOE) para cada categoria
            tx_mau_geral = df.mau.mean()
            woe_geral = np.log(tx_mau_geral / (1 - tx_mau_geral))
            biv["woe"] = biv.logit - woe_geral
            biv["woe_sup"] = biv.logit_sup - woe_geral
            biv["woe_inf"] = biv.logit_inf - woe_geral

            categorias = {cat: i for i, cat in enumerate(biv[item].unique())}
            x_vals = biv[item].map(categorias)

            # Plota o WOE e seus limites para cada categoria
            fig, ax = plt.subplots(2, 1, figsize=(8, 6))
            ax[0].plot(x_vals, biv.woe, ":bo", label="woe")
            ax[0].plot(x_vals, biv.woe_sup, "o:r", label="limite superior")
            ax[0].plot(x_vals, biv.woe_inf, "o:r", label="limite inferior")

            # Ajusta os limites e rótulos do gráfico
            num_cat = biv.shape[0]
            ax[0].set_xlim([-0.3, num_cat - 0.7])
            ax[0].set_ylabel("Weight of Evidence")
            ax[0].legend(bbox_to_anchor=(0.83, 1.17), ncol=3)
            ax[0].set_xticks(list(categorias.values()))
            ax[0].set_xticklabels(list(categorias.keys()), rotation=15)

            ax[1] = biv.cont.plot.bar()

            plt.legend(loc="best")
            plt.tight_layout()

            lista_html.append(
                biv.to_html(index=True, border=1, justify="center", bold_rows=True)
            )

        return display(HTML("<br><br>".join(lista_html)))


def biv_continua(var, ncat, df):

    if type(var) != list:
        df["bom"] = 1 - df.mau
        cat_srs, bins = pd.qcut(
            df[var], ncat, retbins=True, precision=0, duplicates="drop"
        )
        g = df.groupby(cat_srs)

        biv = pd.DataFrame(
            {
                "qt_bom": g["bom"].sum(),
                "qt_mau": g["mau"].sum(),
                "mau": g["mau"].mean(),
                var: g[var].mean(),
                "cont": g[var].count(),
            }
        )

        biv["ep"] = (biv.mau * (1 - biv.mau) / biv.cont) ** 0.5
        biv["mau_sup"] = biv.mau + t.ppf([0.975], biv.cont - 1) * biv.ep
        biv["mau_inf"] = biv.mau + t.ppf([0.025], biv.cont - 1) * biv.ep

        biv["logit"] = np.log(biv.mau / (1 - biv.mau))
        biv["logit_sup"] = np.log(biv.mau_sup / (1 - biv.mau_sup))
        biv["logit_inf"] = np.log(biv.mau_inf / (1 - biv.mau_inf))

        tx_mau_geral = df.mau.mean()
        woe_geral = np.log(df.mau.mean() / (1 - df.mau.mean()))

        biv["woe"] = biv.logit - woe_geral
        biv["woe_sup"] = biv.logit_sup - woe_geral
        biv["woe_inf"] = biv.logit_inf - woe_geral

        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        ax[0].plot(biv[var], biv.woe, ":bo", label="woe")
        ax[0].plot(biv[var], biv.woe_sup, "o:r", label="limite superior")
        ax[0].plot(biv[var], biv.woe_inf, "o:r", label="limite inferior")

        num_cat = biv.shape[0]

        ax[0].set_ylabel("Weight of Evidence")
        ax[0].legend(bbox_to_anchor=(0.83, 1.17), ncol=3)

        ax[1] = biv.cont.plot.bar()
        return None

    if type(var) == list:
        lista = var
        for item in lista:
            df["bom"] = 1 - df.mau
            cat_srs, bins = pd.qcut(
                df[item], ncat, retbins=True, precision=0, duplicates="drop"
            )
            g = df.groupby(cat_srs)

            biv = pd.DataFrame(
                {
                    "qt_bom": g["bom"].sum(),
                    "qt_mau": g["mau"].sum(),
                    "mau": g["mau"].mean(),
                    item: g[item].mean(),
                    "cont": g[item].count(),
                }
            )

            biv["ep"] = (biv.mau * (1 - biv.mau) / biv.cont) ** 0.5
            biv["mau_sup"] = biv.mau + t.ppf([0.975], biv.cont - 1) * biv.ep
            biv["mau_inf"] = biv.mau + t.ppf([0.025], biv.cont - 1) * biv.ep

            biv["logit"] = np.log(biv.mau / (1 - biv.mau))
            biv["logit_sup"] = np.log(biv.mau_sup / (1 - biv.mau_sup))
            biv["logit_inf"] = np.log(biv.mau_inf / (1 - biv.mau_inf))

            tx_mau_geral = df.mau.mean()
            woe_geral = np.log(df.mau.mean() / (1 - df.mau.mean()))

            biv["woe"] = biv.logit - woe_geral
            biv["woe_sup"] = biv.logit_sup - woe_geral
            biv["woe_inf"] = biv.logit_inf - woe_geral

            fig, ax = plt.subplots(2, 1, figsize=(8, 6))
            ax[0].plot(biv[item], biv.woe, ":bo", label="woe")
            ax[0].plot(biv[item], biv.woe_sup, "o:r", label="limite superior")
            ax[0].plot(biv[item], biv.woe_inf, "o:r", label="limite inferior")

            num_cat = biv.shape[0]

            ax[0].set_ylabel("Weight of Evidence")
            ax[0].legend(bbox_to_anchor=(0.83, 1.17), ncol=3)

            ax[1] = biv.cont.plot.bar()
        return None


def IV(variavel, resposta):
    tab = pd.crosstab(variavel, resposta, margins=True, margins_name="total")

    rótulo_evento = 1
    rótulo_nao_evento = 0

    tab["pct_evento"] = tab[rótulo_evento] / tab.loc["total", rótulo_evento]
    tab["pct_nao_evento"] = tab[rótulo_nao_evento] / tab.loc["total", rótulo_nao_evento]
    tab["woe"] = np.log(tab.pct_evento / tab.pct_nao_evento)
    tab["iv_parcial"] = (tab.pct_evento - tab.pct_nao_evento) * tab.woe
    return tab["iv_parcial"].sum()


def calcula_woe(df, var_bin, target):
    tab = pd.crosstab(df[var_bin], df[target])
    tab.columns = ["bom", "mau"]
    tab["%bom"] = tab["bom"] / tab["bom"].sum()
    tab["%mau"] = tab["mau"] / tab["mau"].sum()
    tab["woe"] = np.log(tab["%bom"] / tab["%mau"])
    return tab[["woe"]]


# -------- ENCERRANDO O BLOCO DAS FUNÇÕES -------- #

"""
## Amostragem
Separe os três últimos meses como safras de validação *out of time* (oot).

Variáveis:
Considere que a variável ```data_ref``` não é uma variável explicativa, é somente uma variável indicadora da safra, e não deve ser utilizada na modelagem. 
A variávei ```index``` é um identificador do cliente, e também não deve ser utilizada como covariável (variável explicativa). 
As restantes podem ser utilizadas para prever a inadimplência, incluindo a renda.

"""
# Seleciona apenas registros do ano de 2015 para compor a base de trabalho (desenvolvimento)
df_trabalho = df_original.loc[df_original["data_ref"].dt.year == 2015]
# Garante que a variável 'mau' está no formato inteiro (0 ou 1)
df_trabalho["mau"] = df_trabalho["mau"].astype("int")

# Seleciona registros do ano de 2016 para compor a base de validação out of time (oot)
df_oot = df_original.loc[df_original["data_ref"].dt.year == 2016]
# Garante que a variável 'mau' está no formato inteiro (0 ou 1) também na base oot
df_oot["mau"] = df_oot["mau"].astype("int")

# Cria coluna com o número do mês referente à data de referência
df_trabalho["Mês"] = df_trabalho["data_ref"].dt.month

# Cria coluna com o nome do mês em português
df_trabalho["Nome_mes"] = df_trabalho["data_ref"].dt.month_name(locale="pt_BR")
# Cria coluna com o nome do dia da semana em português
df_trabalho["Dia_semana"] = df_trabalho["data_ref"].dt.day_name(locale="pt_BR")

# Cria coluna com o número do dia do mês
df_trabalho["Dia do mês"] = df_trabalho["data_ref"].dt.day

# Cria coluna com o ano e trimestre (exemplo: '2015-T1')
df_trabalho["Ano_Trimestre"] = (
    df_trabalho["data_ref"].dt.to_period("Q").apply(lambda x: f"{x.year}-T{x.quarter}")
)

# Cria coluna com o ano
df_trabalho["Ano"] = df_trabalho["data_ref"].dt.year

# Conta o número de ocorrências de cada mês na coluna 'data_ref' do dataframe df_trabalho
# O método .dt.month extrai o mês da coluna de datas
# .value_counts() retorna a contagem de cada mês
# .to_frame() transforma o resultado em um DataFrame para melhor visualização
df_trabalho["data_ref"].dt.month.value_counts().to_frame()


# Lista para armazenar amostragem dos meses e formar um novo dataframe de treino
amostras = []

# Para cada mês presente na coluna 'Mês' do dataframe df_trabalho
for mes in df_trabalho["Mês"].unique().tolist():
    # Seleciona uma amostra aleatória de 2000 linhas para o mês atual
    amostra_mes = df_trabalho.loc[df_trabalho["Mês"] == mes].sample(
        2000, random_state=42
    )
    amostras.append(amostra_mes)

# Junta todas as amostras mensais em um único dataframe de treino
df_treino = pd.concat(amostras)

# Cria um dataframe apenas com os registros que possuem 'tempo_emprego' ausente (missing)
df_missings = df_treino.loc[df_treino["tempo_emprego"].isna()]


# Cria um dataframe de metadados com os tipos, quantidade de missings e valores únicos de cada variável
metadados = pd.DataFrame(df_treino.dtypes, columns=["dtype"])
metadados["n_missings"] = df_treino.isna().sum()
metadados["Valores únicos"] = df_treino.nunique()

# Remove as variáveis 'data_ref' e 'index' do metadados, pois não são explicativas
metadados = metadados.drop(["data_ref", "index"], axis=0)

# Para cada variável categórica (do tipo 'object') do dataframe de treino,
# exceto 'tipo_renda', compara a proporção de categorias entre o conjunto completo (df_treino)
# e o subconjunto com missings em 'tempo_emprego' (df_missings).
for variavel in df_treino.select_dtypes("object").columns.to_list():
    if variavel not in ["tipo_renda"]:
        proporcao_de_categorias([df_treino, df_missings], variavel, comparar=True)

# Compara a proporção de categorias especificamente para 'tipo_renda'
# entre o conjunto completo e o subconjunto de missings.
proporcao_de_categorias([df_treino, df_missings], "tipo_renda", comparar=True)

# df_treino['tempo_emprego'] = df_treino['tempo_emprego'].fillna(-1)
df_treino["tempo_emprego_missing"] = (df_treino["tempo_emprego"].isna() == 1).astype(
    int
)

# Reconstrói o dataset 'metadados' com os valores atualizados
metadados = pd.DataFrame(df_treino.dtypes, columns=["dtype"])
metadados["n_missings"] = df_treino.isna().sum()
metadados["Valores únicos"] = df_treino.nunique()
metadados = metadados.drop(["data_ref", "index"], axis=0)

"""
## Descritiva básica univariada

- Descreva a base quanto ao número de linhas, número de linhas para cada mês em ```data_ref```.
- Faça uma descritiva básica univariada de cada variável. Considere as naturezas diferentes: qualitativas e quantitativas.
"""

# Detalhamento do número de linhas por mês do dataframe
print(
    f"Abaixo um dicionário com o número de linhas para cada mês do dataframe: \n {df_treino['Nome_mes'].value_counts().to_dict()}"
)

# Cria uma figura e um eixo para o gráfico
fig, ax = plt.subplots()

# Plota a contagem de registros para cada categoria de 'tipo_renda', separado por 'sexo'
sns.countplot(x="tipo_renda", data=df_treino, hue="sexo", ax=ax)

# Rotaciona os rótulos do eixo x para melhor visualização
plt.xticks(rotation=45)

# Define o título e os rótulos dos eixos do gráfico
ax.set_title("Distribuição por tipo de renda e sexo")
ax.set_xlabel("Tipo de Renda")
ax.set_ylabel("Quantidade")

# Ajusta o layout para evitar sobreposição de elementos
plt.tight_layout()

# Exibe o gráfico
plt.show()

"""
## Descritiva bivariada

Faça uma análise descritiva bivariada de cada variável
"""

# Cria uma figura e um eixo para o gráfico
fig, ax = plt.subplots()

# Plota um gráfico de barras mostrando a média de 'qtd_filhos' para cada categoria de 'tipo_renda', separado por 'sexo'
sns.barplot(data=df_treino, x="tipo_renda", y="qtd_filhos", hue="sexo")

# Rotaciona os rótulos do eixo x para melhor visualização
plt.xticks(rotation=45)

# Define o título e os rótulos dos eixos do gráfico
ax.set_title("Quantidade de filhos por tipo de renda")
ax.set_xlabel("Tipo de Renda")
ax.set_ylabel("Quantidade de filhos")

# Ajusta o layout para evitar sobreposição de elementos
plt.tight_layout()

# Exibe o gráfico
plt.show()

# Cria uma figura e um eixo para o gráfico
fig, ax = plt.subplots()

# Plota um boxplot demonstrando a distribuição da rendapor quartis de cada sub-categoria da variável "Tipo de renda"
sns.boxplot(data=df_treino, x="tipo_renda", y="renda", hue="sexo")

# Rotaciona os rótulos do eixo x
plt.xticks(rotation=45)

# Define um título para o gráfico e o rótulo dos eixos
ax.set_title("Distribuição de renda por tipo de renda")
ax.set_xlabel("Tipo de Renda")
ax.set_ylabel("Renda")

# Ajusta o layout para ecitar sobrposição dos elementos
plt.tight_layout()

# Exibe o gráfico
plt.show()

"""
## Desenvolvimento do modelo

Desenvolva um modelo de *credit scoring* através de uma regressão logística.

- Trate valores missings e outliers
- Trate 'zeros estruturais'
- Faça agrupamentos de categorias conforme vimos em aula
- Proponha uma equação preditiva para 'mau'
- Caso hajam categorias não significantes, justifique

"""

# Atualiza os valores do dataset "metadados"
metadados = atualizar_metadados(df_treino)

# Uso da função "biv_continua" para verificar a distribuição da variável nos agrupamentos criados dentro da função, e o valor de WOE para cada um dos clusters
biv_continua("tempo_emprego", 20, df_treino)

dict_bins = {}

# Cria uma cópia do dataframe de treino para evitar alterações no original
df_treino_1 = df_treino.copy()

# # Preenche todos os valores ausentes (NaN) do dataframe com -1
# df_treino_1.fillna(-1, inplace=True)

_, cat_tempo_emprego_bins = pd.qcut(
    df_treino_1["tempo_emprego"], q=20, duplicates="drop", precision=0, retbins=True
)


# Cria variáveis categóricas (bins) para variáveis contínuas usando qcut (quantis)
df_treino_1["cat_tempo_emprego"] = pd.cut(
    df_treino_1["tempo_emprego"],
    bins=cat_tempo_emprego_bins,
    precision=0,
    duplicates="drop",
)
df_treino_1["cat_tempo_emprego"] = (
    df_treino_1["cat_tempo_emprego"].cat.add_categories("Missing").fillna("Missing")
)

dict_bins["tempo_emprego"] = cat_tempo_emprego_bins

for col in ["qtd_filhos", "qt_pessoas_residencia", "renda"]:
    # Calcula os bins do treino
    _, bins = pd.qcut(
        df_treino_1[col], 20, retbins=True, precision=0, duplicates="drop"
    )
    cat = pd.cut(df_treino_1[col], bins=bins, precision=0, duplicates="drop")
    df_treino_1[f"cat_{col}"] = cat.cat.add_categories("Missing").fillna("Missing")
    dict_bins[col] = bins


# Remove as colunas originais contínuas e a coluna 'bom' do dataframe, pois agora serão usadas as versões categorizadas
df_treino_1.drop(
    columns=["renda", "qt_pessoas_residencia", "qtd_filhos", "tempo_emprego", "bom"],
    inplace=True,
)
# Atualiza o DataFrame de metadados para df_treino_1, incluindo tipos, missings e valores únicos
metadados_02 = atualizar_metadados(df_treino_1)

iv_dict = {}

# Para cada variável do DataFrame de metadados
for variavel in metadados_02.index.to_list():
    # Exclui variáveis contínuas originais e a target 'mau' do cálculo de IV
    if variavel not in [
        "tempo_emprego",
        "qtd_filhos",
        "qt_pessoas_residencia",
        "renda",
        "mau",
    ]:
        # Calcula o IV (Information Value) da variável para o target 'mau'
        iv_calculado = IV(df_treino_1[variavel], df_treino_1["mau"])
        iv_dict[variavel] = iv_calculado

# Adiciona a coluna 'IV' ao DataFrame de metadados, preenchendo com os valores calculados
metadados_02["IV"] = iv_dict

iv_aprovado = metadados_02.loc[metadados_02["IV"] >= 0.02].index.to_list()

# Monta a fórmula para o modelo logístico, incluindo apenas variáveis aprovadas pelo IV
# Remove variáveis que não devem ser usadas como explicativas (indicadores de tempo, identificadores, etc)
formula = "+".join(
    [
        x
        for x in iv_aprovado
        if x
        not in [
            "mau",
            "Ano",
            "Ano_Trimestre",
            "Dia do Mês",
            "Dia_semana",
            "Nome_mes",
            "Mês",
        ]
    ]
)

# Ajusta o modelo de regressão logística usando a fórmula criada e o dataframe df_treino_1
# O target é 'mau' e as variáveis explicativas são as selecionadas na fórmula
modelo = smf.logit(f"mau ~ {formula}", df_treino_1).fit()

"""
## Avaliação do modelo

Avalie o poder discriminante do modelo pelo menos avaliando acurácia, KS e Gini.

Avalie estas métricas nas bases de desenvolvimento e *out of time*.

"""

# Cria uma cópia da base de teste (out of time) para não alterar o original
df_teste = df_oot.copy()

# Para cada variável contínua, aplica os mesmos bins usados no treino
for col in ["tempo_emprego", "qtd_filhos", "qt_pessoas_residencia", "renda"]:
    # Aplica os bins definidos no treino para categorizar os dados da base de teste
    cat = pd.cut(
        df_teste[col], bins=dict_bins[f"{col}"], precision=0, duplicates="drop"
    )
    df_teste[f"cat_{col}"] = cat
    # Adiciona a categoria "Missing" para valores ausentes
    df_teste[f"cat_{col}"] = cat.cat.add_categories("Missing").fillna("Missing")

# Remove as colunas originais contínuas, pois agora serão usadas as versões categorizadas
df_teste.drop(
    columns=["renda", "qt_pessoas_residencia", "qtd_filhos", "tempo_emprego"],
    inplace=True,
)

# Avalia o desempenho do modelo na base de treino
print("Performance do modelo na base de treino:\n")
poder_de_predicao(df_treino_1, modelo)

# Avalia o desempenho do modelo na base de teste (out of time)
print("\nPerformance do modelo na base de teste:\n")
poder_de_predicao(df_teste, modelo)
