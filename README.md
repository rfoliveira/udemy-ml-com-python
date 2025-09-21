# udemy-ml-com-python
Curso Machine Learning com Python

## Links
- [Hugging Face - AI Community](https://huggingface.co/)
    - [Datasets](https://huggingface.co/datasets)
- Repositórios de dados recomendados para estudo pelo instrutor
    - [INEP](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados)
    - [Google dataset search](https://datasetsearch.research.google.com)
    - [Portal brasileiro de dados abertos](https://www.dados.gov.br)
    - [Kaggle](https://www.kaggle.com)
    - [UCI machine learn repository](https://archive.ics.uci.edu/ml/index.php)
    - [OMS](https://www.who.int)
    - [Paho - org. panamericana de saúde](https://www.paho.org/en)
    - [DrivenData](https://www.drivendata.org)

## Observações
- instrutor usa o google colab

## Estudos (estatística)

Mediana = média dos pontos centrais ordenados. Ex.: dado um conjunto de 6 itens x = {10,23,32,40,57,57}, mediana ficaria:
Mediana = (32 + 40) / 2
Nesse caso o conjunto possui um número par de elementos, para casos em que o número de elementos é ímpar, por exemplo:
x = {10,23,32,38,40,57,57}, então 38 seria a mediana

Moda = elemento que aparece mais vezes no conjunto

Média ponderada = quando alguns valores possuem peso maior na média do que outros. 
Somatória da multiplicação de cada elemento por seu peso, dividido pela somatória de seus pesos

Média de uma distribuição de frequência
Somatória da multiplicação de cada elemento por sua frequência (qtd repetições), dividido pelo somatório da frequência

Amplitude = diferença entre elemento de máxima e mínimo
Medidas de posição = Fractis => divisão de um conjunto de dados em partes iguais (quartis, percentis, mediana...)

Outliers = dados discrepantes, muito diferentes dos demais dados pertencentes à variável de análise.
A relevância deles deve ser analisada para definir se continuarão no dataset ou se devem ser tratados (corrigidos, excluídos ou substituídos),
pois se náo forem relevantes, podem interferir significativamente nos resultados das análises.
Eles podem ser identificados por observações diretas no dataset (quando a quantidade for pequena), por gráficos e por funções específicas.
O gráfico mais usado para identificar os outliers é o BoxPlot.

---

## Alguns algoritmos de classificação
- Regressão logística
- Naive Baynes
- Árvore de decisão
- Random forest
- KNN
- Máquinas de vetor de suporte
- XGBoost
- LightGBM
- CatBoost
- Redes neurais artificiais

## Datasets usados
- [Previsão de doença cardíaca](https://www.kaggle.com/fedesoriano/heart-failure-prediction/version/1)

## Escalonamento
Tem por objetivo garantir que nenhuma variável tenha mais importância devido à granes diferenças de valores na criação de um modelo. 
Existem 3 tipos:
1. Padronização (StandardScaler): 
    Recomendado quando a variável segue ou se aproxima de uma distribuição normal.
    Centraliza os dados na média 0 e desvio padrão 1, mantendo assim a coerência na distribuição
    $z = \frac{x - u}{s}$ ,onde:<p>z = z_score, 
    x = valor que estou querendo fazer a padronizaçào ou escalonamento, 
    u = média, 
    s = desvio padrão
    </p>
2. Normalização (MinMaxScaler):
    Recomendado quando a variável não segue a distribuição normal. Coloca os dados em um intervalo fixo, geralmente 0 e 1.
    $x_\text{norm} = \frac{x - x_\text{min}}{x_\text{max} - x_\text{min}}$
3. Escalonamento Robusto (RobustScaler):
    Recomendado quando existem muito outliers na variável
    $x_\text{scaled} = \frac{x - \text{mediana}(x)}{IQR(x)}$

### Lógica
#### Nomear o escalonamento
nome = StandardScaler(), MinMaxScaler() ou RobustScaler()

#### Treino
nome.fit(X) = aprende os parâmetros (média, mediana, máximo, ...)

#### Escalonamento
X_esc = nome.transform(X) # transforma os dados

#### Dado novo
X_novo_esc = nome.transform(X_novo) # transforma novos dados com os mesmos parâmetros