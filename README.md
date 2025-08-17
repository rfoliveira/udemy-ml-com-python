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