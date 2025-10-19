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
- instrutor usa pickle para salvar as variáveis (atributos), do modelo, que é nativo,
porém o jotlib tem uma performance melhor e é recomendado para o uso em ML

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

[Ref. fórmulas em markdown](https://en.wikibooks.org/wiki/LaTeX/Mathematics)

### Lógica
#### Nomear o escalonamento
nome = StandardScaler(), MinMaxScaler() ou RobustScaler()

#### Treino
nome.fit(X) = aprende os parâmetros (média, mediana, máximo, ...)

#### Escalonamento
X_esc = nome.transform(X) # transforma os dados

#### Dado novo
X_novo_esc = nome.transform(X_novo) # transforma novos dados com os mesmos parâmetros

## Naive Baines
Um dos primeiros algoritmos em Machine Learning
**Classificador** probabilístico baseado na aplicação do teorema de Baines

$P(A|B)=\frac{P(B|A)P(A)}{P(B)}$, onde:

P(A|B) = probabilidade de A ocorrer dado que B ocorreu, é igual a
P(B|A) = probabilidade de B ocorrer dado que A ocorreu, vezes
P(A) = probabilidade de A acontecer, dividido por
P(B) = probabilidade de B acontecer

Também pode ser escrita como:
$P(A|B)P(B)=P(A \cap B)=P(B \cap A)$

Premissa: independência entre as variáveis do problema
Trabalha muito bem variáveis categóricas

**Algumas aplicações**
- Filtros de spam
- Diagnósticos médicos
- Classificação de informações textuais
- Análise de crédito
- Separação de documentos
- Previsão de falhas

**Vantagens**
- Rápido e "fácil" de entendimento
- Pouco esforço computacional
- Bom desempenho com muitos dados
- Boas previsões com poucos dados

**Desvantagens**
- Considera atributos independentes (quando temos atributos que são dependentes, ele acaba por não dar resultados não tão bons)
- Atribuição de valor nulo de probabilidade quando uma classe contida no conjunto de teste não se apreenta no conjunto de treino (não é comum ocorrer quando feito um pré-processamento, mas pode ocorrer) Por exemplo: em uma base de análise de crédito, alvo de "chance da pessoa fazer o pagamento da parcela" e no conjunto de teste náo tem uma das opções de "sim" ou "nào", ficando discrepante entre treino e teste, e nesse caso é atribuído valor nulo.