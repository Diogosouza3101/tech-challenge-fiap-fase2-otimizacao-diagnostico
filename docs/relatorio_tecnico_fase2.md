# Relatório Técnico — Tech Challenge Fase 2

## 1. Introdução

Este relatório apresenta a solução desenvolvida para o Tech Challenge da Fase 2 da pós-graduação em Inteligência Artificial para Desenvolvedores.

O projeto escolhido foi o **Projeto 1 — Otimização de Modelos de Diagnóstico**, cujo objetivo é evoluir modelos de diagnóstico médico desenvolvidos anteriormente, utilizando **Algoritmos Genéticos** para otimização de hiperparâmetros e integrando uma **LLM** para interpretação dos resultados em linguagem natural.

A solução foi aplicada ao problema de classificação de câncer de mama, com o objetivo de apoiar a triagem inicial de casos classificados como benignos ou malignos.

> Este projeto possui finalidade acadêmica e não substitui avaliação médica profissional.

---

## 2. Objetivo

O objetivo principal do projeto é desenvolver uma solução capaz de otimizar modelos de Machine Learning aplicados ao diagnóstico médico, utilizando Algoritmos Genéticos para busca de melhores hiperparâmetros e uma LLM para transformar resultados estatísticos em interpretações compreensíveis para médicos e gestores hospitalares.

Os objetivos específicos são:

- Treinar modelos baseline para classificação médica;
- Implementar um Algoritmo Genético para otimização de hiperparâmetros;
- Comparar o desempenho entre modelos originais e otimizados;
- Realizar ao menos três experimentos com diferentes configurações do Algoritmo Genético;
- Aplicar um limiar clínico para priorizar a redução de falsos negativos;
- Integrar uma LLM para interpretação dos resultados;
- Avaliar a qualidade das interpretações geradas;
- Documentar decisões técnicas, limitações e resultados obtidos.

---

## 3. Projeto Escolhido

O projeto escolhido foi o **Projeto 1 — Otimização de Modelos de Diagnóstico**.

Esse projeto foi selecionado por estar diretamente alinhado ao trabalho desenvolvido na Fase 1, que já tratava de diagnóstico médico utilizando Machine Learning. Dessa forma, a Fase 2 foi utilizada para evoluir a solução existente, incorporando otimização por Algoritmos Genéticos e interpretação dos resultados por meio de LLM.

---

## 4. Dataset Utilizado

Foi utilizado o dataset de câncer de mama disponível na biblioteca `scikit-learn`, baseado no Breast Cancer Wisconsin Dataset.

O dataset contém atributos numéricos extraídos de exames relacionados a características celulares, como raio, textura, perímetro, área, suavidade, concavidade e outros indicadores.

A variável alvo foi ajustada para facilitar a interpretação no contexto do projeto:

- `1`: maligno;
- `0`: benigno.

O problema foi tratado como uma tarefa de classificação binária, cujo objetivo é identificar corretamente casos malignos e benignos.

---

## 5. Metodologia

A metodologia foi organizada nas seguintes etapas:

1. Configuração do ambiente no Google Colab;
2. Carregamento do dataset;
3. Ajuste da variável alvo;
4. Separação entre treino e teste;
5. Treinamento dos modelos baseline;
6. Implementação do Algoritmo Genético;
7. Execução de três experimentos obrigatórios;
8. Comparação entre modelo original e modelo otimizado;
9. Ajuste de limiar clínico;
10. Experimento adicional com hotstart e mutação adaptativa;
11. Integração com LLM;
12. Avaliação crítica dos resultados;
13. Salvamento dos arquivos de métricas, histórico e interpretação.

---

## 6. Modelo Baseline

Foram treinados inicialmente dois modelos baseline:

- Random Forest;
- XGBoost.

Esses modelos foram utilizados como referência antes da otimização via Algoritmo Genético.

O Random Forest foi escolhido como modelo principal para otimização por apresentar bom desempenho em dados tabulares, robustez e facilidade de interpretação por meio de métricas como importância das variáveis e matriz de confusão.

### 6.1 Resultado do Random Forest Original

O modelo Random Forest original apresentou os seguintes resultados no conjunto de teste:

| Métrica | Valor |
|---|---:|
| Accuracy | 0.973684 |
| Precision maligno | 1.000000 |
| Recall maligno | 0.928571 |
| F1 maligno | 0.962963 |
| ROC AUC | 0.996528 |
| Especificidade benigno | 1.000000 |
| Verdadeiros negativos | 72 |
| Falsos positivos | 0 |
| Falsos negativos | 3 |
| Verdadeiros positivos | 39 |

Apesar do desempenho elevado, o modelo apresentou **3 falsos negativos**, ou seja, três casos malignos foram classificados como benignos. Em um contexto de triagem médica, esse tipo de erro é crítico, pois pode atrasar a avaliação clínica de casos potencialmente graves.

---

## 7. Algoritmo Genético

O Algoritmo Genético foi implementado para otimizar os hiperparâmetros do modelo Random Forest.

A proposta foi representar cada solução candidata como um indivíduo da população. Cada indivíduo contém uma combinação de hiperparâmetros do modelo. A cada geração, os indivíduos são avaliados por uma função fitness, selecionados, cruzados e mutados para gerar novas populações.

---

### 7.1 Codificação dos Indivíduos

Cada indivíduo foi representado por um dicionário de hiperparâmetros.

Exemplo de indivíduo:

```python
{
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "log2",
    "class_weight": "balanced"
}
```

Os genes utilizados foram:

| Gene | Descrição |
|---|---|
| `n_estimators` | Quantidade de árvores da floresta |
| `max_depth` | Profundidade máxima das árvores |
| `min_samples_split` | Número mínimo de amostras para dividir um nó |
| `min_samples_leaf` | Número mínimo de amostras em uma folha |
| `max_features` | Estratégia de seleção de atributos |
| `class_weight` | Estratégia de balanceamento das classes |

Essa representação foi adequada ao problema porque os hiperparâmetros testados possuem valores discretos ou categóricos.

---

### 7.2 Espaço de Busca

O espaço de busca utilizado foi:

```python
search_space_rf = {
    "n_estimators": [50, 100, 150, 200, 300],
    "max_depth": [None, 3, 5, 8, 10, 15],
    "min_samples_split": [2, 4, 6, 8, 10],
    "min_samples_leaf": [1, 2, 3, 4],
    "max_features": ["sqrt", "log2", None],
    "class_weight": [None, "balanced"]
}
```

Esse espaço foi definido para permitir combinações variadas, sem tornar o custo computacional excessivo para execução no Google Colab.

---

### 7.3 Função Fitness

A função fitness foi construída considerando o contexto médico do problema.

Como falsos negativos são especialmente críticos em triagem médica, o **recall da classe maligna** recebeu o maior peso. Também foram consideradas as métricas F1-score, ROC AUC e especificidade.

A função fitness utilizada foi:

```text
fitness = 0.40 * recall_maligno
        + 0.25 * f1_maligno
        + 0.25 * roc_auc
        + 0.10 * especificidade_benigno
```

A escolha dessa função busca equilibrar:

- Sensibilidade para casos malignos;
- Qualidade geral da classificação;
- Capacidade probabilística do modelo;
- Controle de falsos positivos em casos benignos.

---

### 7.4 Seleção

Foi implementada a **seleção por torneio**.

Nesse método, um conjunto de indivíduos é sorteado aleatoriamente da população e o indivíduo com maior fitness é selecionado como pai. Essa estratégia favorece indivíduos mais aptos, mas ainda permite diversidade na seleção.

---

### 7.5 Crossover

Foi utilizado **crossover uniforme**.

Nesse operador, cada gene do filho pode ser herdado de um dos pais. Essa estratégia é adequada para o problema porque os genes são independentes e representam hiperparâmetros distintos.

Exemplo conceitual:

```text
Pai 1: n_estimators=100, max_depth=5, max_features=sqrt
Pai 2: n_estimators=300, max_depth=None, max_features=log2

Filho: n_estimators=300, max_depth=5, max_features=log2
```

---

### 7.6 Mutação

A mutação foi implementada como alteração aleatória de genes.

Para cada gene, existe uma probabilidade de substituição por outro valor válido dentro do espaço de busca. Essa estratégia permite explorar novas combinações de hiperparâmetros e reduz o risco de convergência prematura.

---

### 7.7 Elitismo

Foi utilizado elitismo para preservar os melhores indivíduos entre as gerações.

A cada geração, os dois melhores indivíduos foram mantidos diretamente na próxima população. Isso evita a perda de soluções promissoras durante as etapas de crossover e mutação.

---

## 8. Experimentos Realizados

Foram realizados três experimentos obrigatórios com diferentes configurações de população, número de gerações, taxa de mutação e taxa de crossover.

| Experimento | População | Gerações | Taxa de Mutação | Taxa de Crossover |
|---|---:|---:|---:|---:|
| GA_RF_01 | 10 | 8 | 0.10 | 0.70 |
| GA_RF_02 | 20 | 12 | 0.20 | 0.80 |
| GA_RF_03 | 30 | 15 | 0.30 | 0.85 |

O melhor experimento entre os três foi:

```text
GA_RF_03_pop30_mut30
```

Hiperparâmetros encontrados:

```python
{
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "log2",
    "class_weight": "balanced"
}
```

---

## 9. Experimento Adicional: Hotstart e Mutação Adaptativa

Além dos três experimentos obrigatórios, foi implementado um experimento adicional com **hotstart** e **mutação adaptativa**.

O hotstart foi utilizado para inicializar parte da população com soluções previamente conhecidas, incluindo:

- A configuração do modelo baseline;
- A melhor configuração encontrada nos experimentos anteriores.

A mutação adaptativa foi utilizada para aumentar a taxa de mutação quando o algoritmo apresentava estagnação, ou seja, várias gerações sem melhoria no fitness.

Esse experimento permitiu monitorar melhor o processo evolutivo, incluindo:

- Fitness médio;
- Desvio padrão do fitness;
- Quantidade de indivíduos únicos;
- Entropia genética;
- Taxa de mutação por geração;
- Tempo de execução por geração;
- Gerações sem melhoria.

Apesar de não alterar o resultado final no conjunto de teste, esse experimento ampliou a análise de diversidade e convergência do Algoritmo Genético.

---

## 10. Resultados

### 10.1 Comparativo Geral

| Modelo | Accuracy | Precision Maligno | Recall Maligno | F1 Maligno | ROC AUC | Especificidade | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest Original | 0.973684 | 1.000 | 0.928571 | 0.962963 | 0.996528 | 1.000000 | 72 | 0 | 3 | 39 |
| Random Forest Otimizado GA | 0.973684 | 1.000 | 0.928571 | 0.962963 | 0.997024 | 1.000000 | 72 | 0 | 3 | 39 |
| RF Otimizado GA + Limiar Clínico | 0.947368 | 0.875 | 1.000000 | 0.933333 | 0.997024 | 0.916667 | 66 | 6 | 0 | 42 |
| RF GA Hotstart Adaptativo | 0.973684 | 1.000 | 0.928571 | 0.962963 | 0.997024 | 1.000000 | 72 | 0 | 3 | 39 |
| RF GA Hotstart + Limiar Clínico | 0.947368 | 0.875 | 1.000000 | 0.933333 | 0.997024 | 0.916667 | 66 | 6 | 0 | 42 |

---

### 10.2 Análise dos Resultados

O modelo Random Forest original já apresentou desempenho elevado, com acurácia de 97,37%, recall maligno de 92,86% e ROC AUC de 0.996528.

Após a otimização via Algoritmo Genético, o modelo manteve a mesma acurácia, o mesmo recall maligno e a mesma matriz de confusão. Entretanto, houve uma pequena melhora no ROC AUC, que passou de 0.996528 para 0.997024.

Isso indica que o Algoritmo Genético encontrou uma configuração com capacidade probabilística ligeiramente superior para separar casos benignos e malignos, embora essa melhoria não tenha alterado as classes previstas com o limiar padrão de 0.50.

---

## 11. Ajuste de Limiar Clínico

O modelo original utiliza, por padrão, um limiar de decisão de 0.50. Ou seja, casos com probabilidade maior ou igual a 0.50 são classificados como malignos.

Foi realizada uma análise de diferentes limiares de decisão, variando de 0.10 a 0.90. O objetivo foi avaliar se a redução do limiar poderia aumentar o recall da classe maligna.

O melhor limiar clínico encontrado foi:

```text
0.20
```

Com esse limiar, o modelo atingiu:

| Métrica | Valor |
|---|---:|
| Accuracy | 0.947368 |
| Precision maligno | 0.875000 |
| Recall maligno | 1.000000 |
| F1 maligno | 0.933333 |
| ROC AUC | 0.997024 |
| Especificidade benigno | 0.916667 |
| Falsos negativos | 0 |
| Falsos positivos | 6 |

A principal vantagem foi a eliminação dos falsos negativos. Em um contexto médico, essa decisão é defensável porque é preferível encaminhar alguns casos benignos para avaliação adicional do que deixar casos malignos sem alerta inicial.

A contrapartida foi o aumento dos falsos positivos, que passaram de 0 para 6, reduzindo a especificidade para 91,67%.

---

## 12. Integração com LLM

Foi integrada uma LLM via Groq/LLaMA para gerar interpretação dos resultados em linguagem natural.

O objetivo foi transformar métricas estatísticas em uma explicação compreensível para médicos e gestores hospitalares.

A LLM recebeu como entrada:

- Métricas dos modelos;
- Matrizes de confusão;
- Comparativo entre modelo original e otimizado;
- Resultado com limiar clínico;
- Instruções de segurança para contexto médico.

---

### 12.1 Prompt Engineering

O prompt foi estruturado para orientar a LLM a:

- Responder em português do Brasil;
- Usar linguagem técnica, mas compreensível;
- Não afirmar que o modelo substitui o médico;
- Não afirmar que o modelo gera diagnóstico definitivo;
- Explicar o trade-off entre recall e especificidade;
- Destacar a importância de reduzir falsos negativos;
- Indicar limitações e riscos do modelo;
- Recomendar o uso apenas como apoio à triagem.

Parte da estrutura do prompt utilizada:

```text
Você é um assistente especializado em interpretar resultados de modelos de Machine Learning aplicados à triagem médica.

Contexto:
Foi desenvolvido um modelo de classificação para apoio ao diagnóstico de câncer de mama, classificando os casos como benigno ou maligno.

Instruções obrigatórias:
- Responda em português do Brasil.
- Use linguagem técnica, mas compreensível para médicos e gestores hospitalares.
- Não afirme que o modelo substitui o médico.
- Não afirme que o modelo gera diagnóstico definitivo.
- Explique que o modelo serve como apoio à triagem e priorização de análise clínica.
- Destaque a importância de reduzir falsos negativos em contexto médico.
```

O prompt completo está disponível em:

```text
reports/prompt_interpretacao_llm.txt
```

---

### 12.2 Resposta Gerada

A resposta original da LLM foi salva em:

```text
reports/interpretacao_resultados_llm_original.txt
```

A resposta apresentou uma estrutura adequada, com resumo executivo, comparação entre modelos, interpretação do limiar clínico, riscos, limitações e recomendação de uso.

Entretanto, foi necessário revisar tecnicamente um trecho em que a LLM afirmou que as versões otimizadas apresentaram acurácias ligeiramente superiores. Essa afirmação foi ajustada, pois os resultados mostram que a acurácia permaneceu igual entre o modelo original e o modelo otimizado com Algoritmo Genético.

A versão revisada foi salva em:

```text
reports/interpretacao_resultados_llm_revisada.txt
```

---

### 12.3 Avaliação da Qualidade da Interpretação

A qualidade da interpretação gerada pela LLM foi avaliada de forma qualitativa com base nos seguintes critérios:

| Critério | Avaliação |
|---|---|
| Clareza | Adequada |
| Coerência técnica | Adequada após revisão |
| Segurança médica | Adequada após reforço de que o modelo não substitui médicos |
| Explicação do trade-off | Adequada |
| Uso dos dados fornecidos | Adequado |
| Risco de alucinação | Mitigado com prompt restritivo e revisão humana |

A avaliação indicou que a LLM foi útil para transformar métricas estatísticas em uma explicação compreensível. No entanto, a revisão humana permaneceu necessária para corrigir imprecisões e garantir segurança no contexto médico.

---

## 13. Monitoramento e Logging

Durante os experimentos, foram gerados arquivos de monitoramento em formato CSV.

Os principais arquivos gerados foram:

| Arquivo | Descrição |
|---|---|
| `comparativo_experimentos_ga_rf.csv` | Comparativo dos experimentos com Algoritmo Genético |
| `historico_geracoes_ga_rf.csv` | Histórico das gerações dos três experimentos obrigatórios |
| `comparativo_final_rf_ga_hotstart_limiar.csv` | Comparativo final entre baseline, GA, hotstart e limiar clínico |
| `historico_ga_rf_04_hotstart_adaptativo.csv` | Histórico do experimento com hotstart e mutação adaptativa |

O monitoramento incluiu:

- Melhor fitness por geração;
- Melhor fitness global;
- Fitness médio;
- Desvio padrão do fitness;
- Tempo de execução;
- Recall maligno;
- F1-score;
- ROC AUC;
- Especificidade;
- Falsos positivos;
- Falsos negativos;
- Entropia genética;
- Quantidade de indivíduos únicos.

---

## 14. Arquitetura da Solução

A arquitetura lógica da solução pode ser representada da seguinte forma:

```text
Dataset de Câncer de Mama
        ↓
Carregamento e Preparação dos Dados
        ↓
Separação Treino/Teste
        ↓
Modelos Baseline
        ↓
Algoritmo Genético
        ↓
Otimização dos Hiperparâmetros
        ↓
Modelo Random Forest Otimizado
        ↓
Avaliação com Métricas
        ↓
Ajuste de Limiar Clínico
        ↓
Resumo Técnico dos Resultados
        ↓
LLM para Interpretação em Linguagem Natural
        ↓
Relatórios, Evidências e Apoio à Triagem
```

A solução foi implementada em Python e executada no Google Colab. Os resultados foram salvos em arquivos CSV e TXT para facilitar rastreabilidade, documentação e análise posterior.

Não foi implementada API nesta versão. A solução foi demonstrada por meio de notebook.

---

## 15. Desafios e Soluções Implementadas

### 15.1 Desafio: Baseline com desempenho muito alto

O modelo Random Forest original já apresentou desempenho elevado. Isso dificultou obter ganhos expressivos após a otimização por Algoritmo Genético.

**Solução implementada:**  
Foi feita uma análise mais crítica das métricas, observando não apenas acurácia, mas também recall maligno, falsos negativos, especificidade e ROC AUC. Essa análise permitiu identificar que o principal ponto de melhoria estava na redução dos falsos negativos.

---

### 15.2 Desafio: Redução de falsos negativos

O modelo original apresentou 3 falsos negativos.

**Solução implementada:**  
Foi avaliado o impacto de diferentes limiares de decisão. O limiar clínico de 0.20 eliminou os falsos negativos e elevou o recall maligno para 100%.

---

### 15.3 Desafio: Monitorar a evolução do Algoritmo Genético

Apenas acompanhar o melhor fitness por geração poderia ser insuficiente para avaliar convergência e diversidade.

**Solução implementada:**  
Foi criado um experimento adicional com hotstart e mutação adaptativa, registrando métricas como fitness médio, desvio padrão, indivíduos únicos, entropia genética e taxa de mutação por geração.

---

### 15.4 Desafio: Segurança na interpretação por LLM

A LLM poderia gerar interpretações excessivamente afirmativas ou clinicamente inadequadas.

**Solução implementada:**  
Foi criado um prompt restritivo, com instruções para não apresentar o modelo como diagnóstico definitivo e reforçar o papel da avaliação médica. A resposta também foi revisada manualmente.

---

## 16. Limitações

A solução apresenta algumas limitações importantes:

- O dataset utilizado é público e controlado;
- O modelo não foi validado em bases externas;
- O modelo não considera histórico clínico, exames complementares ou laudos médicos;
- O ajuste de limiar reduz falsos negativos, mas aumenta falsos positivos;
- A LLM pode gerar imprecisões, exigindo revisão humana;
- A solução não deve ser usada como diagnóstico definitivo;
- Não foi implementado deploy em nuvem nesta versão;
- Não foi implementada API de inferência.

---

## 17. Conclusão

O projeto demonstrou a aplicação de Algoritmos Genéticos para otimização de hiperparâmetros em um modelo de diagnóstico médico.

O modelo Random Forest original já apresentava desempenho elevado. A otimização por Algoritmo Genético manteve a acurácia, recall e matriz de confusão do modelo original, mas apresentou leve ganho em ROC AUC.

A principal melhoria prática foi obtida com o ajuste do limiar clínico para 0.20. Essa estratégia elevou o recall da classe maligna para 100% e eliminou falsos negativos no conjunto de teste, com o custo de aumento dos falsos positivos.

A integração com LLM permitiu transformar métricas técnicas em uma interpretação textual mais acessível para médicos e gestores hospitalares. A resposta gerada foi revisada para garantir coerência técnica e segurança no contexto médico.

Dessa forma, a solução atende ao objetivo do Projeto 1, combinando otimização via Algoritmos Genéticos e interpretação com LLM, mantendo o uso do modelo como ferramenta de apoio à triagem médica.

---

## 18. Referências

- FIAP / Pós Tech. Documento do Tech Challenge — Fase 2.
- Scikit-learn. Breast Cancer Wisconsin Dataset.
- Scikit-learn. Documentação do RandomForestClassifier.
- XGBoost. Documentação oficial.
- LangChain. Documentação oficial.
- Groq. Documentação oficial.
- Materiais das aulas de Algoritmos Genéticos e Desenvolvimento de ML na Cloud.
