# Tech Challenge Fase 2 — Otimização de Diagnóstico Médico com Algoritmos Genéticos e LLM

## 1. Descrição do Projeto

Este projeto foi desenvolvido como parte do Tech Challenge da Fase 2 da pós-graduação em Inteligência Artificial para Desenvolvedores.

O projeto escolhido foi o **Projeto 1 — Otimização de Modelos de Diagnóstico**, cujo desafio consiste em evoluir modelos de diagnóstico médico desenvolvidos no Módulo 1, aplicando **Algoritmos Genéticos** para otimização de hiperparâmetros e integrando uma **LLM** para melhorar a interpretação dos resultados para profissionais de saúde.

A solução utiliza um modelo de classificação para apoio à triagem de câncer de mama, classificando os registros como **benignos** ou **malignos**.

> Este projeto possui finalidade acadêmica e não substitui avaliação médica profissional.

---

## 2. Objetivo

Desenvolver uma solução capaz de:

- Treinar modelos baseline de diagnóstico médico;
- Implementar um Algoritmo Genético para otimização de hiperparâmetros;
- Comparar o desempenho entre modelos originais e otimizados;
- Realizar diferentes experimentos com configurações do Algoritmo Genético;
- Aplicar um limiar clínico para reduzir falsos negativos;
- Integrar uma LLM para gerar interpretações em linguagem natural;
- Documentar resultados, decisões técnicas, limitações e próximos passos da solução.

---

## 3. Dataset Utilizado

Foi utilizado o dataset de câncer de mama disponível na biblioteca `scikit-learn`, baseado no Breast Cancer Wisconsin Dataset.

A variável alvo foi ajustada para:

- `1`: maligno;
- `0`: benigno.

O problema foi tratado como uma tarefa de classificação binária para apoio à triagem médica.

---

## 4. Tecnologias Utilizadas

- Python
- Google Colab
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- SHAP
- LangChain
- Groq / LLaMA
- Pytest
- GitHub

---

## 5. Metodologia

A metodologia foi estruturada nas seguintes etapas:

1. Configuração do ambiente;
2. Carregamento e preparação do dataset;
3. Separação entre treino e teste;
4. Treinamento dos modelos baseline;
5. Implementação do Algoritmo Genético;
6. Execução de experimentos com diferentes configurações;
7. Comparação entre modelo original e modelo otimizado;
8. Ajuste de limiar clínico;
9. Experimento adicional com hotstart e mutação adaptativa;
10. Integração com LLM para interpretação dos resultados;
11. Avaliação crítica dos resultados obtidos.

---

## 6. Modelos Baseline

Foram treinados modelos iniciais sem otimização por Algoritmo Genético, com o objetivo de criar uma base de comparação.

Modelos utilizados:

- Random Forest;
- XGBoost.

O Random Forest foi escolhido como principal modelo para otimização por apresentar desempenho robusto e boa aderência ao problema de classificação tabular.

---

## 7. Algoritmo Genético

O Algoritmo Genético foi implementado para otimizar os hiperparâmetros do modelo Random Forest.

Cada indivíduo da população representa uma configuração candidata de hiperparâmetros.

### 7.1 Codificação dos Indivíduos

Cada indivíduo é representado por um dicionário de hiperparâmetros:

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

### 7.2 Genes Utilizados

Os genes considerados foram:

- `n_estimators`;
- `max_depth`;
- `min_samples_split`;
- `min_samples_leaf`;
- `max_features`;
- `class_weight`.

### 7.3 Operadores Genéticos

Foram implementados os seguintes operadores:

- **Seleção por torneio**;
- **Crossover uniforme**;
- **Mutação por alteração aleatória de genes**;
- **Elitismo**, preservando os melhores indivíduos entre gerações.

### 7.4 Função Fitness

A função fitness foi definida considerando o contexto médico, priorizando o recall da classe maligna, pois falsos negativos são críticos em triagem médica.

```text
fitness = 0.40 * recall_maligno
        + 0.25 * f1_maligno
        + 0.25 * roc_auc
        + 0.10 * especificidade_benigno
```

---

## 8. Experimentos Realizados

Foram realizados três experimentos obrigatórios com diferentes configurações do Algoritmo Genético:

| Experimento | População | Gerações | Taxa de Mutação | Taxa de Crossover |
|---|---:|---:|---:|---:|
| GA_RF_01 | 10 | 8 | 0.10 | 0.70 |
| GA_RF_02 | 20 | 12 | 0.20 | 0.80 |
| GA_RF_03 | 30 | 15 | 0.30 | 0.85 |

Também foi realizado um experimento adicional com **hotstart** e **mutação adaptativa**, com o objetivo de ampliar o monitoramento do processo evolutivo e avaliar diversidade, estagnação e convergência.

---

## 9. Resultados Principais

| Modelo | Accuracy | Recall Maligno | Falsos Negativos | Falsos Positivos | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest Original | 97,37% | 92,86% | 3 | 0 | 0.996528 |
| Random Forest Otimizado GA | 97,37% | 92,86% | 3 | 0 | 0.997024 |
| RF Otimizado GA + Limiar Clínico | 94,74% | 100% | 0 | 6 | 0.997024 |
| RF GA Hotstart Adaptativo | 97,37% | 92,86% | 3 | 0 | 0.997024 |
| RF GA Hotstart + Limiar Clínico | 94,74% | 100% | 0 | 6 | 0.997024 |

A otimização via Algoritmo Genético manteve o desempenho geral do modelo original e trouxe leve ganho em ROC AUC.

A principal melhoria prática foi obtida com o ajuste do limiar clínico para **0.20**, que elevou o recall da classe maligna para **100%** e eliminou falsos negativos no conjunto de teste. Como contrapartida, houve aumento de falsos positivos.

---

## 10. Ajuste de Limiar Clínico

O modelo original utiliza limiar padrão de 0.50 para classificar um caso como maligno.

Neste projeto, foi avaliado o impacto de diferentes limiares de decisão. O limiar clínico de **0.20** foi selecionado por reduzir falsos negativos, priorizando sensibilidade para a classe maligna.

Essa escolha é defensável em um cenário de triagem médica, pois falsos negativos podem atrasar a avaliação clínica de casos potencialmente graves.

---

## 11. Integração com LLM

Foi integrada uma LLM via Groq/LLaMA para gerar interpretação dos resultados em linguagem natural.

A LLM recebeu como entrada:

- Métricas dos modelos;
- Matrizes de confusão;
- Comparativo entre modelo original e otimizado;
- Resultado com limiar clínico;
- Restrições de segurança para o contexto médico.

A resposta gerada foi revisada para garantir:

- Coerência técnica;
- Clareza para médicos e gestores;
- Explicação do trade-off entre recall e especificidade;
- Reforço de que o modelo não substitui avaliação médica.

Os arquivos da integração com LLM estão disponíveis em `reports/`:

```text
prompt_interpretacao_llm.txt
interpretacao_resultados_llm_original.txt
interpretacao_resultados_llm_revisada.txt
```

---

## 12. Monitoramento e Logging

Durante a execução dos experimentos, foram gerados arquivos de acompanhamento em CSV, incluindo:

- Histórico das gerações;
- Melhor fitness por geração;
- Fitness médio;
- Desvio padrão do fitness;
- Tempo de execução;
- Métricas de avaliação;
- Entropia genética;
- Quantidade de indivíduos únicos;
- Taxa de mutação adaptativa.

Principais arquivos:

```text
reports/comparativo_experimentos_ga_rf.csv
reports/historico_geracoes_ga_rf.csv
reports/comparativo_final_rf_ga_hotstart_limiar.csv
reports/historico_ga_rf_04_hotstart_adaptativo.csv
```

---

## 13. Arquitetura da Solução

```text
Dataset de Câncer de Mama
        ↓
Pré-processamento e Separação Treino/Teste
        ↓
Modelos Baseline
        ↓
Algoritmo Genético
        ↓
Otimização de Hiperparâmetros
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
Relatórios e Evidências para Apoio à Triagem
```

---

## 14. Estrutura do Repositório

```text
tech-challenge-fiap-fase2-otimizacao-diagnostico/
│
├── docs/
│   ├── relatorio_tecnico_fase2.md
│   └── roteiro_video.md
│
├── notebooks/
│   └── 01_tech_challenge_fase2_otimizacao_diagnostico.ipynb
│
├── reports/
│   ├── comparativo_experimentos_ga_rf.csv
│   ├── historico_geracoes_ga_rf.csv
│   ├── comparativo_final_rf_ga_hotstart_limiar.csv
│   ├── historico_ga_rf_04_hotstart_adaptativo.csv
│   ├── prompt_interpretacao_llm.txt
│   ├── interpretacao_resultados_llm_original.txt
│   └── interpretacao_resultados_llm_revisada.txt
│
├── tests/
│   └── test_genetic_algorithm.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 15. Como Executar o Projeto

### 15.1 Clonar o repositório

```bash
git clone https://github.com/Diogosouza3101/tech-challenge-fiap-fase2-otimizacao-diagnostico.git
cd tech-challenge-fiap-fase2-otimizacao-diagnostico
```

### 15.2 Criar ambiente virtual

```bash
python -m venv venv
```

### 15.3 Ativar ambiente virtual

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### 15.4 Instalar dependências

```bash
pip install -r requirements.txt
```

### 15.5 Executar notebook

Abrir o notebook em Jupyter ou Google Colab:

```text
notebooks/01_tech_challenge_fase2_otimizacao_diagnostico.ipynb
```

### 15.6 Configurar chave da Groq para LLM

Para executar a etapa de LLM, é necessário informar uma chave da Groq no notebook:

```python
os.environ["GROQ_API_KEY"] = getpass("Cole sua GROQ_API_KEY aqui: ")
```

Caso a etapa de LLM não seja executada novamente, os arquivos gerados anteriormente estão disponíveis em `reports/`.

---

## 16. Testes Automatizados

O projeto contém testes automatizados básicos para validar componentes do Algoritmo Genético.

Para executar:

```bash
pytest tests/
```

Os testes validam:

- Criação de indivíduo;
- Validação dos genes;
- Mutação mantendo valores válidos;
- Cálculo de entropia genética.

---

## 17. Relatório Técnico

O relatório técnico está disponível em:

```text
docs/relatorio_tecnico_fase2.md
```

Ele documenta:

- Implementação do Algoritmo Genético;
- Resultados da otimização;
- Comparativo entre modelos;
- Integração com LLM;
- Prompt engineering;
- Avaliação da qualidade da interpretação;
- Desafios e soluções implementadas;
- Limitações da solução.

---

## 18. Vídeo de Demonstração

Link do vídeo:

```text
"Sendo gravação"
```

O vídeo apresenta:

- Demonstração do notebook;
- Explicação do Algoritmo Genético;
- Resultados da otimização;
- Ajuste de limiar clínico;
- Integração com LLM;
- Estrutura do repositório.

---

## 19. Documentação de API

Não foi implementada API nesta versão do projeto.

A solução foi desenvolvida e demonstrada por meio de notebook em Python no Google Colab.

---

## 20. Limitações

- O projeto foi desenvolvido com dataset público e controlado;
- O modelo precisa ser validado com bases externas antes de qualquer uso real;
- A solução não considera histórico clínico, exames complementares ou avaliação médica individual;
- O ajuste de limiar reduz falsos negativos, mas aumenta falsos positivos;
- A LLM foi utilizada para interpretação textual, mas sua resposta foi revisada para reduzir riscos de interpretação inadequada;
- O modelo não deve ser usado como diagnóstico definitivo.

---

## 21. Aviso Médico

Este projeto é uma solução acadêmica de apoio à triagem médica.

O modelo não substitui médicos, laudos clínicos, exames complementares ou protocolos hospitalares. A decisão final deve ser sempre realizada por profissionais de saúde.

---

## 22. Autor

**Diogo Souza**

Pós-graduação em Inteligência Artificial para Desenvolvedores — FIAP / Pós Tech.
