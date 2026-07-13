
import random
import math
from collections import Counter


SEARCH_SPACE_RF = {
    "n_estimators": [50, 100, 150, 200, 300],
    "max_depth": [None, 3, 5, 8, 10, 15],
    "min_samples_split": [2, 4, 6, 8, 10],
    "min_samples_leaf": [1, 2, 3, 4],
    "max_features": ["sqrt", "log2", None],
    "class_weight": [None, "balanced"],
}


def criar_individuo(search_space):
    return {
        gene: random.choice(valores)
        for gene, valores in search_space.items()
    }


def validar_individuo(individuo, search_space):
    for gene, valor in individuo.items():
        if gene not in search_space:
            return False
        if valor not in search_space[gene]:
            return False
    return True


def mutacao(individuo, search_space, taxa_mutacao):
    individuo_mutado = individuo.copy()

    for gene, valores in search_space.items():
        if random.random() < taxa_mutacao:
            individuo_mutado[gene] = random.choice(valores)

    return individuo_mutado


def calcular_entropia_genetica(populacao, search_space):
    entropias = []

    for gene, valores_possiveis in search_space.items():
        alelos = [individuo[gene] for individuo in populacao]
        contagem = Counter(alelos)
        total = len(alelos)

        entropia = 0

        for qtd in contagem.values():
            p = qtd / total
            entropia -= p * math.log2(p)

        max_entropia = math.log2(len(valores_possiveis)) if len(valores_possiveis) > 1 else 1
        entropia_normalizada = entropia / max_entropia if max_entropia > 0 else 0
        entropias.append(entropia_normalizada)

    return sum(entropias) / len(entropias)


def test_criar_individuo_retorna_dicionario():
    individuo = criar_individuo(SEARCH_SPACE_RF)

    assert isinstance(individuo, dict)
    assert set(individuo.keys()) == set(SEARCH_SPACE_RF.keys())


def test_individuo_possui_genes_validos():
    individuo = criar_individuo(SEARCH_SPACE_RF)

    assert validar_individuo(individuo, SEARCH_SPACE_RF) is True


def test_mutacao_mantem_individuo_valido():
    individuo = criar_individuo(SEARCH_SPACE_RF)
    individuo_mutado = mutacao(individuo, SEARCH_SPACE_RF, taxa_mutacao=1.0)

    assert validar_individuo(individuo_mutado, SEARCH_SPACE_RF) is True


def test_entropia_genetica_retorna_valor_valido():
    populacao = [
        criar_individuo(SEARCH_SPACE_RF)
        for _ in range(10)
    ]

    entropia = calcular_entropia_genetica(populacao, SEARCH_SPACE_RF)

    assert entropia >= 0
    assert entropia <= 1
