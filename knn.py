import numpy as np
from numpy import genfromtxt
import math
from sklearn.model_selection import KFold
import argparse

## ALGORITMO KNN - João Vitor R.##
## COMO EXECUTAR: python knn-iris.py -i iris.csv -k 5 -f 5 ##
## -i Qual arquivo CSV voce dejesa, lembrando que o rótulo deve está na ultima coluna SEMPRE ##
## -k Valor de K vizinhos proximos ##
## -f Quantidade de Folds, divisão entre treino e teste ##


# Carrega o arquivo CSV, pulando o cabeçalho e o tipo de dados é Unicode.
def carregarCSV(arquivo):
    csv = genfromtxt(arquivo, delimiter=',',
                     skip_header=1, dtype='unicode')
    return csv


# Calculamos a Distância Euclidiana de todos os atributos de todos os dados do conjunto de treino
def calculoDistancia(instanciaTeste, treino):
    # Em relacao com todos os atributos da nossa instancia pertencente ao conjunto de teste.
    distancia = 0
    for i in range(len(instanciaTeste)-1):
        distancia += pow((float(instanciaTeste[i]) - float(treino[i])), 2)
    # Retorna a raiz quadrada da distancia calculada.
    return math.sqrt(distancia)


# Calculo de quais são os tipos de vizinhos proximos
def calculoVizinhos(treino, instanciaTeste, k, predicoes, classes):
    distancias, vizinhos, predicao = [], [], [0 for x in range(len(classes))]

    # Para cada instancia do meu conjunto de treino, calcular a distancia e salvar em uma lista
    for i in range(len(treino)):
        distancias.append(
            (calculoDistancia(instanciaTeste, treino[i]), treino[i]))

    # Ordena o conjunto em funcao de encontrar os vizinhos mais proximmos
    distancias = sorted(distancias, key=lambda x: x[0])
    #distancias = distancias.sort()

    for i in range(k):  # Seleciona a quantidade de vizinhos baseado no K definido
        vizinhos.append(distancias[i][1])

    for x in range(len(vizinhos)):  # Contar a quantidade de cada tipo dos vizinhos proximos
        verificacao = vizinhos[x][-1]
        predicao[classes.index(verificacao)] += 1

    # Verificar qual maior quantidade de vizinhos e salvar na lista de predicoes.
    # Pega a posicao do maior numero de vizinhos da lista
    posmaior = predicao.index(max(predicao))
    predicoes.append(classes[posmaior])

# Conta quantos acertos e erros eu obtive e sua porcentagem


def obterPrecisao(teste, predicoes, acertos):
    for x in range(len(teste)):
        if teste[x][-1] in predicoes[x]:
            acertos += 1
    return acertos, (acertos/float(len(teste)))*100.0

# Obtem todos os rótulos possíveis


def obterClasse(data):
    classes = []
    for classe in data:
        if not(classe[-1] in classes):
            classes.append(classe[-1])

    return classes


def main():
    parser = argparse.ArgumentParser('KNN')
    parser.add_argument('-i', '--input-file')
    parser.add_argument('-k', '--k-proximos')
    parser.add_argument('-f', '--folds')
    args = parser.parse_args()

    data = carregarCSV(args.input_file)
    classes = obterClasse(data)

    # K significa numero de vizinhos que serão selecionados, deve-se escolher sempre um numero impar. Nos teste realizados K = 5 obteve bons resultados.
    k = int(args.k_proximos)
    qttotal, qtacertos = 0, 0
    # Divide meu conjunto de dados em conjuntos de treino e teste, primeira informacao é a quantidade de conjuntos
    kfold = KFold(int(args.folds), True, 1)

    print('\n')
    for train, test in kfold.split(data):
        predicoes = []
        acertos = 0
        qttotal += (len(data[test]))
        # Para cada dado do conjunto de teste, realizar os calculos de vizinhos.
        for x in range(len(data[test])):
            calculoVizinhos(data[train], data[test][x], k, predicoes, classes)
        precisao = obterPrecisao(data[test], predicoes, acertos)
        qtacertos += precisao[0]

        print("     Precisão de {0}%\n     Com {1} acertos e {2} erros. \n".format(round(precisao[1], 3), precisao[0], ((
            len(data[test]))-precisao[0])))

    print("\nPrecisão total de {0}%\n".format(
        (qtacertos/qttotal)*100.0))


main()
