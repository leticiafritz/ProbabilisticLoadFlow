#################################################################
# NOME: LETÍCIA FRITZ HENRIQUE
# E-MAIL: LETICIA.HENRIQUE@ENGENHARIA.UFJF.BR
# PROJETO: PLANEJAMENTO DA INCLUSÃO DE ESTAÇÃO DE RECARGA EM UMA
#          MICRORREDE COM RED ATRAVÉS DA PROJEÇÃO DA CARGA, PREÇO
#          TARIFÁRIO E QUANTIDADE DE EV EM MICRORREDE UTILIZANDO
#          SARIMA PARA PREVISÃO E O FLUXO DE CARGA PROBABILISTICO
#          PARA CÁLCULO TÉCNICO
# VERSÃO: 3.1
# PROXIMO PASSO: 18. PROJEÇÃO DO AUMENTO DE EV
#                19. PROJEÇÃO DO AUMENTO DA CARGA
#                20. SIMULAR CASO COM V2G
#                21. SIMULAR CASO COM RESPOSTA A DEMANDA
#                22. BLOCKCHAIN
#################################################################

# BIBLIOTECAS
from montecarlo import Montecarlo


# FUNÇÃO PRINCIPAL
def main():

    # Dados de Entrada
    print("FLUXO DE CARGA PROBABILISTICO PARA REDES COM RECURSOS ENERGÉTICOS DISTRIBUÍDOS")
    print("-----")
    simulation_number = input("NÚMERO DE SIMULAÇÕES DE MONTE CARLO: ")
    print("-----")
    print("Conheça o modo de correlação entre os REDs:")
    print("[0] sem correlação")
    print("[1] correlação LOAD-LOAD")
    print("[2] correlação LOAD-PV")
    print("[3] correlação LOAD-EV")
    print("[4] correlação LOAD-PV-EV")
    simulation_mode = input("MODO DE SIMULAÇÃO DESEJADO: ")
    print("Iniciando simulação, por favor, aguarde.")

    # Simulação de Monte Carlo
    simulation = Montecarlo(simulation_number, simulation_mode)
    simulation.set_simulation()


# RUN
if __name__ == '__main__':
    main()
