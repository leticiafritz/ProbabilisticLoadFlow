#################################################################
# NOME: LETÍCIA FRITZ HENRIQUE
# E-MAIL: LETICIA.HENRIQUE@ENGENHARIA.UFJF.BR
# PROJETO: PLANEJAMENTO DA INCLUSÃO DE ESTAÇÃO DE RECARGA EM UMA
#          MICRORREDE COM RED ATRAVÉS DA PROJEÇÃO DA CARGA, PREÇO
#          TARIFÁRIO E QUANTIDADE DE EV EM MICRORREDE UTILIZANDO
#          SARIMA PARA PREVISÃO E O FLUXO DE CARGA PROBABILISTICO
#          PARA CÁLCULO TÉCNICO
# VERSÃO: 3.1
# PROXIMO PASSO: 19. PROJEÇÃO DO AUMENTO DA CARGA
#                20. PROJEÇÃO DO AUMENTO DE EV
#                21. SIMULAR CASO COM RESPOSTA A DEMANDA
#                22. BLOCKCHAIN
#################################################################

# BIBLIOTECAS
import sys

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
    print("-----")
    print("Modo de simulação do veículo elétrico:")
    print("[0] sem V2G")
    print("[1] com V2G")
    simulation_ev = input("MODO DE SIMULAÇÃO DO VEÍCULO ELÉTRICO:")

    # Verificando possibilidade de entrada
    if (0 <= int(simulation_mode) <= 4) and (0 <= int(simulation_ev) <= 1):
        print("Iniciando simulação, por favor, aguarde ...")
    else:
        print("Os modos escolhidos não são permitidos. Encerrando processo.")
        sys.exit()

    # Simulação de Monte Carlo
    simulation = Montecarlo(simulation_number, simulation_mode, simulation_ev)
    simulation.set_simulation()


# RUN
if __name__ == '__main__':
    main()
