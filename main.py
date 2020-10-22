#################################################################
# NOME: LETÍCIA FRITZ HENRIQUE
# E-MAIL: LETICIA.HENRIQUE@ENGENHARIA.UFJF.BR
# PROJETO: FLUXO DE CARGA PROBABILISTICO COM RED
# VERSÃO: 2.2
# PROXIMO PASSO: 9. CALCULAR PDF DAS POTÊNCIAS DE SAÍDA; 10. AJUSTAR RESPOSTA DE MONTE CARLO;
#                11. IMPLEMENTAR MÉTODO ANALÍTICO
#################################################################

# BIBLIOTECAS
from montecarlo import Montecarlo


# FUNÇÃO PRINCIPAL
def main():

    # Monte Carlo
    simulation = Montecarlo(1)
    simulation.set_simulation()


# RUN
if __name__ == '__main__':
    main()
