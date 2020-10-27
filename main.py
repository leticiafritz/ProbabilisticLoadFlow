#################################################################
# NOME: LETÍCIA FRITZ HENRIQUE
# E-MAIL: LETICIA.HENRIQUE@ENGENHARIA.UFJF.BR
# PROJETO: FLUXO DE CARGA PROBABILISTICO COM RED
# VERSÃO: 2.3
# PROXIMO PASSO: 11. TESTAR SMC; 12. IMPLEMENTAR MÉTODO ANALÍTICO
#################################################################

# BIBLIOTECAS
from montecarlo import Montecarlo


# FUNÇÃO PRINCIPAL
def main():

    # Monte Carlo
    simulation = Montecarlo(20)
    simulation.set_simulation()


# RUN
if __name__ == '__main__':
    main()
