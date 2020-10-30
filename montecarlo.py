# ---------- BIBLIOTECAS ----------
import os
import sys

import pandas as pd
import py_dss_interface
import numpy as np

from sample import Sample
from probabilitydistfunc import Pdf
from probabilitydistfunc import Pdfvoltagebus
from probabilitydistfunc import Pdftotalvoltage
from probabilitydistfunc import Pdfoutputpower

# ---------- OBJETOS ----------
dss = py_dss_interface.DSSDLL()

# ---------- VARIAVEIS GLOBAIS ----------
PV_BUS_LIST = ["810", "818", "820", "822", "838", "840", "842", "844", "846", "848", "856", "862", "864", "890"]
PV_BUS_NODE_LIST = [".2", ".1", ".1", ".1", ".2", ".1.2.3", ".1", ".1.2.3", ".2", ".1.2.3", ".2", ".2", ".1",
                    ".1.2.3"]
PV_BUS_PHASE_LIST = [1, 1, 1, 1, 1, 3, 1, 3, 1, 3, 1, 1, 1, 3]
PV_BUS_KV_LIST = [14.376, 14.376, 14.376, 14.376, 14.376, 24.900, 14.376, 24.900, 14.376, 24.900, 14.376, 14.376,
                  14.376, 24.900]
PV_BUS_KW_LIST = [8.0, 17.0, 17.0, 67.5, 14.0, 27.0, 4.5, 405.0, 12.5, 60.0, 2.0, 14.0, 1.0, 450.0]
PV_BUS_KW_LIST = [element * 0.75 for element in PV_BUS_KW_LIST]
EV_BUS_LIST = ["840", "844", "848", "860", "890"]
EV_BUS_CONNECTION_LIST = ["wye", "wye", "delta", "wye", "delta"]
EV_BUS_MODEL_LIST = [5, 2, 1, 1, 5]
EV_BUS_KV_LIST = [24.900, 24.900, 24.900, 24.900, 4.160]
EV_BUS_MAX_NUMBER_LIST = [3, 41, 6, 6, 45]


def get_daily_load(sample):
    # Criar LoadShapes
    load_name = dss.loads_allnames()
    for i in range(len(load_name)):
        dss.text("New LoadShape.LoadShape_" + load_name[i])

    # Configurar LoadShapes
    loadshape_name = dss.loadshapes_allnames()
    for i in range(dss.loadshapes_count() - 1):
        dss.loadshapes_write_name(loadshape_name[i + 1])
        dss.loadshapes_write_npts(24)
        dss.loadshapes_write_name(loadshape_name[i + 1])
        dss.loadshapes_write_hrinterval(1)
        # Gerando amostras
        load = sample.get_load_sample()
        new_load = [float(item) for item in load]
        dss.loadshapes_write_name(loadshape_name[i + 1])
        dss.loadshapes_write_pmult(new_load)
        dss.loadshapes_write_name(loadshape_name[i + 1])
        dss.loadshapes_normalize()

    # Acrescentar loadshape nas cargas
    load_name = dss.loads_allnames()
    for i in range(dss.loads_count()):
        dss.loads_write_name(load_name[i])
        dss.loads_write_daily("LoadShape_" + load_name[i])


def get_daily_pv(sample):
    # Cria loadshape com as curvas de geração PV
    for item in PV_BUS_LIST:
        dss.text("New LoadShape.LoadShape_pv_" + item)

    for item in PV_BUS_LIST:
        dss.loadshapes_write_name("LoadShape_pv_" + item)
        dss.loadshapes_write_npts(24)
        dss.loadshapes_write_name("LoadShape_pv_" + item)
        dss.loadshapes_write_hrinterval(1)
        # Gerando amostras
        pv = sample.get_pv_sample()
        new_pv = [float(item) for item in pv]
        dss.loadshapes_write_name("LoadShape_pv_" + item)
        dss.loadshapes_write_pmult(new_pv)
        dss.loadshapes_write_name("LoadShape_pv_" + item)
        dss.loadshapes_normalize()

    # Criando geradores PV
    for i in range(len(PV_BUS_LIST)):
        dss.text("New Generator.PV_" + PV_BUS_LIST[i] + " phases=" + str(PV_BUS_PHASE_LIST[i]))
        dss.text("~ bus1=" + PV_BUS_LIST[i] + PV_BUS_NODE_LIST[i] + " kv=" + str(PV_BUS_KV_LIST[i]))
        dss.text("~ kW=" + str(PV_BUS_KW_LIST[i]) + " pf=1 model=1 conn=wye")
        dss.text("~ daily=LoadShape_pv_" + PV_BUS_LIST[i])


def get_daily_ev(sample):
    # Cria loadshape com carregamento de EV
    for item in EV_BUS_LIST:
        dss.text("New LoadShape.LoadShape_ev_" + item)

    n = 0
    ev_power = [0] * len(EV_BUS_LIST)
    for item in EV_BUS_LIST:
        dss.loadshapes_write_name("LoadShape_ev_" + item)
        dss.loadshapes_write_npts(24)
        dss.loadshapes_write_name("LoadShape_ev_" + item)
        dss.loadshapes_write_hrinterval(1)
        # Gerando amostras
        ev = sample.get_ev_sample()
        ev_power[n] = ev.power
        n += 1
        dss.loadshapes_write_name("LoadShape_ev_" + item)
        dss.loadshapes_write_pmult(ev.curve)
        dss.loadshapes_normalize()

    # Criando carga PEV
    for i in range(len(EV_BUS_LIST)):
        total_power = np.random.randint(int(EV_BUS_MAX_NUMBER_LIST[i] / 3), EV_BUS_MAX_NUMBER_LIST[i]) * ev_power[i]
        dss.text("New Load.EV_" + EV_BUS_LIST[i] + " bus1=" + EV_BUS_LIST[i] + " phases=3")
        dss.text("~ conn=" + EV_BUS_CONNECTION_LIST[i] + " Model=" + str(EV_BUS_MODEL_LIST[i]))
        dss.text("~ kV=" + str(EV_BUS_KV_LIST[i]) + " kW=" + str(total_power) + " kVAR=10")
        dss.text("~ Status=variable daily=LoadShape_ev_" + EV_BUS_LIST[i])


def get_all_monitors():
    lines_name = dss.lines_allnames()
    for item in lines_name:
        dss.text("New Monitor.M" + item + "_power element=line." + item + " terminal=1 mode=1 ppolar=no")
        dss.text("New Monitor.M" + item + "_voltage element=line." + item + " terminal=1 mode=0")


class Montecarlo:
    def __init__(self, number):
        self.number = number

    def set_simulation(self):
        # Carregando arquivos
        load_det = pd.read_csv(os.path.dirname(sys.argv[0]) + "/curve_load.csv", header=None)

        # Gerando amostras
        sample = Sample(load_det)

        # Gerando vetores para o PDF
        power_p = np.zeros(self.number)
        power_q = np.zeros(self.number)
        losses_p = np.zeros(self.number)
        losses_q = np.zeros(self.number)

        # Criando matrizes de tensão
        vmag1_matrix = np.zeros((24, self.number))
        vmag2_matrix = np.zeros((24, self.number))
        vmag3_matrix = np.zeros((24, self.number))
        vphase1_matrix = np.zeros((24, self.number))
        vphase2_matrix = np.zeros((24, self.number))
        vphase3_matrix = np.zeros((24, self.number))

        # Criando matriz para os medidores
        dss.text("compile {}".format(os.path.dirname(sys.argv[0]) + "/ieee34.dss"))
        monitors_total_vmag1 = np.zeros((len(dss.lines_allnames()), 24 * self.number))
        monitors_total_vmag2 = np.zeros((len(dss.lines_allnames()), 24 * self.number))
        monitors_total_vmag3 = np.zeros((len(dss.lines_allnames()), 24 * self.number))

        # Criando matrizes de potência
        monitors_total_p1 = np.zeros((len(dss.lines_allnames()), 24 * self.number))
        monitors_total_q1 = np.zeros((len(dss.lines_allnames()), 24 * self.number))
        monitors_total_p2 = np.zeros((len(dss.lines_allnames()), 24 * self.number))
        monitors_total_q2 = np.zeros((len(dss.lines_allnames()), 24 * self.number))
        monitors_total_p3 = np.zeros((len(dss.lines_allnames()), 24 * self.number))
        monitors_total_q3 = np.zeros((len(dss.lines_allnames()), 24 * self.number))

        for n in range(self.number):
            # Compilando a rede
            dss.text("compile {}".format(os.path.dirname(sys.argv[0]) + "/ieee34.dss"))
            dss.text("New Energymeter.Feeder Line.L1 1")

            # Colocando curva de carga nas barras
            get_daily_load(sample)

            # Colocando curva de geração nas barras
            get_daily_pv(sample)

            # Colocando curva de PEV
            get_daily_ev(sample)

            # Criando monitores para todas as barras
            get_all_monitors()

            # Solução
            dss.text("Set Mode=daily")
            dss.text("Set stepsize=1h")
            dss.text("Set number=24")
            dss.solution_solve()

            # Recebendo Medidor
            power_p[n], power_q[n] = dss.meters_registervalues()[0], dss.meters_registervalues()[1]
            losses_p[n], losses_q[n] = dss.meters_registervalues()[12], dss.meters_registervalues()[13]

            # Recebendo Monitores Linha 19 (Barra 840)
            dss.monitors_saveall()
            for hour in range(24):
                dss.monitors_write_name("Ml19_voltage")
                vmag1_matrix[hour, n] = dss.monitors_channel(1)[hour]
                dss.monitors_write_name("Ml19_voltage")
                vphase1_matrix[hour, n] = dss.monitors_channel(2)[hour]
                dss.monitors_write_name("Ml19_voltage")
                vmag2_matrix[hour, n] = dss.monitors_channel(3)[hour]
                dss.monitors_write_name("Ml19_voltage")
                vphase2_matrix[hour, n] = dss.monitors_channel(4)[hour]
                dss.monitors_write_name("Ml19_voltage")
                vmag3_matrix[hour, n] = dss.monitors_channel(5)[hour]
                dss.monitors_write_name("Ml19_voltage")
                vphase3_matrix[hour, n] = dss.monitors_channel(6)[hour]

            # Análise global da rede para tensão
            monitors_name = dss.monitors_allnames()
            i = 0
            num = 0
            for item in monitors_name:
                if (i % 2) != 0:
                    for hour in range(24):
                        dss.monitors_write_name(item)
                        monitors_total_vmag1[num, n * 24 + hour] = dss.monitors_channel(1)[hour]
                        dss.monitors_write_name(item)
                        monitors_total_vmag2[num, n * 24 + hour] = dss.monitors_channel(3)[hour]
                        dss.monitors_write_name(item)
                        monitors_total_vmag3[num, n * 24 + hour] = dss.monitors_channel(5)[hour]
                    num += 1
                i += 1

            # Análise global da rede para potências de saída
            monitors_name = dss.monitors_allnames()
            i = 0
            num = 0
            for item in monitors_name:
                if (i % 2) == 0:
                    for hour in range(24):
                        dss.monitors_write_name(item)
                        monitors_total_p1[num, n * 24 + hour] = dss.monitors_channel(1)[hour]
                        dss.monitors_write_name(item)
                        monitors_total_q1[num, n * 24 + hour] = dss.monitors_channel(2)[hour]
                        dss.monitors_write_name(item)
                        monitors_total_p2[num, n * 24 + hour] = dss.monitors_channel(3)[hour]
                        dss.monitors_write_name(item)
                        monitors_total_q2[num, n * 24 + hour] = dss.monitors_channel(4)[hour]
                        dss.monitors_write_name(item)
                        monitors_total_p3[num, n * 24 + hour] = dss.monitors_channel(5)[hour]
                        dss.monitors_write_name(item)
                        monitors_total_q3[num, n * 24 + hour] = dss.monitors_channel(6)[hour]
                    num += 1
                i += 1

        # Construindo pdf das magnitudes totais
        pdf = Pdf(power_p, power_q, losses_p, losses_q)
        pdf.get_pdf_plot()
        pdf.get_pdf_csv()

        # Construindo boxplot das tensões
        allvoltagebus = Pdftotalvoltage(self.number, monitors_name, monitors_total_vmag1,
                                        monitors_total_vmag2, monitors_total_vmag3)
        allvoltagebus.get_voltage_all_bus_plot()
        allvoltagebus.get_voltage_all_bus_csv()

        # Construindo curva média de tensão
        meanvoltage = Pdfvoltagebus(vmag1_matrix, vphase1_matrix, vmag2_matrix, vphase2_matrix,
                                    vmag3_matrix, vphase3_matrix)
        meanvoltage.get_mean_pdf_voltage_csv()
        meanvoltage.get_pdf_voltage_plot()

        # Recebendo PDF das potências de todas as barras
        alloutputpower = Pdfoutputpower(self.number, monitors_name, monitors_total_p1, monitors_total_q1,
                                        monitors_total_p2, monitors_total_q2, monitors_total_p3, monitors_total_q3)
        alloutputpower.get_outputpower_ev_bus()
        alloutputpower.get_mean_outputpower_ev_bus()

