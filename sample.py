# ---------- BIBLIOTECAS ----------
import warnings

warnings.filterwarnings('ignore')

import numpy as np
from numpy.random import multivariate_normal
import scipy.stats as ss
import pandas as pd
from copulas.multivariate import GaussianMultivariate

# ---------- VARIAVEIS GLOBAIS ----------
# VARIAVEIS GLOBAIS PARA FDP
SD_LOAD = .05
ALFA_PV, BETA_PV = 4, 2.5
MU_EV, SD_EV = 3.2, 0.88  # 30, 5
MU_EV_HOUR_ARRIVE, SD_EV_HOUR = 20, 2.4  # [Wu, 2013], carro estaciona para carregar às 17:36

# VARIAVEIS GLOBAIS PARA PV
R_CERTAIN_POINT = 150
R_STANDARD_CONDITION = 1000
R_FACTOR = 500
PV_POWER_GENERATION = 250
# VARIAVEIS GLOBAIS PARA EV
EV_BATTERY_CAPACITY = 62  # [kWh] Nissan Leaf
EV_CONSUMPTION_KM = 62 / 350  # [kWh] Nissan Leaf
EV_CONSUMPTION_V2G = 0.5
EV_VOLTAGE = 220  # [V]
EV_CURRENT_CHARGE = 32  # [A]


# FUNÇÃO DE ESTADO DA CARGA DO EV
def get_ev_soc():
    # Distribuição de probabilidade acumulativa da distância percorrida pelo carro
    dist = np.random.lognormal(MU_EV, SD_EV)

    # Estado da carga da bateria mínimo (o SOC não pode ter menos que 20% de carga no momento da chegada)
    # O SOC não deve ser menor que o SOC_min, durante o período de descarga
    soc_min = 0.2 + ((dist * EV_CONSUMPTION_KM) / (2 * EV_BATTERY_CAPACITY))

    # Energia necessária para carregar totalmente a bateria
    soc_hini = 1 - ((dist * EV_CONSUMPTION_KM + EV_CONSUMPTION_V2G) / EV_BATTERY_CAPACITY)

    # Estado de carga da bateria antes de descarregar
    soc_init = np.random.uniform(soc_hini + soc_min, 1)

    return soc_init, soc_min, soc_hini


class Sample:
    def __init__(self, load, mode, pv_connection, ev_connection, ev_max_connection):
        self.load = pd.DataFrame(load)
        self.mode = mode
        self.copula = GaussianMultivariate()
        self.pv_connection = pv_connection
        self.ev_connection = ev_connection
        self.ev_max_connection = ev_max_connection

    def get_load_sample(self, bus, pv_curve, ev_curve):
        load = {}
        if self.mode == 0:
            for n in range(bus):
                load[n] = np.random.normal(self.load.iloc[:, n],
                                           SD_LOAD)  # distribuição normal [Morshed, 2018]  [Unidade: kWh]
        elif self.mode == 1:
            for n in range(bus):
                load_aux = np.random.normal(self.load.iloc[:, n], SD_LOAD)
                load[n] = [float(item) for item in load_aux]
            df = pd.DataFrame.from_dict(load)
            self.copula.fit(df)
            load = self.copula.sample(len(df))
        elif self.mode == 2:
            load_aux = {}
            pv_total_curve = {0: pv_curve[0], 1: pv_curve[1], 2: pv_curve[2], 3: pv_curve[2], 4: pv_curve[3],
                              5: pv_curve[4], 6: pv_curve[5], 7: pv_curve[5], 8: pv_curve[5], 9: pv_curve[6],
                              10: pv_curve[7], 11: pv_curve[7], 12: pv_curve[7], 13: pv_curve[7],
                              14: pv_curve[8], 15: pv_curve[8], 16: pv_curve[8], 17: pv_curve[9],
                              18: pv_curve[9], 19: pv_curve[10], 20: pv_curve[11], 21: pv_curve[12],
                              22: pv_curve[13]}
            pv_number = 0
            for n in range(bus):
                load[n] = np.random.normal(self.load.iloc[:, n], SD_LOAD)
                if n in self.pv_connection:
                    load_aux[0] = [float(item) for item in load[n]]
                    pv_aux = pv_total_curve[pv_number]
                    load_aux[1] = [float(item) for item in pv_aux]
                    df = pd.DataFrame.from_dict(load_aux)
                    self.copula.fit(df)
                    load_sample = self.copula.sample(len(df))
                    load[n] = load_sample[0]
                    pv_number += 1
        elif self.mode == 3:
            load_aux = {}
            ev_total_curve = {0: ev_curve[0], 1: ev_curve[0], 2: ev_curve[0], 3: ev_curve[1], 4: ev_curve[1],
                              5: ev_curve[1], 6: ev_curve[1], 7: ev_curve[2], 8: ev_curve[2], 9: ev_curve[3],
                              10: ev_curve[3], 11: ev_curve[3], 12: ev_curve[3], 13: ev_curve[3],
                              14: ev_curve[3], 15: ev_curve[3], 16: ev_curve[4]}
            ev_number = 0
            for n in range(bus):
                load[n] = np.random.normal(self.load.iloc[:, n], SD_LOAD)
                if n in self.ev_connection:
                    load_aux[0] = [float(item) for item in load[n]]
                    ev_aux = ev_total_curve[ev_number]
                    load_aux[1] = [float(item) for item in ev_aux]
                    df = pd.DataFrame.from_dict(load_aux)
                    self.copula.fit(df)
                    load_sample = self.copula.sample(len(df))
                    load[n] = load_sample[0]
                    ev_number += 1
        elif self.mode == 4:
            load_aux = {}
            pv_total_curve = {0: pv_curve[0], 1: pv_curve[1], 2: pv_curve[2], 3: pv_curve[2], 4: pv_curve[3],
                              5: pv_curve[4], 6: pv_curve[5], 7: pv_curve[5], 8: pv_curve[5], 9: pv_curve[6],
                              10: pv_curve[7], 11: pv_curve[7], 12: pv_curve[7], 13: pv_curve[7],
                              14: pv_curve[8], 15: pv_curve[8], 16: pv_curve[8], 17: pv_curve[9],
                              18: pv_curve[9], 19: pv_curve[10], 20: pv_curve[11], 21: pv_curve[12],
                              22: pv_curve[13]}
            ev_total_curve = {0: ev_curve[0], 1: ev_curve[0], 2: ev_curve[0], 3: ev_curve[1], 4: ev_curve[1],
                              5: ev_curve[1], 6: ev_curve[1], 7: ev_curve[2], 8: ev_curve[2], 9: ev_curve[3],
                              10: ev_curve[3], 11: ev_curve[3], 12: ev_curve[3], 13: ev_curve[3],
                              14: ev_curve[3], 15: ev_curve[3], 16: ev_curve[4]}
            pv_number = 0
            ev_number = 0
            for n in range(bus):
                load[n] = np.random.normal(self.load.iloc[:, n], SD_LOAD)
                if (n in self.pv_connection) and (n in self.ev_connection):
                    load_aux[0] = [float(item) for item in load[n]]
                    pv_aux = pv_total_curve[pv_number]
                    load_aux[1] = [float(item) for item in pv_aux]
                    ev_aux = ev_total_curve[ev_number]
                    load_aux[2] = [float(item) for item in ev_aux]
                    df = pd.DataFrame.from_dict(load_aux)
                    self.copula.fit(df)
                    load_sample = self.copula.sample(len(df))
                    load[n] = load_sample[0]
                    pv_number += 1
                    ev_number += 1
                elif n in self.pv_connection:
                    load_aux[0] = [float(item) for item in load[n]]
                    pv_aux = pv_total_curve[pv_number]
                    load_aux[1] = [float(item) for item in pv_aux]
                    df = pd.DataFrame.from_dict(load_aux)
                    self.copula.fit(df)
                    load_sample = self.copula.sample(len(df))
                    load[n] = load_sample[0]
                    pv_number += 1
                elif n in self.ev_connection:
                    load_aux[0] = [float(item) for item in load[n]]
                    ev_aux = ev_total_curve[ev_number]
                    load_aux[1] = [float(item) for item in ev_aux]
                    df = pd.DataFrame.from_dict(load_aux)
                    self.copula.fit(df)
                    load_sample = self.copula.sample(len(df))
                    load[n] = load_sample[0]
                    ev_number += 1

        return load

    # FUNÇÃO DE POTÊCIA GERADA DA PV
    def get_pv_sample(self, bus):
        pv_sample = {}
        for i in range(bus):
            # Função de distribuição de probabilidade da radiação solar
            radiation = ss.beta.pdf(np.linspace(0, 1, 24), ALFA_PV, BETA_PV, 0, 1)  # beta [Yaotang, 2016]
            radiation = radiation * R_FACTOR
            # Configurando a curva da potência gerada kW
            pv = [0] * 24
            for n in range(np.size(pv)):
                if 0 <= radiation[n] < R_CERTAIN_POINT:
                    pv[n] = PV_POWER_GENERATION * (radiation[n] ** 2 / (R_CERTAIN_POINT * R_STANDARD_CONDITION))
                elif R_CERTAIN_POINT <= radiation[n] < R_STANDARD_CONDITION:
                    pv[n] = PV_POWER_GENERATION * (radiation[n] / R_STANDARD_CONDITION)
                elif radiation[n] >= R_STANDARD_CONDITION:
                    pv[n] = PV_POWER_GENERATION
            pv_sample[i] = pv

        return pv_sample

    # CONFIGURANDO AMOSTRA EV
    def get_ev_sample(self, bus, mode):
        ev_curve = {}
        ev_curve_aux = [0] * 24
        ev_power = {}
        ev_power_aux = 0
        ev_incoming = [0] * bus
        ev_t_duration = []
        for bus_i in range(bus):
            # Quantidade de EV
            ev_incoming[bus_i] = np.random.randint(int(self.ev_max_connection[bus_i] / 3),
                                                   self.ev_max_connection[bus_i])
            for ev_i in range(ev_incoming[bus_i]):
                # SOC do veículo elétrico
                soc_init, soc_min, soc_hini = get_ev_soc()
                # Estimando tempos do carregamento
                t_duration_charge = 0
                while t_duration_charge <= 0:
                    choice = np.random.randint(1, 4)
                    if choice == 1:
                        t_duration_charge = np.random.randint(1, 6)
                    elif choice == 2:
                        t_duration_charge = round(np.random.normal(3, 0.50))
                    else:
                        t_duration_charge = round(np.random.normal(6, 0.75))
                ev_t_duration.append(t_duration_charge)
                t_start_charge = int(np.random.normal(MU_EV_HOUR_ARRIVE, SD_EV_HOUR))
                while t_start_charge > 24:
                    t_start_charge = int(np.random.normal(MU_EV_HOUR_ARRIVE, SD_EV_HOUR))
                # Construindo curva
                curve = [0] * (t_start_charge - 1)
                curve.extend([1] * t_duration_charge)
                if len(curve) < 24:
                    curve.extend([0] * (24 - len(curve)))
                else:
                    curve_aux = curve[24:]
                    n = len(curve_aux)
                    for i in range(n):
                        curve[i] = curve_aux[i]
                ev_curve_aux = ev_curve_aux + np.asarray(curve[0:24])
                # Energia do carro
                energy = (soc_init - soc_hini) * EV_BATTERY_CAPACITY
                ev_power_aux = ev_power_aux + energy / t_duration_charge
            if mode == 1:
                ev_curve_aux = [-item for item in ev_curve_aux]
                charge_time = np.zeros(8)
                discharge_time = np.random.randint(5, 15, 8)
                discharge_curve = np.concatenate((charge_time, discharge_time))
                discharge_curve = np.concatenate((discharge_curve, charge_time))
                ev_curve_aux = ev_curve_aux + discharge_curve
            ev_curve[bus_i] = ev_curve_aux
            ev_power[bus_i] = ev_power_aux

        return ev_curve, ev_power, ev_incoming, ev_t_duration, EV_BATTERY_CAPACITY
