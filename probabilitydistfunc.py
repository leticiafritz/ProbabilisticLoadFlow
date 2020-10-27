import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Pdf:
    def __init__(self, power_p, power_q, losses_p, losses_q):
        self.power_p = power_p
        self.power_q = power_q
        self.losses_p = losses_p
        self.losses_q = losses_q

    def get_pdf_plot(self):
        # Plot Potência Ativa
        sns.set_theme(style="whitegrid")
        sns.displot(self.power_p, kind='kde')
        plt.xlabel("Total Power [kWh]")
        plt.ylabel("PDF")
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig("plot_powerP.png", dpi=199)
        # Plot Potência Reativa
        sns.set_theme(style="whitegrid")
        sns.displot(self.power_q, kind='kde')
        plt.xlabel("Total Reactive Power [kWh]")
        plt.ylabel("PDF")
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig("plot_powerQ.png", dpi=199)
        # Plot Perdas
        sns.set_theme(style="whitegrid")
        sns.displot(self.power_q, kind='kde')
        plt.xlabel("Total Losses [kWh]")
        plt.ylabel("PDF")
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig("plot_lossesP.png", dpi=199)
        # Plot reativas Perdas
        sns.set_theme(style="whitegrid")
        sns.displot(self.power_q, kind='kde')
        plt.xlabel("Total Losses [kWh]")
        plt.ylabel("PDF")
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig("plot_lossesQ.png", dpi=199)

    def get_pdf_csv(self):
        # Criando dicionário
        power_dict = {'Total Power [kWh]': self.power_p,
                      'Total Reactive Power [kvarh]': self.power_q,
                      'Losses Power [kWh]': self.losses_p,
                      'Losses Reactive Power [kvarh]': self.losses_q}
        # Criando dataframe
        df = pd.DataFrame(power_dict)
        # Gerando CSV
        df.to_csv('data_total_results.csv')


class Pdfvoltagebus:
    def __init__(self, vmag1, vphase1, vmag2, vphase2, vmag3, vphase3):
        self.vmag1 = np.mean(vmag1, axis=1) / ((24.900 * 1000) / np.sqrt(3))
        self.vphase1 = np.mean(vphase1, axis=1)
        self.vmag2 = np.mean(vmag2, axis=1) / ((24.900 * 1000) / np.sqrt(3))
        self.vphase2 = np.mean(vphase2, axis=1)
        self.vmag3 = np.mean(vmag3, axis=1) / ((24.900 * 1000) / np.sqrt(3))
        self.vphase3 = np.mean(vphase3, axis=1)

    def get_pdf_voltage_plot(self):
        # Plot Tensão
        sns.set_theme(style="darkgrid")
        data_voltage = np.concatenate((self.vmag1, self.vmag2, self.vmag3))
        ref_voltage = np.concatenate((['V1 '] * len(self.vmag1), ['V2 '] * len(self.vmag2),
                                      ['V3 '] * len(self.vmag3)))
        data = {'voltage': ref_voltage, 'value': data_voltage}
        sns.displot(data=data, x="value", hue="voltage", kind="kde")
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.xlabel(r'$V_{840} [pu]$')
        plt.savefig("plot_voltage840.png", dpi=199)
        plt.show()

    def get_mean_pdf_voltage_csv(self):
        # Criando dicionário
        voltage_dict = {'Vmag1 [pu]': self.vmag1, 'Vphase1 [degree]': self.vphase1,
                        'Vmag2 [pu]': self.vmag2, 'Vphase2 [degree]': self.vphase2,
                        'Vmag3 [pu]': self.vmag3, 'Vphase3 [degree]': self.vphase3}
        # Criando dataframe
        df = pd.DataFrame(voltage_dict)
        # Gerando CSV
        df.to_csv('data_voltage_840.csv')


class Pdftotalvoltage:
    def __init__(self, number, monitors_name, monitors_total_vmag1,
                 monitors_total_vmag2, monitors_total_vmag3):
        self.number = number
        self.monitors_name = monitors_name
        self.total_vmag1 = monitors_total_vmag1
        self.total_vmag2 = monitors_total_vmag2
        self.total_vmag3 = monitors_total_vmag3

    def get_voltage_all_bus_plot(self):
        ref_voltage = np.concatenate((['800'] * (self.number * 24 * 3),
                                      ['814r'] * (self.number * 24 * 3),
                                      ['816'] * (self.number * 24 * 3),
                                      ['834'] * (self.number * 24 * 3),
                                      ['836'] * (self.number * 24 * 3),
                                      ['844'] * (self.number * 24 * 3),
                                      ['846'] * (self.number * 24 * 3),
                                      ['888'] * (self.number * 24 * 3)))
        data_voltage = np.concatenate((self.total_vmag1[0] / ((24.900 * 1000) / np.sqrt(3)),
                                       self.total_vmag2[0] / ((24.900 * 1000) / np.sqrt(3)),
                                       self.total_vmag3[0] / ((24.900 * 1000) / np.sqrt(3)),
                                       self.total_vmag1[6] / ((24.900 * 1000) / np.sqrt(3)),
                                       self.total_vmag2[6] / ((24.900 * 1000) / np.sqrt(3)),
                                       self.total_vmag3[6] / ((24.900 * 1000) / np.sqrt(3)),
                                       self.total_vmag1[8] / ((24.900 * 1000) / np.sqrt(3)),
                                       self.total_vmag2[8] / ((24.900 * 1000) / np.sqrt(3)),
                                       self.total_vmag3[8] / ((24.900 * 1000) / np.sqrt(3)),
                                       self.total_vmag1[17] / ((24.900 * 1000) / np.sqrt(3)),
                                       self.total_vmag2[17] / ((24.900 * 1000) / np.sqrt(3)),
                                       self.total_vmag3[17] / ((24.900 * 1000) / np.sqrt(3)),
                                       self.total_vmag1[19] / ((24.900 * 1000) / np.sqrt(3)),
                                       self.total_vmag2[19] / ((24.900 * 1000) / np.sqrt(3)),
                                       self.total_vmag3[19] / ((24.900 * 1000) / np.sqrt(3)),
                                       self.total_vmag1[21] / ((24.900 * 1000) / np.sqrt(3)),
                                       self.total_vmag2[21] / ((24.900 * 1000) / np.sqrt(3)),
                                       self.total_vmag3[21] / ((24.900 * 1000) / np.sqrt(3)),
                                       self.total_vmag1[22] / ((24.900 * 1000) / np.sqrt(3)),
                                       self.total_vmag2[22] / ((24.900 * 1000) / np.sqrt(3)),
                                       self.total_vmag3[22] / ((24.900 * 1000) / np.sqrt(3)),
                                       self.total_vmag1[31] / ((4.16 * 1000) / np.sqrt(3)),
                                       self.total_vmag2[31] / ((4.16 * 1000) / np.sqrt(3)),
                                       self.total_vmag3[31] / ((4.16 * 1000) / np.sqrt(3))))
        monitors_v1 = {'name': ref_voltage, 'value': data_voltage}
        # Boxplot tensão
        sns.set_theme(style="whitegrid")
        sns.boxplot(x="name", y="value", data=monitors_v1, palette="Set2", width=0.8)
        plt.xlabel("Buses")
        plt.ylabel("Voltage [pu]")
        plt.ylim((0.96, 1.06))
        plt.savefig("plot_total_voltage.png", dpi=199)

    def get_voltage_all_bus_csv(self):
        # Criando dicionário
        voltage1_mean = np.mean(self.total_vmag1, axis=1)
        voltage1_std = np.std(self.total_vmag1, axis=1)
        voltage2_mean = np.mean(self.total_vmag2, axis=1)
        voltage2_std = np.std(self.total_vmag2, axis=1)
        voltage3_mean = np.mean(self.total_vmag3, axis=1)
        voltage3_std = np.std(self.total_vmag3, axis=1)
        monitors = [item for item in self.monitors_name if "_voltage" in item]
        voltage_dict = {'name': monitors, 'Mean_V1 [V]': voltage1_mean,
                        'STD_V1 [V]': voltage1_std, 'Mean_V2 [V]': voltage2_mean,
                        'STD_V2 [V]': voltage2_std, 'Mean_V3 [V]': voltage3_mean,
                        'STD_V3 [V]': voltage3_std}
        # Criando dataframe
        df = pd.DataFrame(voltage_dict)
        # Gerando CSV
        df.to_csv('data_total_voltage.csv')
