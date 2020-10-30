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
        ax = plt.gca()
        line = ax.lines[0]
        result = {'PowerP_x_kWh': line.get_xdata(), 'PowerP_y': line.get_ydata()}
        df = pd.DataFrame(result)
        df.to_csv('plot_powerP.csv')
        # Plot Potência Reativa
        sns.set_theme(style="whitegrid")
        sns.displot(self.power_q, kind='kde')
        plt.xlabel("Total Reactive Power [kWh]")
        plt.ylabel("PDF")
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig("plot_powerQ.png", dpi=199)
        ax = plt.gca()
        line = ax.lines[0]
        result = {'PowerQ_x_kvarh': line.get_xdata(), 'PowerQ_y': line.get_ydata()}
        df = pd.DataFrame(result)
        df.to_csv('plot_powerQ.csv')
        # Plot Perdas
        sns.set_theme(style="whitegrid")
        sns.displot(self.power_q, kind='kde')
        plt.xlabel("Total Losses [kWh]")
        plt.ylabel("PDF")
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig("plot_lossesP.png", dpi=199)
        ax = plt.gca()
        line = ax.lines[0]
        result = {'LossesP_x_kWh': line.get_xdata(), 'LossesP_y': line.get_ydata()}
        df = pd.DataFrame(result)
        df.to_csv('plot_lossesP.csv')
        # Plot reativas Perdas
        sns.set_theme(style="whitegrid")
        sns.displot(self.power_q, kind='kde')
        plt.xlabel("Total Losses [kWh]")
        plt.ylabel("PDF")
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig("plot_lossesQ.png", dpi=199)
        ax = plt.gca()
        line = ax.lines[0]
        result = {'LossesQ_x_kWh': line.get_xdata(), 'LossesQ_y': line.get_ydata()}
        df = pd.DataFrame(result)
        df.to_csv('plot_lossesQ.csv')

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
        ax = plt.gca()
        line, line1, line2 = ax.lines[0], ax.lines[1], ax.lines[2]
        result = {'voltage840_y1': line.get_ydata(), 'voltage840_x1_pu': line.get_xdata(),
                  'voltage840_y2': line1.get_ydata(), 'voltage840_x2_pu': line1.get_xdata(),
                  'voltage840_y3': line2.get_ydata(), 'voltage840_x3_pu': line2.get_xdata()}
        df = pd.DataFrame(result)
        df.to_csv('plot_voltage840.csv')

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
        ax = plt.gca()
        line = ax.lines[0]
        result = {'total_voltage_x_pu': line.get_xdata(), 'total_voltage_y': line.get_ydata()}
        df = pd.DataFrame(result)
        df.to_csv('plot_total_voltage.csv')

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


class Pdfoutputpower:
    def __init__(self, number, monitors_name, matrix_p1, matrix_q1, matrix_p2, matrix_q2, matrix_p3, matrix_q3):
        monitors_name = [item for item in monitors_name if "_power" in item]
        monitors_name = [monitors_name[21], monitors_name[22], monitors_name[29], monitors_name[31]]
        matrix_p1 = [matrix_p1[21][:], matrix_p1[22][:], matrix_p1[29][:], matrix_p1[31][:]]
        matrix_q1 = [matrix_q1[21][:], matrix_q1[22][:], matrix_q1[29][:], matrix_q1[31][:]]
        matrix_p2 = [matrix_p2[21][:], matrix_p2[22][:], matrix_p2[29][:], matrix_p2[31][:]]
        matrix_q2 = [matrix_q2[21][:], matrix_q2[22][:], matrix_q2[29][:], matrix_q2[31][:]]
        matrix_p3 = [matrix_p3[21][:], matrix_p3[22][:], matrix_p3[29][:], matrix_p3[31][:]]
        matrix_q3 = [matrix_q3[21][:], matrix_q3[22][:], matrix_q3[29][:], matrix_q3[31][:]]
        self.number = number
        self.monitors_name = monitors_name
        self.matrix_p1 = matrix_p1
        self.matrix_q1 = matrix_q1
        self.matrix_p2 = matrix_p2
        self.matrix_q2 = matrix_q2
        self.matrix_p3 = matrix_p3
        self.matrix_q3 = matrix_q3

    def get_outputpower_ev_bus(self):
        # Organizando dados
        p1_output = []
        p2_output = []
        p3_output = []
        monitors_output = list()
        for i in range(len(self.monitors_name)):
            for n in range(self.number * 24):
                monitors_output.append(self.monitors_name[i])
            p1_output_aux = self.matrix_p1[i]
            p2_output_aux = self.matrix_p2[i]
            p3_output_aux = self.matrix_p3[i]
            if i == 0:
                p1_output = p1_output_aux
                p2_output = p2_output_aux
                p3_output = p3_output_aux
            else:
                p1_output = np.concatenate((p1_output, p1_output_aux))
                p2_output = np.concatenate((p2_output, p2_output_aux))
                p3_output = np.concatenate((p3_output, p3_output_aux))

        # Fase 1
        data = {'bus': monitors_output, 'value': p1_output}
        # Plot da potência de saída
        sns.set_theme(style="darkgrid")
        sns.displot(data=data, x="value", hue="bus", kind="kde")
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.xlabel('Power Phase 1 [MW]')
        plt.savefig("plot_outputpower_p1.png", dpi=199)
        ax = plt.gca()
        line, line1, line2, line3 = ax.lines[0], ax.lines[1], ax.lines[2], ax.lines[3]
        result = {'outputpower_y1': line.get_ydata(), 'outputpower_x1_pu': line.get_xdata(),
                  'outputpower_y2': line1.get_ydata(), 'outputpower_x2_pu': line1.get_xdata(),
                  'outputpower_y3': line2.get_ydata(), 'outputpower_x3_pu': line2.get_xdata(),
                  'outputpower_y4': line3.get_ydata(), 'outputpower_x4_pu': line3.get_xdata()}
        df = pd.DataFrame(result)
        df.to_csv('plot_outputpower_p1.csv')

        # Fase 2
        data = {'bus': monitors_output, 'value': p2_output}
        # Plot da potência de saída
        sns.set_theme(style="darkgrid")
        sns.displot(data=data, x="value", hue="bus", kind="kde")
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.xlabel('Power Phase 2 [MW]')
        plt.savefig("plot_outputpower_p2.png", dpi=199)
        ax = plt.gca()
        line, line1, line2, line3 = ax.lines[0], ax.lines[1], ax.lines[2], ax.lines[3]
        result = {'outputpower_y1': line.get_ydata(), 'outputpower_x1_pu': line.get_xdata(),
                  'outputpower_y2': line1.get_ydata(), 'outputpower_x2_pu': line1.get_xdata(),
                  'outputpower_y3': line2.get_ydata(), 'outputpower_x3_pu': line2.get_xdata(),
                  'outputpower_y4': line3.get_ydata(), 'outputpower_x4_pu': line3.get_xdata()}
        df = pd.DataFrame(result)
        df.to_csv('plot_outputpower_p2.csv')

        # Fase 3
        data = {'bus': monitors_output, 'value': p3_output}
        # Plot da potência de saída
        sns.set_theme(style="darkgrid")
        sns.displot(data=data, x="value", hue="bus", kind="kde")
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.xlabel('Power Phase 3 [MW]')
        plt.savefig("plot_outputpower_p3.png", dpi=199)
        ax = plt.gca()
        line, line1, line2, line3 = ax.lines[0], ax.lines[1], ax.lines[2], ax.lines[3]
        result = {'outputpower_y1': line.get_ydata(), 'outputpower_x1_pu': line.get_xdata(),
                  'outputpower_y2': line1.get_ydata(), 'outputpower_x2_pu': line1.get_xdata(),
                  'outputpower_y3': line2.get_ydata(), 'outputpower_x3_pu': line2.get_xdata(),
                  'outputpower_y4': line3.get_ydata(), 'outputpower_x4_pu': line3.get_xdata()}
        df = pd.DataFrame(result)
        df.to_csv('plot_outputpower_p3.csv')

    def get_mean_outputpower_ev_bus(self):
        time = np.arange(24)
        for i in range((self.number*3)-1):
            time = np.concatenate((time, np.arange(24)))
        p_type = np.concatenate(((['P1'] * self.number * 24), (['P3'] * self.number * 24), (['P2'] * self.number * 24)))

        for i in range(len(self.monitors_name)):
            bus1_ev = np.zeros((self.number * 24, 1))
            bus2_ev = np.zeros((self.number * 24, 1))
            bus3_ev = np.zeros((self.number * 24, 1))
            num = 0
            for simulation in range(self.number):
                for hour in range(24):
                    bus1_ev[num] = self.matrix_p1[i][num]
                    bus2_ev[num] = self.matrix_p2[i][num]
                    bus3_ev[num] = self.matrix_p3[i][num]
                    num += 1
            bus_ev = np.concatenate(([float(item) for item in bus1_ev],
                                     [float(item) for item in bus2_ev],
                                     [float(item) for item in bus3_ev]))
            bus_ev_dict = {'time': time, 'type': p_type, 'value': bus_ev}
            sns.relplot(x="time", y="value", hue="type", kind="line", data=bus_ev_dict)
            plt.gcf().subplots_adjust(bottom=0.15)
            plt.xlabel('Time [h]')
            plt.ylabel('Output Power [MW]')
            file_str = 'plot_outputpower_' + str(self.monitors_name[i]) + '_ev.png'
            plt.savefig(file_str, dpi=199)
            df = pd.DataFrame(bus_ev_dict)
            file_str = 'plot_outputpower_' + str(self.monitors_name[i]) + '_ev.csv'
            df.to_csv(file_str)

            plt.show()
