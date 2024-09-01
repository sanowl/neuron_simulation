# Various ion channel models

import numpy as np

class IonChannel:
    @staticmethod
    def boltzmann(V, V_half, k):
        return 1 / (1 + np.exp((V_half - V) / k))

class SodiumChannel(IonChannel):
    @classmethod
    def alpha_m(cls, V):
        return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))

    @classmethod
    def beta_m(cls, V):
        return 4 * np.exp(-(V + 65) / 18)

    @classmethod
    def alpha_h(cls, V):
        return 0.07 * np.exp(-(V + 65) / 20)

    @classmethod
    def beta_h(cls, V):
        return 1 / (1 + np.exp(-(V + 35) / 10))

    @classmethod
    def m_inf(cls, V):
        return cls.alpha_m(V) / (cls.alpha_m(V) + cls.beta_m(V))

    @classmethod
    def h_inf(cls, V):
        return cls.alpha_h(V) / (cls.alpha_h(V) + cls.beta_h(V))

    @classmethod
    def tau_m(cls, V):
        return 1 / (cls.alpha_m(V) + cls.beta_m(V))

    @classmethod
    def tau_h(cls, V):
        return 1 / (cls.alpha_h(V) + cls.beta_h(V))

class PotassiumChannel(IonChannel):
    @classmethod
    def alpha_n(cls, V):
        return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))

    @classmethod
    def beta_n(cls, V):
        return 0.125 * np.exp(-(V + 65) / 80)

    @classmethod
    def n_inf(cls, V):
        return cls.alpha_n(V) / (cls.alpha_n(V) + cls.beta_n(V))

    @classmethod
    def tau_n(cls, V):
        return 1 / (cls.alpha_n(V) + cls.beta_n(V))

class CalciumChannel(IonChannel):
    @classmethod
    def s_inf(cls, V):
        return cls.boltzmann(V, -30, 9.5)

    @classmethod
    def u_inf(cls, V):
        return cls.boltzmann(V, -30, -9.5)

    @classmethod
    def tau_s(cls, V):
        return 0.05 + 0.3 / (1 + np.exp((V + 27) / 10))

    @classmethod
    def tau_u(cls, V):
        return 0.2 + 0.3 / (1 + np.exp(-(V + 27) / 10))

class HCNChannel(IonChannel):
    @classmethod
    def r_inf(cls, V):
        return 1 / (1 + np.exp((V + 76) / 7))

    @classmethod
    def tau_r(cls, V):
        return 1 / (np.exp(-14.59 - 0.086 * V) + np.exp(-1.87 + 0.0701 * V))
