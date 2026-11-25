#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOTOR DE EXPERIMENTOS DETERMINISTAS v1.0 — POST-EARTHQUAKE RECOVERY & INDICATOR TRAP (CALI, COLOMBIA)
Proyecto: 2026_terremoto_cali (Coautoría con Ivan Taylor — International Journal of Disaster Risk Reduction, Elsevier)
Replica exactamente el modelo integrado 'sfd earthquake integrated with ivan ver colombia 2026.stmx':
  - Horizonte: 0.0 a 60.0 Months, DT = 0.0625, integración RK4
  - 16 Stocks únicos activos (+ 6 fantasmas inter-módulo = 22 en XML) en 6 subsistemas:
      1. Execution (4 stocks): projects_in_execution_pipeline, certified_construction_capacity,
         inspection_backlog, fund_absorption_and_inspection_capacity
      2. Fiscal (1 stock): recovery_fund_balance ($2,500M USD inicial)
      3. Governance / PMU (5 stocks): error_ontologico, presion_politica, legitimidad_gobierno,
         fatiga_pmu, capacidad_lectura_pmu
      4. Housing (3 stocks): housing_recovery_backlog (200,000), stock_of_latent_vulnerable_structures,
         stock_seismically_resilient_housing
      5. Infrastructure & Livelihoods (2 stocks): critical_infrastructure_backlog (1,000),
         livelihood_social_vulnerability_backlog (200,000)
      6. Metrics (1 stock): cumulative_reported_recovery_index
  - 30 Flujos, 44 Auxiliares, 53 Constantes, 21 Lazos LTM

Incluye además:
  - Verificación de conservación exacta de viviendas (200,000 unidades)
  - Convergencia RK4 a 3 pasos (0.0625 -> 0.03125 -> 0.015625) + contraste Euler
  - Pruebas de Condiciones Extremas Barlas sin clamping (Extremo 1: Crisis Fiscal; Extremo 2: Capacidad 100% Resiliente)
  - Sobol Panel A (7 parámetros estructurales, n=512 -> N=4,608 corridas, IC 95% + bandas P10-P90)
  - Sobol Panel B (3 palancas de política: Control de Presión Política, Inspección/Cumplimiento NSR-10 y Reabastecimiento Fiscal; N=1,280)
  - Curador automático de los 199 registros en datos_empiricos/*.csv para diálogo directo con IJDRR.

Uso:
  python 03_proyectos/2026_terremoto_cali/modelo_stella/experimentos_terremoto_cali_sobol.py --proyecto 2026_terremoto_cali
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from SALib.analyze import sobol as sobol_analyze
    try:
        from SALib.sample import sobol as sobol_sample
    except ImportError:
        from SALib.sample import saltelli as sobol_sample
    TIENE_SALIB = True
except ImportError:
    TIENE_SALIB = False


# ==============================================================================
# 53 PARÁMETROS CONSTANTES EXACTOS DE 'sfd earthquake integrated with ivan ver colombia 2026.stmx'
# ==============================================================================
PARAMS_BASE: Dict[str, float] = {
    # Execution
    "unit_cost_resilient_housing": 0.025,
    "standard_construction_delay": 6.0,
    "baseline_construction_capacity": 4000.0,
    "base_training_rate": 100.0,
    "diversion_fraction": 0.02,
    "unit_cost_per_inspection": 0.001,
    "inspection_processing_time": 3.0,
    "capacity_gain_factor": 0.01,
    "capacity_depreciation_time": 60.0,
    "resilient_contracting_target_time": 8.24,
    "unidades_gestionables_por_punto_capacidad": 1000.0,
    "max_absorption_boost": 1.0,
    "half_saturation_capacity": 200.0,
    # Fiscal
    "total_initial_recovery_budget": 2500.0,
    "financial_adjustment_time": 1.0,
    "base_housing_monthly_funding_demand": 80.0,
    "infrastructure_demand_weight": 10.0,
    "inspection_demand_weight": 2.0,
    "technocratic_quick_repair_fraction": 0.2,
    "policy_switch": 0.0,
    "external_replenishment_amount": 0.0,
    "external_replenishment_start_time": 24.0,
    # Governance / PMU
    "tau_error": 6.0,
    "indice_techo": 100.0,
    "ganancia_generacion_presion": 10.0,
    "tau_apaciguamiento": 12.0,
    "ganancia_recuperacion_legitimidad": 4.0,
    "coef_erosion_legitimidad": 0.05,
    "ganancia_acumulacion_fatiga": 3.0,
    "recup_fatiga": 0.03,
    "ganancia_aprendizaje": 2.5,
    "tau_erosion_capacidad": 24.0,
    "beta_desacople": 1.0,
    # Housing
    "initial_housing_backlog": 200000.0,
    "quick_repair_target_time": 8.0,
    "quick_repair_unit_cost": 0.005,
    "fraction_low_compliance_repairs": 0.65,
    "average_time_to_latent_failure": 24.0,
    "residual_seismic_vulnerability": 0.15,
    # Infrastructure & Livelihoods
    "initial_infrastructure_backlog": 1000.0,
    "unit_cost_infrastructure": 0.5,
    "target_infrastructure_delivery_delay": 18.0,
    "initial_livelihood_backlog": 200000.0,
    "vulnerability_accumulation_rate_base": 0.0,
    "base_livelihood_recovery_rate": 3500.0,
    "target_livelihood_time": 12.0,
    # Metrics
    "total_target_cases": 200000.0,
    "housing_weight_in_substantive_index": 0.4,
    "infrastructure_weight_in_substantive_index": 0.3,
    "livelihood_weight_in_substantive_index": 0.3,
}

# 4 Escenarios de Política + 2 Extremos Barlas
ESCENARIOS: Dict[str, Dict[str, float]] = {
    "01_Indicator_Driven_Political": {
        "policy_switch": 1.0,
        "fraction_low_compliance_repairs": 0.65,
        "inspection_demand_weight": 2.0,
        "external_replenishment_amount": 0.0,
    },
    "02_FastTrack_Deregulation": {
        "policy_switch": 1.0,
        "fraction_low_compliance_repairs": 0.85,
        "diversion_fraction": 0.04,
        "inspection_demand_weight": 0.5,
        "external_replenishment_amount": 0.0,
    },
    "03_Technocratic_Resilient_NSR10": {
        "policy_switch": 0.0,
        "technocratic_quick_repair_fraction": 0.20,
        "fraction_low_compliance_repairs": 0.20,
        "inspection_demand_weight": 4.0,
        "external_replenishment_amount": 0.0,
    },
    "04_Adaptive_Hybrid_Safeguards": {
        "policy_switch": 0.0,
        "technocratic_quick_repair_fraction": 0.05,
        "fraction_low_compliance_repairs": 0.05,
        "inspection_demand_weight": 6.0,
        "external_replenishment_amount": 50.0,
        "external_replenishment_start_time": 18.0,
    },
    "05_Extreme1_Fiscal_Collapse": {
        "policy_switch": 1.0,
        "total_initial_recovery_budget": 500.0,
        "fraction_low_compliance_repairs": 0.90,
        "external_replenishment_amount": 0.0,
    },
    "06_Extreme2_Zero_Cosmetic_FullFund": {
        "policy_switch": 0.0,
        "technocratic_quick_repair_fraction": 0.0,
        "fraction_low_compliance_repairs": 0.0,
        "total_initial_recovery_budget": 6000.0,
        "inspection_demand_weight": 8.0,
    },
}


def estado_inicial(p: Dict[str, float]) -> np.ndarray:
    """16 stocks únicos + 1 acumulador contable de reparaciones rápidas estables (para test de conservación)."""
    return np.array([
        0.0,                                 # 0: projects_in_execution_pipeline
        p["baseline_construction_capacity"], # 1: certified_construction_capacity
        0.0,                                 # 2: inspection_backlog
        50.0,                                # 3: fund_absorption_and_inspection_capacity
        p["total_initial_recovery_budget"],  # 4: recovery_fund_balance
        0.85,                                # 5: error_ontologico
        50.0,                                # 6: presion_politica
        60.0,                                # 7: legitimidad_gobierno
        10.0,                                # 8: fatiga_pmu
        20.0,                                # 9: capacidad_lectura_pmu
        p["initial_housing_backlog"],        # 10: housing_recovery_backlog
        0.0,                                 # 11: stock_of_latent_vulnerable_structures
        0.0,                                 # 12: stock_seismically_resilient_housing
        p["initial_infrastructure_backlog"], # 13: critical_infrastructure_backlog
        p["initial_livelihood_backlog"],     # 14: livelihood_social_vulnerability_backlog
        0.0,                                 # 15: cumulative_reported_recovery_index
        0.0,                                 # 16: quick_repairs_compliant_accounting (test conservación)
    ], dtype=float)


def evaluar_sistema(t: float, y: np.ndarray, p: Dict[str, float], aplicar_clamp: bool = True) -> Tuple[np.ndarray, Dict[str, float]]:
    if aplicar_clamp:
        pipe = max(0.0, float(y[0]))
        cert_cap = max(1e-6, float(y[1]))
        insp_b = max(0.0, float(y[2]))
        abs_cap = max(0.0, float(y[3]))
        fund = max(0.0, float(y[4]))
        err_ont = max(0.0, min(1.0, float(y[5])))
        pres_pol = max(0.0, min(p["indice_techo"], float(y[6])))
        legit = max(0.0, min(p["indice_techo"], float(y[7])))
        fatiga = max(0.0, min(p["indice_techo"], float(y[8])))
        cap_pmu = max(1e-6, min(p["indice_techo"], float(y[9])))
        h_back = max(0.0, float(y[10]))
        h_lat = max(0.0, float(y[11]))
        h_res = max(0.0, float(y[12]))
        k_inf = max(0.0, float(y[13]))
        l_vul = max(0.0, float(y[14]))
        rep_idx = max(0.0, min(1.0, float(y[15])))
        q_ok = max(0.0, float(y[16]))
    else:
        pipe, cert_cap, insp_b, abs_cap, fund = float(y[0]), max(1e-6, float(y[1])), float(y[2]), float(y[3]), float(y[4])
        err_ont, pres_pol, legit, fatiga, cap_pmu = float(y[5]), float(y[6]), float(y[7]), float(y[8]), max(1e-6, float(y[9]))
        h_back, h_lat, h_res, k_inf, l_vul, rep_idx, q_ok = (
            float(y[10]), float(y[11]), float(y[12]), float(y[13]), float(y[14]), float(y[15]), float(y[16])
        )

    techo = p["indice_techo"]
    pres_pol_norm = pres_pol / techo
    avail_serv = max(0.0, min(1.0, 1.0 - (k_inf / p["initial_infrastructure_backlog"])))

    # Substantive Recovery Index & Indicator Gap
    sub_idx = (
        p["housing_weight_in_substantive_index"] * (h_res / p["initial_housing_backlog"])
        + p["infrastructure_weight_in_substantive_index"] * avail_serv
        + p["livelihood_weight_in_substantive_index"] * max(0.0, min(1.0, 1.0 - (l_vul / p["initial_livelihood_backlog"])))
    )
    ind_gap = max(0.0, rep_idx - sub_idx)
    perc_need = max(0.0, min(2.0, (1.0 - rep_idx) + ind_gap))

    # Fiscal Module
    ext_disb = p["external_replenishment_amount"] if t >= p["external_replenishment_start_time"] else 0.0
    tot_h_dem = p["base_housing_monthly_funding_demand"] * perc_need
    frac_quick = p["policy_switch"] * pres_pol_norm + (1.0 - p["policy_switch"]) * p["technocratic_quick_repair_fraction"]
    frac_res = 1.0 - frac_quick

    des_quick = tot_h_dem * frac_quick
    des_res = tot_h_dem * frac_res
    des_inf = p["infrastructure_demand_weight"] * perc_need
    des_gov = p["inspection_demand_weight"] * perc_need
    tot_des = des_quick + des_res + des_inf + des_gov

    denom_liq = tot_des * p["financial_adjustment_time"]
    liq_rat = min(1.0, fund / denom_liq) if denom_liq > 1e-9 else 1.0

    alloc_quick = des_quick * liq_rat
    alloc_res = des_res * liq_rat
    inf_exp = des_inf * liq_rat
    alloc_gov = des_gov * liq_rat

    # Execution Module
    eff_cap_abs = 1.0 + p["max_absorption_boost"] * (abs_cap / (abs_cap + p["half_saturation_capacity"]))
    sobrecarga = pipe / max(1e-6, cap_pmu * p["unidades_gestionables_por_punto_capacidad"] * eff_cap_abs)
    eff_over = 1.0 / (1.0 + sobrecarga)

    proj_contr = min(alloc_res / p["unit_cost_resilient_housing"], h_back / p["resilient_contracting_target_time"])
    phys_comp = min(pipe / p["standard_construction_delay"], cert_cap) * eff_over

    quick_exec = min(alloc_quick / p["quick_repair_unit_cost"], h_back / p["quick_repair_target_time"])
    work_train = p["base_training_rate"] * perc_need
    div_labor = cert_cap * min(1.0, (quick_exec * p["diversion_fraction"]) / cert_cap)

    insp_req = alloc_gov / p["unit_cost_per_inspection"]
    comp_verif = insp_b / p["inspection_processing_time"]
    cap_dev = comp_verif * p["capacity_gain_factor"]
    cap_dep = abs_cap / p["capacity_depreciation_time"]

    # Governance / PMU Module
    discrep = min(1.0, p["beta_desacople"] * (ind_gap + (1.0 - (cap_pmu / techo)) * rep_idx))
    d_err = (discrep - err_ont) / p["tau_error"]
    gen_pres = p["ganancia_generacion_presion"] * err_ont * (1.0 - legit / techo) * (1.0 - pres_pol_norm)
    apacig = (pres_pol * rep_idx) / p["tau_apaciguamiento"]
    rec_leg = p["ganancia_recuperacion_legitimidad"] * rep_idx * (1.0 - legit / techo)
    ero_leg = p["coef_erosion_legitimidad"] * legit * err_ont * pres_pol_norm
    ac_fat = p["ganancia_acumulacion_fatiga"] * min(1.0, sobrecarga) * (1.0 - fatiga / techo)
    rec_fat = p["recup_fatiga"] * fatiga
    autoorg = (legit / techo) * (1.0 - fatiga / techo)
    aprend = p["ganancia_aprendizaje"] * autoorg * (1.0 - cap_pmu / techo)
    ero_cap = (cap_pmu * (fatiga / techo)) / p["tau_erosion_capacidad"]

    # Housing, Infrastructure & Metrics
    h_low_comp = quick_exec * p["fraction_low_compliance_repairs"]
    h_quick_ok = quick_exec * (1.0 - p["fraction_low_compliance_repairs"])
    str_fail = (h_lat / p["average_time_to_latent_failure"]) * (1.0 + p["residual_seismic_vulnerability"])

    inf_deliv = min(inf_exp / p["unit_cost_infrastructure"], k_inf / p["target_infrastructure_delivery_delay"])
    vul_acc = p["vulnerability_accumulation_rate_base"]
    econ_react = min(l_vul / p["target_livelihood_time"], p["base_livelihood_recovery_rate"] * avail_serv)

    adm_close = quick_exec + phys_comp
    rep_inc = (adm_close / p["total_target_cases"]) * max(0.0, 1.0 - rep_idx)

    # Derivadas de los 16 stocks + 1 acumulador contable
    dy = np.array([
        proj_contr - phys_comp,
        work_train - div_labor,
        insp_req - comp_verif,
        cap_dev - cap_dep,
        ext_disb - alloc_quick - alloc_res - inf_exp - alloc_gov,
        d_err,
        gen_pres - apacig,
        rec_leg - ero_leg,
        ac_fat - rec_fat,
        aprend - ero_cap,
        str_fail - quick_exec - phys_comp,
        h_low_comp - str_fail,
        phys_comp,
        -inf_deliv,
        vul_acc - econ_react,
        rep_inc,
        h_quick_ok,
    ], dtype=float)

    resid_viv = (h_back + h_lat + h_res + q_ok) - p["initial_housing_backlog"]

    aux = {
        "reported_index_pct": 100.0 * rep_idx,
        "substantive_index_pct": 100.0 * sub_idx,
        "indicator_gap_pct": 100.0 * ind_gap,
        "perceived_need": perc_need,
        "liquidity_rationing": liq_rat,
        "sobrecarga_admin": sobrecarga,
        "quick_repair_rate": quick_exec,
        "resilient_completion_rate": phys_comp,
        "latent_failure_rate": str_fail,
        "avail_public_services": avail_serv,
        "residual_housing_cons": resid_viv,
    }
    return dy, aux


def simular(
    overrides: Dict[str, float],
    dt: float = 0.0625,
    t0: float = 0.0,
    tf: float = 60.0,
    metodo: str = "RK4",
    aplicar_clamp: bool = True,
) -> pd.DataFrame:
    p = dict(PARAMS_BASE)
    p.update(overrides)
    y = estado_inicial(p)

    n_steps = int(round((tf - t0) / dt))
    pasos_mes = max(1, int(round(1.0 / dt)))
    filas: List[Dict[str, float]] = []

    for step in range(n_steps + 1):
        t = t0 + step * dt
        dy, aux = evaluar_sistema(t, y, p, aplicar_clamp=aplicar_clamp)
        if step % (pasos_mes * 6) == 0 or step == n_steps:
            filas.append({
                "Month": round(t, 2),
                "Housing_Backlog": round(float(y[10]), 1),
                "Latent_Vulnerable": round(float(y[11]), 1),
                "Resilient_Housing": round(float(y[12]), 1),
                "Pipeline": round(float(y[0]), 1),
                "Certified_Cap": round(float(y[1]), 1),
                "Fund_Balance": round(float(y[4]), 2),
                "Error_Ontologico": round(float(y[5]), 4),
                "Presion_Politica": round(float(y[6]), 2),
                "Legitimidad": round(float(y[7]), 2),
                "Fatiga_PMU": round(float(y[8]), 2),
                "Cap_PMU": round(float(y[9]), 2),
                "Infra_Backlog": round(float(y[13]), 1),
                "Livelihood_Backlog": round(float(y[14]), 1),
                **{k: round(float(v), 4) for k, v in aux.items()},
            })
        if step == n_steps:
            break

        if metodo.lower() == "rk4":
            k1 = dy
            k2, _ = evaluar_sistema(t + 0.5 * dt, y + 0.5 * dt * k1, p, aplicar_clamp=aplicar_clamp)
            k3, _ = evaluar_sistema(t + 0.5 * dt, y + 0.5 * dt * k2, p, aplicar_clamp=aplicar_clamp)
            k4, _ = evaluar_sistema(t + dt, y + dt * k3, p, aplicar_clamp=aplicar_clamp)
            y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            y = y + dt * dy

        if aplicar_clamp:
            for idx_i in range(len(y)):
                y[idx_i] = max(0.0, y[idx_i])

    return pd.DataFrame(filas)


def ejecutar_sobol_panel_a(n_base: int = 512) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Panel A: 7 parámetros estructurales sobre el régimen Indicator-Driven (n=512 -> N=4,608 corridas, IC 95%)."""
    if not TIENE_SALIB:
        return pd.DataFrame(), {"ejecutado": False}

    nombres = [
        "fraction_low_compliance_repairs",
        "average_time_to_latent_failure",
        "standard_construction_delay",
        "baseline_construction_capacity",
        "diversion_fraction",
        "base_housing_monthly_funding_demand",
        "total_initial_recovery_budget",
    ]
    bounds = [
        [0.40, 0.85],
        [12.0, 36.0],
        [4.0, 10.0],
        [2500.0, 6000.0],
        [0.005, 0.045],
        [50.0, 110.0],
        [1500.0, 3500.0],
    ]
    problem = {"num_vars": len(nombres), "names": nombres, "bounds": bounds}
    X = sobol_sample.sample(problem, n_base, calc_second_order=False)
    n_tot = len(X)

    y_res = np.zeros(n_tot)
    y_lat = np.zeros(n_tot)
    y_gap = np.zeros(n_tot)
    y_back = np.zeros(n_tot)

    for i, fila in enumerate(X):
        ov = {"policy_switch": 1.0}
        for j, nm in enumerate(nombres):
            ov[nm] = float(fila[j])
        fin = simular(ov, dt=0.125, metodo="RK4").iloc[-1]
        y_res[i] = fin["Resilient_Housing"]
        y_lat[i] = fin["Latent_Vulnerable"]
        y_gap[i] = fin["indicator_gap_pct"]
        y_back[i] = fin["Housing_Backlog"]

    si_res = sobol_analyze.analyze(problem, y_res, calc_second_order=False, conf_level=0.95, print_to_console=False)
    si_lat = sobol_analyze.analyze(problem, y_lat, calc_second_order=False, conf_level=0.95, print_to_console=False)
    si_gap = sobol_analyze.analyze(problem, y_gap, calc_second_order=False, conf_level=0.95, print_to_console=False)

    # Invarianza ordinal: Adaptive Hybrid > Indicator-Driven en Resilient Housing y < en Indicator Gap
    rng = np.random.default_rng(2026)
    idxs = rng.choice(n_tot, size=min(60, n_tot), replace=False)
    inv_ok = 0
    for idx in idxs:
        ov0 = {"policy_switch": 1.0}
        for j, nm in enumerate(nombres):
            ov0[nm] = float(X[idx][j])
        ov1 = dict(ov0)
        ov1.update(ESCENARIOS["04_Adaptive_Hybrid_Safeguards"])
        f0 = simular(ov0, dt=0.125).iloc[-1]
        f1 = simular(ov1, dt=0.125).iloc[-1]
        if (f1["Resilient_Housing"] > f0["Resilient_Housing"]) and (f1["indicator_gap_pct"] < f0["indicator_gap_pct"]):
            inv_ok += 1

    df_s = pd.DataFrame({
        "Parametro": nombres,
        "S1_Resilient": np.round(si_res["S1"], 4),
        "ST_Resilient": np.round(si_res["ST"], 4),
        "ST_Res_CI95": np.round(si_res["ST_conf"], 4),
        "ST_Latent": np.round(si_lat["ST"], 4),
        "ST_Gap": np.round(si_gap["ST"], 4),
        "ST_Gap_CI95": np.round(si_gap["ST_conf"], 4),
    }).sort_values("ST_Resilient", ascending=False)

    meta = {
        "ejecutado": True,
        "n_base": int(n_base),
        "n_corridas": int(n_tot),
        "resilient_m60_p10_p90": [round(float(np.percentile(y_res, 10)), 1), round(float(np.percentile(y_res, 90)), 1)],
        "latent_m60_p10_p90": [round(float(np.percentile(y_lat, 10)), 1), round(float(np.percentile(y_lat, 90)), 1)],
        "gap_m60_p10_p90": [round(float(np.percentile(y_gap, 10)), 2), round(float(np.percentile(y_gap, 90)), 2)],
        "backlog_m60_p10_p90": [round(float(np.percentile(y_back, 10)), 1), round(float(np.percentile(y_back, 90)), 1)],
        "invarianza_ordinal_pct": round(100.0 * inv_ok / len(idxs), 1),
    }
    return df_s, meta


def ejecutar_sobol_panel_b(n_pol: int = 256) -> Tuple[pd.DataFrame, int]:
    """Panel B: Separación funcional de las 3 palancas de política [0, 1]^3 (Gobernanza Anti-Presión, Inspección NSR-10, Reabastecimiento Fiscal)."""
    if not TIENE_SALIB:
        return pd.DataFrame(), 0

    nombres = ["Palanca_1_Gobernanza_Techno", "Palanca_2_Inspeccion_NSR10", "Palanca_3_Replenishment_Fiscal"]
    problem = {"num_vars": 3, "names": nombres, "bounds": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]}
    X = sobol_sample.sample(problem, n_pol, calc_second_order=False)
    n_tot = len(X)

    y_res, y_lat, y_gap, y_sub = np.zeros(n_tot), np.zeros(n_tot), np.zeros(n_tot), np.zeros(n_tot)
    for i, fila in enumerate(X):
        p_gov, p_ins, p_fis = float(fila[0]), float(fila[1]), float(fila[2])
        ov = {
            "policy_switch": 1.0 - p_gov,                                # 1 (political) -> 0 (technocratic)
            "technocratic_quick_repair_fraction": 0.25 - 0.20 * p_gov,   # 0.25 -> 0.05
            "fraction_low_compliance_repairs": 0.75 - 0.70 * p_ins,      # 0.75 -> 0.05
            "inspection_demand_weight": 1.0 + 5.0 * p_ins,               # 1.0 -> 6.0
            "external_replenishment_amount": 60.0 * p_fis,               # 0 -> 60 M USD/mo
            "external_replenishment_start_time": 18.0,
        }
        fin = simular(ov, dt=0.125).iloc[-1]
        y_res[i] = fin["Resilient_Housing"]
        y_lat[i] = fin["Latent_Vulnerable"]
        y_gap[i] = fin["indicator_gap_pct"]
        y_sub[i] = fin["substantive_index_pct"]

    si_res = sobol_analyze.analyze(problem, y_res, calc_second_order=False, conf_level=0.95, print_to_console=False)
    si_lat = sobol_analyze.analyze(problem, y_lat, calc_second_order=False, conf_level=0.95, print_to_console=False)
    si_gap = sobol_analyze.analyze(problem, y_gap, calc_second_order=False, conf_level=0.95, print_to_console=False)
    si_sub = sobol_analyze.analyze(problem, y_sub, calc_second_order=False, conf_level=0.95, print_to_console=False)

    df_b = pd.DataFrame({
        "Palanca_Politica": nombres,
        "ST_Resilient": np.round(si_res["ST"], 4),
        "ST_Res_CI95": np.round(si_res["ST_conf"], 4),
        "ST_Latent": np.round(si_lat["ST"], 4),
        "ST_Gap": np.round(si_gap["ST"], 4),
        "ST_Substantive": np.round(si_sub["ST"], 4),
    })
    return df_b, n_tot


def curar_literatura_local_ijdrr(proy_dir: Path, ssot_dir: Path) -> int:
    """Extrae los mejores artículos peer-reviewed (priorizando IJDRR) de los 199 registros en datos_empiricos/*.csv."""
    dat_dir = proy_dir / "datos_empiricos"
    csvs = [
        dat_dir / "terremotos y vulnerabilidad social revlit.csv",
        dat_dir / "respuesta terremoto 10 agosto26 trampa SE.csv",
    ]
    vistos: Dict[str, Dict[str, Any]] = {}
    for c in csvs:
        if not c.exists():
            continue
        try:
            with open(c, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
                for r in csv.DictReader(f):
                    tit = (r.get("reference_title") or r.get("title") or "").strip()
                    if not tit:
                        continue
                    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", (r.get("reference_doi") or r.get("doi") or "").strip(), flags=re.I)
                    rev = (r.get("reference_journal") or r.get("journal") or "").strip()
                    anio = (r.get("reference_year") or r.get("year") or "").strip()[:4]
                    auts = (r.get("reference_authors") or r.get("authors") or "").strip()[:42]
                    texto = f"{tit} {rev}".lower()
                    score = 0
                    if "disaster risk reduction" in rev.lower() or "ijdrr" in rev.lower():
                        score += 20
                    if any(k in texto for k in ("housing", "reconstruction", "build back better", "earthquake", "seismic", "recovery")):
                        score += 8
                    if any(k in texto for k in ("system dynamics", "indicator", "goodhart", "governance", "vulnerability", "resilience")):
                        score += 6
                    if anio.isdigit() and int(anio) >= 2020:
                        score += 5
                    clave = doi.lower() if doi else re.sub(r"[^a-z0-9]", "", tit.lower())[:55]
                    if clave not in vistos or score > vistos[clave]["score"]:
                        vistos[clave] = {"anio": anio or "-", "autores": auts, "titulo": tit, "revista": rev, "doi": doi, "score": score}
        except Exception:
            continue

    ordenados = sorted(vistos.values(), key=lambda x: (x["score"], x["anio"]), reverse=True)
    md = [
        "# Literatura Curada desde CSVs Locales (Prioridad IJDRR) — `2026_terremoto_cali`",
        f"**Generado:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | **Registros únicos:** `{len(ordenados)}`\n",
        "| Año | Autores | Título | Revista | DOI |",
        "| :---: | :--- | :--- | :--- | :--- |",
    ]
    for it in ordenados[:30]:
        md.append(f"| {it['anio']} | {it['autores']} | {it['titulo'][:76]} | {it['revista'][:36]} | `{it['doi'] or '-'}` |")

    (ssot_dir / "top_literatura_terremoto_ijdrr.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return len(ordenados)


def main() -> None:
    parser = argparse.ArgumentParser(description="Motor RK4 + Barlas + Sobol + Curador IJDRR para 2026_terremoto_cali.")
    parser.add_argument("--proyecto", default="2026_terremoto_cali")
    parser.add_argument("--n-sobol", type=int, default=512)
    args = parser.parse_args()

    p_script = Path(__file__).resolve()
    raiz = next((parent for parent in p_script.parents if (parent / "03_proyectos").exists()), Path.cwd())
    proy_dir = raiz / "03_proyectos" / args.proyecto
    csv_dir = proy_dir / "modelo_stella" / "corridas_csv"
    ssot_dir = proy_dir / "ssot"
    csv_dir.mkdir(parents=True, exist_ok=True)
    ssot_dir.mkdir(parents=True, exist_ok=True)

    print("=========================================================================")
    print(f" MOTOR RK4 + BARLAS + SOBOL (MODELO INTEGRADO CON IVAN) | {args.proyecto}")
    print("=========================================================================")

    trajs: Dict[str, pd.DataFrame] = {}
    res_rows = []
    max_cons_err = 0.0
    for nom, ov in ESCENARIOS.items():
        df = simular(ov, dt=0.0625, metodo="RK4")
        df.to_csv(csv_dir / f"{nom}.csv", index=False)
        trajs[nom] = df
        fin = df.iloc[-1]
        err_c = float(df["residual_housing_cons"].abs().max())
        max_cons_err = max(max_cons_err, err_c)
        res_rows.append({
            "Escenario": nom,
            "Backlog_M60": round(float(fin["Housing_Backlog"]), 1),
            "Latent_Peak": round(float(df["Latent_Vulnerable"].max()), 1),
            "Latent_M60": round(float(fin["Latent_Vulnerable"]), 1),
            "Resilient_M60": round(float(fin["Resilient_Housing"]), 1),
            "RepIdx_pct": round(float(fin["reported_index_pct"]), 2),
            "SubIdx_pct": round(float(fin["substantive_index_pct"]), 2),
            "Gap_pct": round(float(fin["indicator_gap_pct"]), 2),
            "Fund_M60": round(float(fin["Fund_Balance"]), 1),
            "Legit_M60": round(float(fin["Legitimidad"]), 1),
        })
        print(
            f"   [OK] {nom:<34} | Back={fin['Housing_Backlog']:8.1f} | "
            f"Lat={fin['Latent_Vulnerable']:7.1f} | Res={fin['Resilient_Housing']:8.1f} | "
            f"Gap={fin['indicator_gap_pct']:5.2f}% | Fund={fin['Fund_Balance']:6.1f}"
        )
    df_res = pd.DataFrame(res_rows)

    # Convergencia RK4 a 3 pasos + Euler
    conv_rows = []
    for sc in ("01_Indicator_Driven_Political", "04_Adaptive_Hybrid_Safeguards"):
        r1 = simular(ESCENARIOS[sc], dt=0.0625, metodo="RK4").iloc[-1]
        r2 = simular(ESCENARIOS[sc], dt=0.03125, metodo="RK4").iloc[-1]
        r3 = simular(ESCENARIOS[sc], dt=0.015625, metodo="RK4").iloc[-1]
        eu = simular(ESCENARIOS[sc], dt=0.0625, metodo="Euler").iloc[-1]
        conv_rows.append({
            "Escenario": sc,
            "RK4_0.0625": round(float(r1["Resilient_Housing"]), 3),
            "RK4_0.03125": round(float(r2["Resilient_Housing"]), 3),
            "RK4_0.015625": round(float(r3["Resilient_Housing"]), 3),
            "Diff_RK4_Max": f"{abs(float(r1['Resilient_Housing']) - float(r3['Resilient_Housing'])):.2e}",
            "Euler_0.0625": round(float(eu["Resilient_Housing"]), 3),
            "Diff_RK4_Eu": f"{abs(float(r1['Resilient_Housing']) - float(eu['Resilient_Housing'])):.2e}",
        })
    df_conv = pd.DataFrame(conv_rows)

    # Barlas unclamped
    df_u1 = simular(ESCENARIOS["05_Extreme1_Fiscal_Collapse"], dt=0.0625, aplicar_clamp=False)
    df_u2 = simular(ESCENARIOS["06_Extreme2_Zero_Cosmetic_FullFund"], dt=0.0625, aplicar_clamp=False)

    print(f"1. [SOBOL PANEL A] Ejecutando sensibilidad global 7 parámetros (n={args.n_sobol}, N={args.n_sobol*9})...")
    df_sob_a, meta_a = ejecutar_sobol_panel_a(n_base=args.n_sobol)
    df_sob_a.to_csv(ssot_dir / "sobol_panel_a_terremoto.csv", index=False)

    print("2. [SOBOL PANEL B] Ejecutando separación funcional de 3 palancas de política (N=1,280)...")
    df_sob_b, n_tot_b = ejecutar_sobol_panel_b(n_pol=256)
    df_sob_b.to_csv(ssot_dir / "sobol_panel_b_terremoto.csv", index=False)

    n_lit = curar_literatura_local_ijdrr(proy_dir, ssot_dir)
    print(f"3. [LITERATURA LOCAL] {n_lit} estudios curados en ssot/top_literatura_terremoto_ijdrr.md")

    cols_t = ["Month", "Housing_Backlog", "Latent_Vulnerable", "Resilient_Housing", "reported_index_pct", "substantive_index_pct", "indicator_gap_pct", "Fund_Balance", "Legitimidad"]
    md = [
        f"# Resultados Verificados RK4 + Barlas + Sobol (Modelo Integrado con Ivan Taylor) — `{args.proyecto}`",
        f"**Generado:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **SSOT:** `sfd earthquake integrated with ivan ver colombia 2026.stmx` (0–60 Months, RK4, DT=0.0625)\n",
        "## 1. Matriz Comparativa de 6 Escenarios al Mes 60",
        df_res.to_markdown(index=False),
        "\n## 2. Trayectorias Semestrales (Meses 0–60) — Indicator-Driven vs. Adaptive Hybrid",
        "### 2.1 `01_Indicator_Driven_Political`",
        trajs["01_Indicator_Driven_Political"][cols_t].to_markdown(index=False),
        "\n### 2.2 `03_Technocratic_Resilient_NSR10`",
        trajs["03_Technocratic_Resilient_NSR10"][cols_t].to_markdown(index=False),
        "\n### 2.3 `04_Adaptive_Hybrid_Safeguards`",
        trajs["04_Adaptive_Hybrid_Safeguards"][cols_t].to_markdown(index=False),
        "\n## 3. Verificación Numérica: Conservación de Vivienda, Convergencia RK4 y Barlas Unclamped",
        f"- **Conservación exacta de viviendas (`max |H_back + H_lat + H_res + Q_ok - 200,000|`):** `{max_cons_err:.2e}`",
        f"- **Barlas Extremo 1 Unclamped (Crisis Fiscal $500M):** `min(Fund)={df_u1['Fund_Balance'].min():.2f}`, `Backlog_M60={df_u1['Housing_Backlog'].iloc[-1]:.1f}`, `Gap={df_u1['indicator_gap_pct'].iloc[-1]:.2f}%`",
        f"- **Barlas Extremo 2 Unclamped (Cero Cosmético + $6000M):** `Latent_max={df_u2['Latent_Vulnerable'].max():.2f}`, `Resilient_M60={df_u2['Resilient_Housing'].iloc[-1]:.1f}`\n",
        df_conv.to_markdown(index=False),
        f"\n## 4. Sobol Panel A — Incertidumbre Paramétrica (n={args.n_sobol}, N={meta_a.get('n_corridas', 0)}, IC 95%)",
        f"- **Metadatos:** `{json.dumps(meta_a, ensure_ascii=False)}`\n",
        df_sob_a.to_markdown(index=False),
        f"\n## 5. Sobol Panel B — Separación Funcional de las 3 Palancas de Política (N={n_tot_b}, IC 95%)",
        df_sob_b.to_markdown(index=False),
    ]
    ruta_rep = ssot_dir / "resumen_experimentos_terremoto_cali.md"
    ruta_rep.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n✅ Reporte completo guardado en: {ruta_rep}")
    print("=========================================================================")


if __name__ == "__main__":
    main()