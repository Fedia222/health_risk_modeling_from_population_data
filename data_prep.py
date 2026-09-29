import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data_raw"
OUTPUT = ROOT / "data" / "processed" / "cohort.csv"

FILES = [
    "DEMO_F.xpt",
    "BMX_F.xpt",
    "NHANES_2009_2010_MORT_2019_PUBLIC.dat",
]


def prepare_data(rebuild=False):
    if OUTPUT.exists() and not rebuild:
        print(f"Données déjà préparées : {OUTPUT}")
        return OUTPUT

    paths = {name: RAW / name for name in FILES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Fichiers absents dans {RAW} : {', '.join(missing)}"
        )
    demo = pd.read_sas(paths["DEMO_F.xpt"], format="xport")
    body = pd.read_sas(paths["BMX_F.xpt"], format="xport")
    # Positions officielles CDC : pandas compte depuis 0, borne finale exclue.
    mortality = pd.read_fwf(
        paths["NHANES_2009_2010_MORT_2019_PUBLIC.dat"],
        colspecs=[(0, 6), (14, 15), (15, 16), (45, 48)],
        names=["SEQN", "ELIGSTAT", "MORTSTAT", "PERMTH_EXM"],
        na_values=["."],
    )
    for table in (demo, body, mortality):
        if table.SEQN.isna().any() or not table.SEQN.is_unique:
            raise ValueError("Identifiants absents ou dupliqués")
    if set(demo.SEQN) != set(mortality.SEQN):
        raise ValueError("Les identifiants NHANES et mortalité ne correspondent pas")

    # Une ligne par personne ; les IMC manquants restent présents.
    data = demo.merge(body[["SEQN", "BMXBMI"]], on="SEQN",
                      how="left", validate="one_to_one")
    data = data.merge(mortality, on="SEQN", how="left", validate="one_to_one")
    adults = data.loc[data.RIDAGEYR.ge(18)]
    eligible = adults.loc[adults.ELIGSTAT.eq(1)]
    examined = eligible.loc[eligible.WTMEC2YR.gt(0)]
    cohort = examined.loc[
        examined.MORTSTAT.isin([0, 1]) & examined.PERMTH_EXM.notna()
    ].copy()
    if cohort.empty or cohort.PERMTH_EXM.lt(0).any():
        raise ValueError("Cohorte vide ou durée de suivi négative")

    # Durée jusqu'au décès ou à la censure, depuis l'examen.
    cohort["time_months"] = cohort.PERMTH_EXM
    cohort["event"] = cohort.MORTSTAT.astype(int)
    cohort["time_5y"] = cohort.time_months.clip(upper=60)
    cohort["event_5y"] = (
        cohort.event.eq(1) & cohort.time_months.le(60)
    ).astype(int)
    # Une censure avant 5 ans implique un statut à 5 ans inconnu.
    known = cohort.event_5y.eq(1) | cohort.time_months.ge(60)
    cohort["death_by_5y"] = cohort.event_5y.astype("Int64").where(known)

    columns = [
        "SEQN", "RIDAGEYR", "RIAGENDR", "BMXBMI",
        "time_months", "event", "time_5y", "event_5y", "death_by_5y",
        "WTMEC2YR", "SDMVSTRA", "SDMVPSU",
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    cohort[columns].to_csv(temporary, index=False)
    temporary.replace(OUTPUT)
    pd.DataFrame({
        "étape": ["Sources", "Adultes", "Éligibles mortalité", "Examinés",
                  "Statut et durée disponibles"],
        "n": [len(data), len(adults), len(eligible), len(examined), len(cohort)],
    }).to_csv(OUTPUT.parent / "cohort_flow.csv", index=False)
    print(f"Cohorte : {len(cohort)} personnes")
    print(f"Décès à 5 ans : {cohort.event_5y.sum()}")
    print(f"IMC manquants : {cohort.BMXBMI.isna().sum()}")
    print(f"CSV enregistré : {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true",
                        help="Recalculer le CSV avec les fichiers bruts disponibles")
    args = parser.parse_args()
    prepare_data(rebuild=args.rebuild)
