import janus_swi as janus
import os
import re

list_patient_to_test = [
    "patient90", "patient86", "patient52", "patient22", "patient27",
    "patient60", "patient65", "patient107", "patient34", "patient96", "patient57"
]

def parse_ground_truth(filepath):
    real_pos, real_neg = [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if match := re.match(r'pos\(complication\((patient\d+)\)\)\.', line):
                real_pos.append(match.group(1))
            elif match := re.match(r'neg\(complication\((patient\d+)\)\)\.', line):
                real_neg.append(match.group(1))
    return real_pos, real_neg

def load_learned_rules(file="learned_rules.pl"):
    with open(file) as f:
        return f.read()

def evaluate_individual_patients(rules_str, test_bk_path, test_ex_path):
    print("\n********** TEST PAR PATIENT **********")

    # Ground truth
    real_pos, real_neg = parse_ground_truth(test_ex_path)

    # Charger BK + règles
    janus.consult(test_bk_path)
    janus.consult(test_ex_path)
    janus.consult("prog", rules_str)

    results = []
    for patient in list_patient_to_test:
        pred = janus.query_once("complication(P)", {"P": patient}).get("truth", False)
        real = (
            "pos" if patient in real_pos else
            "neg" if patient in real_neg else
            "unknown"
        )

        status = "✅ Correct" if (
            (pred and real == "pos") or (not pred and real == "neg")
        ) else "❌ Incorrect"

        print(f"{patient} → Prédit: {pred}, Vérité: {real} → {status}")
        results.append((patient, pred, real, status))

    return results

if __name__ == "__main__":
    rules = load_learned_rules("learned_rules.pl")
    test_bk = "test0/bk.pl"
    test_ex = "test0/exs.pl"
    evaluate_individual_patients(rules, test_bk, test_ex)
