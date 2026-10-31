import os
import re
import janus_swi as janus
from popper.util import Settings, format_rule
from popper.loop import learn_solution

TRAIN_FOLDER = "examples/complication1"
TEST_FOLDER = "examples/testcomplication"
RULES_FILE = "learned_rules.pl"
BK_FILE = os.path.join(TEST_FOLDER, "bk.pl")
EXS_FILE = os.path.join(TEST_FOLDER, "exs.pl")


def save_rules_to_file(prog, path=RULES_FILE):
    with open(path, "w") as f:
        for rule in prog:
            f.write(format_rule(rule) + ".\n")


def extract_truths(path):
    pos_patients, neg_patients = set(), set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("pos("):
                m = re.match(r"pos\(complication\(([^)]+)\)\)\.", line)
                if m:
                    pos_patients.add(m.group(1))
            elif line.startswith("neg("):
                m = re.match(r"neg\(complication\(([^)]+)\)\)\.", line)
                if m:
                    neg_patients.add(m.group(1))
    return pos_patients, neg_patients


def evaluate():
    print("\n********** TEST PAR PATIENT **********")

    # Nettoyage
    janus.query_once("retractall(complication(_))")
    janus.query_once("retractall(pos(_))")
    janus.query_once("retractall(neg(_))")

    # Charger le contexte de test
    janus.consult(BK_FILE)
    janus.consult(EXS_FILE)

    # Injecter les règles induites
    with open(RULES_FILE) as f:
        rules = f.read()
    janus.consult("prog", rules)

    # Vérités terrain
    pos_patients, neg_patients = extract_truths(EXS_FILE)
    all_patients = sorted(list(pos_patients | neg_patients))

    # Évaluation
    tp = tn = fp = fn = 0
    for p in all_patients:
        q = janus.query_once("complication(P)", {"P": p})
        prediction = q["truth"]
        label = "pos" if p in pos_patients else "neg"
        correct = (prediction and label == "pos") or (not prediction and label == "neg")

        if prediction and label == "pos":
            tp += 1
        elif not prediction and label == "neg":
            tn += 1
        elif prediction and label == "neg":
            fp += 1
        elif not prediction and label == "pos":
            fn += 1

        print(f"{p} → Prédit: {prediction}, Vérité: {label} → {'✅ Correct' if correct else '❌ Incorrect'}")

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)

    print("\n✨ Métriques Globales :")
    print(f"TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall (Sensibilité): {recall:.2f}")
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    print("\n********** TRAINING SOLUTION **********")
    settings = Settings(kbpath=TRAIN_FOLDER, cmd_line=False, timeout=600, show_stats=True)
    prog, score, stats = learn_solution(settings)

    if prog is not None:
        settings.print_prog_score(prog, score)
        save_rules_to_file(prog)
        evaluate()
    else:
        print("NO SOLUTION")
