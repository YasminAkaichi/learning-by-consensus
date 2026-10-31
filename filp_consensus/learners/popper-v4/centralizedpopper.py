import os
import re
import janus_swi as janus
from popper.util import Settings, format_rule
from popper.loop import learn_solution

# === FICHIERS ===
TRAIN_KB = "traincomplication"
TEST_FOLDER = "examples/testcomplication"
BK_FILE = os.path.join(TEST_FOLDER, "bk.pl")
EXS_FILE = os.path.join(TEST_FOLDER, "exs.pl")

def extract_truths(path):
    pos, neg = set(), set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("pos("):
                m = re.match(r"pos\(complication\(([^)]+)\)\)\.", line)
                if m:
                    pos.add(m.group(1))
            elif line.startswith("neg("):
                m = re.match(r"neg\(complication\(([^)]+)\)\)\.", line)
                if m:
                    neg.add(m.group(1))
    return pos, neg

def train():
    print("\n********** APPRENTISSAGE **********")
    settings = Settings(
        kbpath=TRAIN_KB,
        cmd_line=False,
        timeout=600,
        show_stats=True,
        debug=False,
    )
    prog, score, stats = learn_solution(settings)
    settings.print_prog_score(prog, score)
    return prog

def evaluate(prog):
    print("\n********** TEST PAR PATIENT **********")

    # Nettoyage état Prolog
    janus.query_once("retractall(complication(_))")
    janus.query_once("retractall(pos(_))")
    janus.query_once("retractall(neg(_))")

    # Charger la base de connaissance + exemples
    janus.consult(BK_FILE)
    janus.consult(EXS_FILE)

    # Injecter les règles
    rules_str = ":- dynamic complication/1.\n"
    rules_str += "\n".join(format_rule(r) for r in prog)
    janus.consult("prog", rules_str)

    # Vérité terrain
    pos_patients, neg_patients = extract_truths(EXS_FILE)
    all_patients = sorted(list(pos_patients | neg_patients))

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

    # Affichage métriques
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)

    print("\n✨ Métriques Globales :")
    print(f"TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall (Sensibilité): {recall:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    prog = train()
    evaluate(prog)
