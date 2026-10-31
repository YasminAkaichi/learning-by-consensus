#this is working perfectly + accuracies etC. 
import janus_swi as janus
import os
import re

RULES_FILE = "learned_rules.pl"
TEST_FOLDER = "examples/testcomplication"
BK_FILE = os.path.join(TEST_FOLDER, "bk.pl")
EXS_FILE = os.path.join(TEST_FOLDER, "exs.pl")


def load_rules(path):
    with open(path) as f:
        return f.read()


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

    # Clean state
    janus.query_once("retractall(complication(_))")
    janus.query_once("retractall(pos(_))")
    janus.query_once("retractall(neg(_))")

    # Load BK and EXS
    janus.consult(BK_FILE)
    janus.consult(EXS_FILE)

    # Inject rules
    rules = load_rules(RULES_FILE)
    janus.consult("prog", rules)

    # Ground truth
    pos_patients, neg_patients = extract_truths(EXS_FILE)
    all_patients = sorted(list(pos_patients | neg_patients))

    # Evaluation
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
    evaluate()
