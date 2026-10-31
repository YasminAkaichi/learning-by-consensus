from popper.util import Settings, format_rule
from popper.loop import learn_solution
from popper.tester import Tester
from janus_swi import query_once

# ----------- ÉTAPE 1 : APPRENTISSAGE SUR COMPILATION1 -----------
print("\n********** APPRENTISSAGE **********")
train_settings = Settings(
    kbpath='traincomplication',
    cmd_line=False,
    timeout=600,
    show_stats=True,
    debug=False,
)

prog, score, stats = learn_solution(train_settings)

if prog is None:
    print("NO SOLUTION")
    exit()

train_settings.print_prog_score(prog, score)

# Sauvegarde optionnelle du programme appris
with open("learned_rules.pl", "w") as f:
    for rule in prog:
        f.write(format_rule(rule) + "\n")

# ----------- ÉTAPE 2 : ÉVALUATION SUR TESTCOMPILATION -----------
print("\n********** TEST **********")
test_settings = Settings(
    kbpath='examples/testcomplication',
    cmd_line=False,
    timeout=300,
    show_stats=False,
    debug=False,
)

tester = Tester(test_settings)

# Prédictions avec le programme appris
preds, pos_ids, neg_ids = tester.predict_labels(prog)

# Évaluation
print("\n🔍 Évaluation sur examples/testcomplication :")
tp = sum(1 for x in pos_ids if preds.get(x, False))
fn = sum(1 for x in pos_ids if not preds.get(x, False))
fp = sum(1 for x in neg_ids if preds.get(x, False))
tn = sum(1 for x in neg_ids if not preds.get(x, False))

precision = tp / (tp + fp + 1e-9)
recall = tp / (tp + fn + 1e-9)
accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)

print(f"TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, Accuracy: {accuracy:.2f}")
