max_clauses(6).
max_vars(5).
max_body(6). 
head_pred(complication,1).

body_pred(recentnecksurgery,2).
body_pred(fixation_level,2).

body_pred(diagnostic,2).
body_pred(lemonscore,2).
body_pred(intubation_duration,2).
body_pred(age,2).

body_pred(indication,2).
body_pred(number_probes,2).
body_pred(other_conditions,2).
body_pred(number_attempts,2).
body_pred(cormack,2).
type(complication,(patient,)).

type(recentnecksurgery,(patient,real)).
type(fixation_level,(patient,real)).

type(diagnostic,(patient,real)).

type(lemonscore,(patient,real)).
type(intubation_duration,(patient,real)).
type(age,(patient,real)).

type(indication,(patient,real)).
type(number_probes,(patient,real)).
type(other_conditions,(patient,real)).
type(number_attempts,(patient,real)).
type(cormack,(patient,real)).
direction(complication,(in,)).


direction(recentnecksurgery,(in,out)).
direction(fixation_level,(in,out)).
direction(diagnostic,(in,out)).
direction(lemonscore,(in,out)).
direction(intubation_duration,(in,out)).
direction(age,(in,out)).

direction(indication,(in,out)).
direction(number_probes,(in,out)).
direction(other_conditions,(in,out)).
direction(number_attempts,(in,out)).

direction(cormack,(in,out)).
:- clause(C), #count{V : var_type(C,V,patient)} != 1.