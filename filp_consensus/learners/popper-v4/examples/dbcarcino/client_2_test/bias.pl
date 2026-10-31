max_clauses(6).
max_vars(5).
max_body(6).

%% Head predicate
head_pred(carcinogen,1).
type(carcinogen,(chemical,)).

%% Body predicates
body_pred(atom_br,2).
body_pred(atom_c,2).
body_pred(atom_cl,2).
body_pred(atom_h,2).
body_pred(atom_i,2).
body_pred(atom_n,2).
body_pred(atom_o,2).
body_pred(atom_p,2).
body_pred(atom_s,2).
body_pred(avg_charge,2).
body_pred(bond_sbond_1_count,2).
body_pred(bond_sbond_2_count,2).
body_pred(bond_sbond_3_count,2).
body_pred(bond_sbond_7_count,2).
body_pred(std_charge,2).

%% Type declarations
type(chemical).
type(value).
type(atom_br,(chemical,value)).
type(atom_c,(chemical,value)).
type(atom_cl,(chemical,value)).
type(atom_h,(chemical,value)).
type(atom_i,(chemical,value)).
type(atom_n,(chemical,value)).
type(atom_o,(chemical,value)).
type(atom_p,(chemical,value)).
type(atom_s,(chemical,value)).
type(avg_charge,(chemical,value)).
type(bond_sbond_1_count,(chemical,value)).
type(bond_sbond_2_count,(chemical,value)).
type(bond_sbond_3_count,(chemical,value)).
type(bond_sbond_7_count,(chemical,value)).
type(std_charge,(chemical,value)).