\ENCODING=ISO-8859-1
\Problem name: 

Maximize
 obj: 0.25 V_0 + 0.25 V_1 + 0.25 V_2 + 0.25 V_3
Subject To
 c1:  - Q_0_0 + 0.233052534610033 V_0 + 0.716947451233864 V_3 + r_0_0  = 0
 c2:  - Q_1_0 + 0.0591641059145331 V_0 + 0.890835869312286 V_2 + r_1_0  = 0
 c3:  - Q_2_0 + 0.651487365365028 V_1 + 0.298512634634972 V_3 + r_2_0  = 0
 c4:  - Q_3_0 + 0.705493804812431 V_0 + 0.244506223499775 V_3 + r_3_0  = 0
 c5:  - Q_0_1 + 0.553029119968414 V_1 + 0.396970908343792 V_2 + r_0_1  = 0
 c6:  - Q_1_1 + 0.639633893966675 V_0 + 0.310366134345531 V_3 + r_1_1  = 0
 c7:  - Q_2_1 + 0.412500181794167 V_0 + 0.537499761581421 V_2 + r_2_1  = 0
 c8:  - Q_3_1 + 0.306794323027134 V_1 + 0.643205705285072 V_2 + r_3_1  = 0
 c9:  - Q_0_2 + 0.388765549659729 V_0 + 0.561234506964683 V_1 + r_0_2  = 0
 c10: - Q_1_2 + 0.875844669342041 V_0 + 0.0741553377360105 V_3 + r_1_2  = 0
 c11: - Q_2_2 + 0.166596934944391 V_0 + 0.783403050899506 V_3 + r_2_2  = 0
 c12: - Q_3_2 + 0.732404953241348 V_0 + 0.217595018446445 V_2 + r_3_2  = 0
 c13: - Q_0_3 + 0.532131540775299 V_2 + 0.417868459224701 V_3 + r_0_3  = 0
 c14: - Q_1_3 + 0.410478265583515 V_1 + 0.539521706104279 V_3 + r_1_3  = 0
 c15: - Q_2_3 + 0.474224981665611 V_0 + 0.475774961709976 V_2 + r_2_3  = 0
 c16: - Q_3_3 + 0.637062636017799 V_0 + 0.312937335669994 V_3 + r_3_3  = 0
 c17: - Q_0_0 + V_0 >= 0
 c18: - Q_1_0 + V_1 >= 0
 c19: - Q_2_0 + V_2 >= 0
 c20: - Q_3_0 + V_3 >= 0
 c21: - Q_0_1 + V_0 >= 0
 c22: - Q_1_1 + V_1 >= 0
 c23: - Q_2_1 + V_2 >= 0
 c24: - Q_3_1 + V_3 >= 0
 c25: - Q_0_2 + V_0 >= 0
 c26: - Q_1_2 + V_1 >= 0
 c27: - Q_2_2 + V_2 >= 0
 c28: - Q_3_2 + V_3 >= 0
 c29: - Q_0_3 + V_0 >= 0
 c30: - Q_1_3 + V_1 >= 0
 c31: - Q_2_3 + V_2 >= 0
 c32: - Q_3_3 + V_3 >= 0
 c33: - Q_0_0 + V_0 + 100000000 I_0_0 <= 100000000
 c34: - Q_1_0 + V_1 + 100000000 I_1_0 <= 100000000
 c35: - Q_2_0 + V_2 + 100000000 I_2_0 <= 100000000
 c36: - Q_3_0 + V_3 + 100000000 I_3_0 <= 100000000
 c37: - Q_0_1 + V_0 + 100000000 I_0_1 <= 100000000
 c38: - Q_1_1 + V_1 + 100000000 I_1_1 <= 100000000
 c39: - Q_2_1 + V_2 + 100000000 I_2_1 <= 100000000
 c40: - Q_3_1 + V_3 + 100000000 I_3_1 <= 100000000
 c41: - Q_0_2 + V_0 + 100000000 I_0_2 <= 100000000
 c42: - Q_1_2 + V_1 + 100000000 I_1_2 <= 100000000
 c43: - Q_2_2 + V_2 + 100000000 I_2_2 <= 100000000
 c44: - Q_3_2 + V_3 + 100000000 I_3_2 <= 100000000
 c45: - Q_0_3 + V_0 + 100000000 I_0_3 <= 100000000
 c46: - Q_1_3 + V_1 + 100000000 I_1_3 <= 100000000
 c47: - Q_2_3 + V_2 + 100000000 I_2_3 <= 100000000
 c48: - Q_3_3 + V_3 + 100000000 I_3_3 <= 100000000
 c49: I_0_0 + I_0_1 + I_0_2 + I_0_3  = 1
 c50: I_1_0 + I_1_1 + I_1_2 + I_1_3  = 1
 c51: I_2_0 + I_2_1 + I_2_2 + I_2_3  = 1
 c52: I_3_0 + I_3_1 + I_3_2 + I_3_3  = 1
Bounds
      Q_0_0 Free
      Q_0_1 Free
      Q_0_2 Free
      Q_0_3 Free
      Q_1_0 Free
      Q_1_1 Free
      Q_1_2 Free
      Q_1_3 Free
      Q_2_0 Free
      Q_2_1 Free
      Q_2_2 Free
      Q_2_3 Free
      Q_3_0 Free
      Q_3_1 Free
      Q_3_2 Free
      Q_3_3 Free
      V_0 Free
      V_1 Free
      V_2 Free
      V_3 Free
 0 <= I_0_0 <= Inf
 0 <= I_0_1 <= Inf
 0 <= I_0_2 <= Inf
 0 <= I_0_3 <= Inf
 0 <= I_1_0 <= Inf
 0 <= I_1_1 <= Inf
 0 <= I_1_2 <= Inf
 0 <= I_1_3 <= Inf
 0 <= I_2_0 <= Inf
 0 <= I_2_1 <= Inf
 0 <= I_2_2 <= Inf
 0 <= I_2_3 <= Inf
 0 <= I_3_0 <= Inf
 0 <= I_3_1 <= Inf
 0 <= I_3_2 <= Inf
 0 <= I_3_3 <= Inf
-1 <= r_0_0 <= 1
-1 <= r_0_1 <= 1
-1 <= r_0_2 <= 1
-1 <= r_0_3 <= 1
-1 <= r_1_0 <= 1
-1 <= r_1_1 <= 1
-1 <= r_1_2 <= 1
-1 <= r_1_3 <= 1
-1 <= r_2_0 <= 1
-1 <= r_2_1 <= 1
-1 <= r_2_2 <= 1
-1 <= r_2_3 <= 1
-1 <= r_3_0 <= 1
-1 <= r_3_1 <= 1
-1 <= r_3_2 <= 1
-1 <= r_3_3 <= 1
Binaries
 I_0_0  I_0_1  I_0_2  I_0_3  I_1_0  I_1_1  I_1_2  I_1_3  I_2_0  I_2_1  I_2_2 
 I_2_3  I_3_0  I_3_1  I_3_2  I_3_3 
End
