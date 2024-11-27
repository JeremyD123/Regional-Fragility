# Regional-fragility
Seismic fragility assessment of regional building portfolios using machine learning and Poisson binomial distribution

Building damage database: InputData/Regional Building Damage.xlsx

Data Preprocessing Program - Preprocessing.py
Path_intput - database path
Path_output - saved path
X_train - training set input
Y_train - training set output
X_test - testing set input
Y_test - testing set output

Probabilistic machine learning models predict the unsafety probability of each building - NGB_Pre.py
prob_US_build - extensive probability of each building

Poisson binomial distribution calculates the proportion of unsafety probability for building portfolio (i.e., regional fragility) - poibin_for_Fra.py
p_unsafe - the expected proportion of unsafe buildings in the city
