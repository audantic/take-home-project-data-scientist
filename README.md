# README


## 1 Setup

### 1.1 Set up the repo for the Jupyter Notebook

Clone this repository.

Start by cloning the repo, to create a local copy of this folder in your machine.

In the terminal:

```
cd ~/Documents/

git clone https://github.com/audantic/take-home-project-data-analyst.git

cd take-home-project-data-analyst
```

### 1.2 Set up Docker

Install Docker: https://www.docker.com/community-edition#/download


This may take a while depending on download speeds.

```
docker pull jupyter/scipy-notebook
```

### 1.3 Start the notebook

```
docker run -it --rm -p 8888:8888 -v $(pwd):/home/jovyan/work jupyter/scipy-notebook
```

In the terminal, it will say something like:

```bash
$ docker run -it --rm -p 8888:8888 -v $(pwd):/home/jovyan/work jupyter/scipy-notebook
/usr/local/bin/start-notebook.sh: ignoring /usr/local/bin/start-notebook.d/*

Container must be run with group root to update passwd file
Executing the command: jupyter notebook
[I 20:38:18.195 NotebookApp] The Jupyter Notebook is running at:
[I 20:38:18.195 NotebookApp] http://[all ip addresses on your system]:8888/?token=c08dd7b6e98ac6658203af4426626e1ae2514cdb8c60c864
[I 20:38:18.195 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 20:38:18.196 NotebookApp] 
    
    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://localhost:8888/?token=c08dd7b6e98ac6658203af4426626e1ae2514cdb8c60c864
```

Copy/paste this URL into your browser.

Once the notebook is running, go into the work directory and open the "project.ipynb"



## 2 Dataset

The purpose of the task is to demonstrate your skillset, and how you would approach a prediction problem.

We would like you to build a predictive model for a given house selling, given the data set and using Python or R.

This can be a quick and dirty model.

Approximately 15% of the rows have audantic_target=1, so the dataset is unbalanced.

Columns | Explanation
--------|-------------
audantic_target | response var, y
pid | property id, ignore for modeling
did | document id, ignore for modeling
fips | categorical var, county
zipcode | categorical var, zipcode
seller_occupied | categorical var, if the seller lives in the house
square_footage | categorical var, square footage
year_built | categorical var, year built
estimated_value | categorical var, estimated value of the house
length_of_ownership | categorical var, how long the property was owned for
est_household_income_val | continuous var, income of the homeowners
mosaic_hh_val | continuous var, demographic descriptor
mosaic_zip4_val | continuous var, demographic descriptor of neighborhood
mosaic_diff | categorical var, difference between home and neighborhood
cat_a | categorical var
cat_ce | categorical var
cat_a_c | categorical var
cat_a_i | categorical var
cat_a_j | categorical var
cat_a_m | categorical var
cat_a_sk | categorical var
cat_e_b | categorical var
cat_e_e | categorical var
cat_e | categorical var
cat_e_r | categorical var
cat_e_s | categorical var
cat_g | categorical var
cat_h | categorical var
cat_j | categorical var
cat_m | categorical var
cat_n | categorical var
cat_s | categorical var


Data sample:

audantic_target|pid|did|fips|zipcode|seller_occupied|square_footage|year_built|estimated_value|length_of_ownership|est_household_income_val|mosaic_hh_val|mosaic_zip4_val|mosaic_diff|cat_a|cat_ce|cat_a_c|cat_a_i|cat_a_j|cat_a_m|cat_a_sk|cat_e_b|cat_e_e|cat_e|cat_e_r|cat_e_s|cat_g|cat_h|cat_j|cat_m|cat_n|cat_s
-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-
0|369|603808466|36103|11946.0|0|1260|1997|418000|13.0294|0|72|13|-59|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0
0|383|574522015|36103|11767.0|1|0|0|301000|10.7515|87500|11|16|5|0.0|1.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0
0|960|605013654|36103|11704.0|1|0|0|263000|9.7029|137500|22|18|-4|1.0|1.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0
0|973|605452151|36103|11776.0|0|0|0|412000|20.0|225000|17|16|-1|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0
0|1136|602447312|36055|14586.0|1|1363|2000|172000|5.0021|62500|15|15|0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0
0|1243|608617670|36029|14150.0|0|1640|1939|114000|13.473|62500|66|66|0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0
0|1494|605007197|36059|11553.0|1|1344|1957|397000|11.3402|87500|18|18|0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0
0|1527|571263516|36059|11580.0|0|907|1953|370000|4.3094|300000|10|10|0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0
0|1565|569012570|36059|11590.0|1|1165|1958|401000|0.2053|112500|18|18|0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0