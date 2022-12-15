# iInsureU123 Wanderlust Dataset
This dataset contains information about customers of insurance company iInsureU123 in relation the gene variant DRD4, known as the Wanderlust gene. The dataset has been anonymised to ensure data privacy is protected for sharing purposes, with a k-anonymity value of 15.

The customers included in the study are all Caucasian, live in the United Kingdom and have all been born overseas.

The language used in this dataset is English only.

## Files Overview
Filename structure:
```bash
Anonymised_data directory
    ├── data.zip
    │   └── data.csv
    ├── wanderlust.txt
    └── keys.zip
        ├── keytable.csv
        ├── postkeytable.csv
        └── edukeytable.csv
```

This directory includes the relevant files:
* data.zip: password protected zip folder that contains the anonymised dataset data.csv. Note that data.csv is a read-only file.
* wanderlust.txt: 20-character alphanumeric password to unlock data.zip
* keys.zip: zip folder that contains three read-only csv files that encode key-value relationship between pseudo-anonymised variables of continent of birth, postcode and education level and their raw values. The csv files are:
    * keytable.csv: maps placeholder string ('Color'; 'continent_of_birth' in data.csv) to raw continent of birth ('Continent')
    * postkeytable.csv: maps placeholder string ('ID'; 'postcode_region' in data.csv) to UK residence region ('UK Region')
    * edukeytable.csv: maps placeholder string ('Color'; 'education_level' in data.csv) to raw education level ('Education Level')

Files were created in December 2022.

## Data Description
The anonymised data file 'data.csv' is a 1000x12 dataset. 

An example of a row in the data can be found below (not real data):
| gender | banded_age | continent_of_birth | postcode_region | cc_status | banded_weight | banded_height | blood_group | banded_weekly_avg_drinks | banded_weekly_avg_cigret | education_level | n_countries_visited_grouped |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| F | (40,60] | Blue | 2 | 0 | (90,100] | (1.8, 1.9] | O+ | medium | low | Black | (30,35] |

The columns included in this dataset are:
* gender: binary column indicating M for male customers and F for female customers
* banded_age: age of customer, stored in intervals of 20 years
* continent_of_birth: region of birth of customer, pseudoanonymised to colors 'Red', 'Blue', 'Green' and 'Yellow'
* postcode_region: region of residence in the United Kingdom, pseudoanonymised to '0', '1', '2', '3', '4', '5' 
* cc_status: 0/1 binary variable indicating if customer has the DRD4 Wanderlust gene
* banded_weight: weight of customer, in intervals of 10kg
* banded_height: height of customer, in intervals of 0.1m
* blood_group: blood group of customer
* banded_weekly_avg_drinks: average weekly alcoholic drinks consumed by customer, represented as low/medium/high based on UK Health Recommendations with low being <6, medium being between 6 and 12 and high being >12.
* banded_weekly_avg_cigret: average weekly number of cigarettes smoled by customer, represented as low/medium/high based on UK Health Recommendations with low being <40 per week, medium being between 40 and 175, and high being >175 packs a week.
* education_level: education level of customer pseudoanonymised to 'Black', 'Brown', 'Grey' and 'White'
* n_countries_visited_grouped: number of countries visited by the customer, stored in intervals of 5

## Accessing Data
1. Retrieve password string from wanderlust.txt
2. Unzip data.zip that contains anonymised data, providing retrieved password string when prompted
3. Access anonymised data through the embedded data.csv file.
4. Pseudoanonymised variables can be retrieved for further analysis through the .csv files embedded in keys.zip

## Data Collection Methodology
The data was collected from iInsureU123 customers during signup to the insurance company, where customers agreed to undergo a set of tests and were asked to fill in a questionnaire with some personal details.

## Support
For additional support, please email hda_group8@ic.ac.uk. 