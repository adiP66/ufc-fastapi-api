Production grade ML UFC prediction API project with built in real-time data scraping from ufcstats and other websites. 
This project covers
- Fight outcome prediction
- Event intelligence
- Ranking aggregation
- Data Scraping
- ML feature engineering

Anyone can use this api in thier project directly to fetch data ranging from ufc 2 (1998) to the latest event as the dataset is constantly being updated.
Cheers.

Tech Stack 
API - FASTAPI
ML - Autogluon
Data - Pandas, numpy
scraping - requests, bs4
matching - thefuzz, rapidfuzz
pipeline - custom feature engineering

## IMPORTANT NOTE 
One thing to note is that the model was trained on a linux environment and THEREFORE the api/model can only be run on a linux environment, if you are on Windows I strongly suggest you use WSL.
Python version used for the autogluon mdoel is 3.12.12 (altogether different production) , even though I have used 3.12.5 for this uploaded project.

ROUTES
<img width="1392" height="1225" alt="image" src="https://github.com/user-attachments/assets/faccb520-7ce6-4bfa-ba4e-99935902b793" />
