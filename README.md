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

<img width="1348" height="1126" alt="image" src="https://github.com/user-attachments/assets/f4bd6afb-13a1-4ac8-9b28-f0da70b97de1" />

In a traditional way:
<img width="2557" height="1344" alt="image" src="https://github.com/user-attachments/assets/6f8dfc6f-2797-4ea7-8b76-0c9c34b347f8" />
