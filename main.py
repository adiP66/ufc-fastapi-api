import pandas as pd 
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from autogluon.tabular import TabularPredictor
import production_feature_pipeline as pipeline
import os 
from contextlib import asynccontextmanager
import traceback
import logging
import requests
from bs4 import BeautifulSoup
import json
from thefuzz import process
from rapidfuzz import process
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_PATH = "autogluon_wfv_2025"
DATA_PATH = "ufc_fights_full_with_odds.csv"

predictor = None
historical_df = None
all_fighter_names = []
all_events_dict = {}


#We inherit from the BaseModel as it is a class that defines the structure and validation requirements for our data, it automaticaly detects and validates incoming data, its primray function is data integrity
class FightRequest(BaseModel):
    fighter_one : str = Field(..., description='Fighter one name')
    fighter_two : str =  Field(..., description='Fighter two name')
    weight_class : str = "Lightweight" #Default weight class if not provided
    fighter_one_odds: Optional[int] = Field(None, description='fighter one vegas odds')
    fighter_two_odds: Optional[int] = Field(None, description='fighter two vegas odds')




@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, historical_df, all_fighter_names, all_events_dict

    if os.path.exists(MODEL_PATH):
        try:
            predictor = TabularPredictor.load(MODEL_PATH)
            print("Autogluon model loaded")
        except Exception as e:
            print(f"Failed to load model: {e}")
    else:
        print(f"Model folder not found on {MODEL_PATH}")

    if os.path.exists(DATA_PATH):
        historical_df = pd.read_csv(DATA_PATH)
        historical_df['event_date'] = pd.to_datetime(historical_df['event_date'])
        fighter_names = pd.concat([historical_df['fighter_a_name'], historical_df['fighter_b_name']], ignore_index=True)
        all_fighter_names = fighter_names.unique().tolist()

        url = "http://ufcstats.com/statistics/events/completed?page=all"

        response = requests.get(url)

        soup = BeautifulSoup(response.text, 'html.parser')
        all_events = soup.find_all('a', class_='b-link b-link_style_black')


        for event in all_events:
            event_name = event.text.strip()
            event_href = event.get('href')
            all_events_dict[event_name] = event_href
        
        print(f"Historical Data Loaded: {len(historical_df)} fights")
    else:
        print(f"Data path not found on {DATA_PATH}")

    

    yield

    print("API shutting down")

app = FastAPI(title="UFC prediction API", lifespan=lifespan)



@app.post("/predict")
async def predict(fight: FightRequest):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if historical_df is None:
        raise HTTPException(status_code=500, detail='Dataset not loaded')
    
    result_fighter_one = process.extractOne(fight.fighter_one, all_fighter_names)
    fighter_one = result_fighter_one[0]
    result_fighter_two = process.extractOne(fight.fighter_two, all_fighter_names)
    fighter_two = result_fighter_two[0]

    # Filter to only fights involving these two fighters (much smaller dataset)
    fighter_mask = (
        (historical_df['fighter_a_name'] == fighter_one) |
        (historical_df['fighter_b_name'] == fighter_one) |
        (historical_df['fighter_a_name'] == fighter_two) |
        (historical_df['fighter_b_name'] == fighter_two)
    )
    
    # Get fights for these fighters + some recent context fights
    fighter_fights = historical_df[fighter_mask].copy()
    
    if len(fighter_fights) == 0:
        raise HTTPException(status_code=404, detail=f"No fight history found for {fighter_one} or {fighter_two}")
    
    # Add recent fights for opponent context (last 500 fights for weight class priors)
    recent_fights = historical_df.tail(500).copy()
    
    # Combine and dedupe
    subset_df = pd.concat([fighter_fights, recent_fights]).drop_duplicates(subset=['fight_id'])
    
    # Create hypothetical row for prediction
    hypothetical_row = {
        "fight_id": "DUMMY-0001",
        "event_date": pd.Timestamp.today().normalize().strftime("%Y-%m-%d"),
        "weight_class": fight.weight_class,
        "outcome": np.nan,
        "fighter_a_name": fighter_one,
        "fighter_b_name": fighter_two,
        "method": "Decision - Unanimous",
        "A_open_odds": fight.fighter_one_odds if fight.fighter_one_odds is not None else np.nan,
        "B_open_odds": fight.fighter_two_odds if fight.fighter_two_odds is not None else np.nan,
        "opening_odds_diff": (fight.fighter_one_odds - fight.fighter_two_odds) if (fight.fighter_one_odds is not None and fight.fighter_two_odds is not None) else np.nan,
        "implied_prob_A": (1 / abs(fight.fighter_one_odds)) if (fight.fighter_one_odds is not None and fight.fighter_one_odds != 0) else np.nan
    }
    
    # Add dummy stats (pipeline will compute from history)
    for prefix in ['fighter_a_', 'fighter_b_']:
        hypothetical_row.update({
            f"{prefix}age": 30.0,
            f"{prefix}height": 70.0,
            f"{prefix}reach": 72.0,
            f"{prefix}weight": 155.0,
            f"{prefix}total_fights": 10.0,
            f"{prefix}win_percentage": 0.5,
            f"{prefix}recent_wins": 2.0,
            f"{prefix}win_streak": 1.0,
            f"{prefix}sig_strikes_landed_per_min": 3.0,
            f"{prefix}sig_strikes_absorbed_per_min": 3.0,
            f"{prefix}sig_strike_accuracy": 0.45,
            f"{prefix}sig_strike_defense": 0.55,
            f"{prefix}takedowns_landed_per_fight": 1.0,
            f"{prefix}takedown_attempts_per_fight": 2.5,
            f"{prefix}takedown_defense": 0.60,
            f"{prefix}submission_attempts_per_fight": 0.5,
            f"{prefix}ko_tko_win_rate": 0.30,
            f"{prefix}control_time_per_fight": 120.0,
            f"{prefix}reversals": 0.1,
            f"{prefix}opponent_reversals": 0.1,
            f"{prefix}round1_control_time": 30.0,
            f"{prefix}round1_opponent_control_time": 30.0,
            f"{prefix}round1_reversals": 0.0,
            f"{prefix}round1_opponent_reversals": 0.0,
            f"{prefix}round1_sig_strikes_landed": 10.0,
            f"{prefix}round1_sig_strikes_attempted": 20.0,
            f"{prefix}round1_opponent_sig_strikes_landed": 10.0,
            f"{prefix}head_strikes_landed": 15.0,
            f"{prefix}head_strikes_attempted": 35.0,
            f"{prefix}body_strikes_landed": 5.0,
            f"{prefix}body_strikes_attempted": 10.0,
            f"{prefix}leg_strikes_landed": 5.0,
            f"{prefix}leg_strikes_attempted": 10.0,
            f"{prefix}distance_strikes_landed": 12.0,
            f"{prefix}distance_strikes_attempted": 25.0,
            f"{prefix}clinch_strikes_landed": 3.0,
            f"{prefix}clinch_strikes_attempted": 6.0,
            f"{prefix}ground_strikes_landed": 3.0,
            f"{prefix}ground_strikes_attempted": 6.0,
            f"{prefix}opponent_head_strikes_landed": 15.0,
            f"{prefix}opponent_body_strikes_landed": 5.0,
            f"{prefix}opponent_leg_strikes_landed": 5.0,
            f"{prefix}opponent_distance_strikes_landed": 12.0,
            f"{prefix}opponent_clinch_strikes_landed": 3.0,
            f"{prefix}opponent_ground_strikes_landed": 3.0,

        })

    temp_df = pd.concat([subset_df, pd.DataFrame([hypothetical_row])], ignore_index=True)
    # Debug: Print columns before pipeline
    logger.info(f"Columns in temp_df before pipeline: {list(temp_df.columns)}")

    try:
        print("DEBUGGING COLUMNS BEFORE PIPELINE")
        print(sorted(temp_df.columns.to_list()))
        missing = [
            "opening_odds_diff",
            "implied_prob_A",
            "A_open_odds",
            "B_open_odds"
        ]

        print(" REQUIRED ODDS COLUMNS")
        for col in missing:
            print(col, '->', col in temp_df.columns)


        processed_df, feature_cols = pipeline.build_prefight_features(temp_df)
        input_features = processed_df[processed_df["fight_id"] == "DUMMY-0001"]
        
        if input_features.empty:
            raise HTTPException(status_code=400, detail='Pipeline filtered out this matchup. Fighters may have <2 UFC fights.')

        model_input = input_features
        prediction_class = predictor.predict(model_input).iloc[0]
        prediction_proba = predictor.predict_proba(model_input).iloc[0]

        prob_fighter1_win = prediction_proba[1]
        prob_fighter2_win = prediction_proba[0]

        return {
            "matchup": f"{fighter_one} vs {fighter_two}",
            "predicted_winner": fighter_one if prediction_class == 1 else fighter_two,
            "confidence": f"{max(prob_fighter1_win, prob_fighter2_win):.2%}",
            "fighter_one_prob": float(prob_fighter1_win),
            "fighter_two_prob": float(prob_fighter2_win),
            "features_used": len(feature_cols),
            "fights_in_context": len(subset_df)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")
    

@app.get('/get-fighter-stats/{fighter_name}')
async def get_fighter_info(fighter_name):
    result = process.extractOne(fighter_name, all_fighter_names)
    if result[1] > 50: 
        fighter_name = result[0]
        mask = (historical_df['fighter_a_name'] == fighter_name) | (historical_df['fighter_b_name'] == fighter_name)
        fighter_name_df =  historical_df[mask][['fight_id', 'fighter_a_name', 'fighter_b_name']]
        print(fighter_name_df)
        if fighter_name in fighter_name_df['fighter_a_name'].to_list() or fighter_name_df['fighter_b_name'].to_list():
            fight_id = fighter_name_df['fight_id'].to_list()[-1]
            url = f"http://ufcstats.com/fight-details/{fight_id}"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            fighter_details = soup.find_all("a", class_='b-link b-fight-details__person-link')
            for fighter in fighter_details:
                if fighter.text.strip() == fighter_name:
                    fighter_detail_link = fighter.get('href')
                    response = requests.get(fighter_detail_link)        
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # career_stats = soup.find_all('div', class_='b-list__info-box-left clearfix')
                    career_stats = soup.find_all('li', class_='b-list__box-list-item')
                    stats_dict = {}
                    for stat in career_stats:
                        stats_dict[stat.text.replace('\n', '').strip().split(":")[0]] = stat.text.replace('\n', '').split(":")[-1].strip()
                    filtered_dict = {k:v for k,v in stats_dict.items() if k != '' and v != ''}    
                    filtered_dict['fighter_name'] = fighter_name
                    return filtered_dict
    else:
        raise HTTPException(status_code =404, detail=f'Fighter with name "{fighter_name}" not found, Try again')

@app.get('/get-all-events')
async def get_event_info():    
    return all_events_dict

@app.get("/numbered-event-details/{numbered_event}")
async def get_numbered_event_info(numbered_event):
    url = None
    for k,v in all_events_dict.items():
        event = k.split(":")[0]
        event_number=  event.split()[-1]
        if event_number.isdigit() and int(event_number) == int(numbered_event):
            url = v
        else:
            raise HTTPException(status_code =404, detail=f'Event {numbered_event} not found')
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table_rows = soup.find_all('tr', class_='b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click')
    event_fights = {}
    for row in table_rows:
        each_fight_link = row.find_all('a')[0].get('href')
        each_fight = row.find_all('a', class_= 'b-link b-link_style_black')

        fighter_a = each_fight[0].text.strip()
        fighter_b = each_fight[1].text.strip()
        event_fights['numbered event'] = f"UFC {numbered_event}"
        event_fights[f"{fighter_a} vs {fighter_b}"] = each_fight_link

    return event_fights

@app.get('/fights')
async def get_all_fights():
    fights = historical_df.drop_duplicates()
    fights
    all_fights = fights[['fight_id', 'fighter_a_name', 'fighter_b_name', 'outcome']]
    return all_fights.to_dict(orient='records')


@app.get('/fighters')
async def get_all_fighters():
    return all_fighter_names


@app.get('/fighter/{fighter_name}/fights')
async def get_fighter_fights(fighter_name):
    fuzzo = process.extractOne(fighter_name, all_fighter_names)
    if fuzzo[1] > 60:
        actual_name = fuzzo[0]
        mask = (historical_df['fighter_a_name'] == actual_name) | (historical_df['fighter_b_name'] == actual_name)
        new_df = historical_df[mask]
        return new_df.fillna(0).to_dict(orient='records')    
    else:
        raise HTTPException(status_code=404, detail=f'Fighter with name {fighter_name} not found')

    
@app.get('/fights/{id}')
async def get_fight_details(id):
    fight_details = historical_df[historical_df['fight_id'] == id]
    if fight_details.empty:
        raise HTTPException(status_code=404, detail=f'fight with id {id} not found')
    return fight_details.fillna(0).to_dict()


@app.get('/rankings/{division}')
async def get_rankings(division):
    divisions = ["Heavyweight", "Light Heavyweight", "Men's Pound-for-Pound Top Rank", "Welterweight", "Lightweight", "Middleweight",
             "Featherweight", "Bantamweight", "Flyweight", "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight"]
    division_name = process.extractOne(division, divisions)[0]
    response  = requests.get('https://gidstats.com/ranking/ufc/')
    soup = BeautifulSoup(response.text, 'html.parser')
    categories = soup.find_all('li', class_='category')

    ranking_dict = {}
    fighters_list = None

    for category in categories:
        category_name = category.find('h3', class_='category__title')
        if category_name is not None:
            category_div = category_name.text
            if category_div == division_name:
                fighters_list = category.find_all('li', class_='fighters-list__item')

    for fighter in fighters_list:
        ranking = fighter.find('p', class_='number').text.strip()
        fighter_name = fighter.find('p', class_='name').text.strip()
        fighter_name = fighter_name.replace('Champion', '')
        fighter_name = process.extractOne(fighter_name, all_fighter_names)[0]
        if ranking == '':
            ranking_dict[fighter_name] = 'Champion'
        else:
            ranking_dict[fighter_name] = ranking


    return ranking_dict