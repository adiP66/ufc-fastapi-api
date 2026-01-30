"""
Production Feature Engineering Pipeline V2 for UFC Fight Prediction.

STREAMLINED VERSION - Only keeps high-value feature layers:
- Ratio features (*_ratio, *_acc, *share)
- Z-score Decayed Average (*_dec_adjperf_dec_avg) ⭐ MOST PREDICTIVE LAYER
- Volume/Rate baseline (*_dec_avg)
- Strength of Schedule (sos_ewm) - Single best win metric

All 4 computational layers (career_avg, ewm_avg, zscore, zscore_dec_avg) are still 
computed internally for proper feature engineering, but only the most predictive 
layers are kept as final model features to reduce multicollinearity.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any

def build_prefight_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Main entry point to build the streamlined pre-fight feature set.
    
    Args:
        df: Raw dataframe containing per-fight stats (ufc_fights_ml_updated.csv).
            
    Returns:
        df_final: Dataframe with only pre-fight differential features and target label.
        feature_cols: List of feature column names to use for training.
    """
    print("=== STARTING PRODUCTION FEATURE PIPELINE V2 (STREAMLINED) ===")
    
    # 1. Data Cleaning & Sorting
    df = _clean_and_sort(df)
    
    # 2. Apply Strict Filters
    df = _apply_strict_filters(df)
    
    # 3. Convert to Long Format
    long_df = _convert_to_long_format(df)
    
    # 4. Compute Physical Attribute Ratios BEFORE history
    long_df = _compute_physical_ratios(long_df)
    
    # 5. Reconstruct Pre-Fight History (ALL LAYERS - internal computation)
    long_df = _compute_history_features(long_df)

    # 6. Compute Base Ratios & Per-Minute Stats
    long_df = _compute_ratio_features(long_df)
    
    # 7. Compute Opponent-Adjusted Performance (Z-Scores)
    long_df = _compute_opponent_adjusted_features(long_df)
    
    # 8. Compute Elo Ratings (NEW - high-value feature!)
    long_df = _compute_elo_ratings(long_df)
    
    # 9. Momentum - REMOVED (user requested removal to reduce feature count)
    # long_df = _compute_momentum(long_df)
    
    # 10. Merge Back & Create Differentials (SELECTIVE FEATURE FILTERING)
    df_final, feature_cols = _merge_and_create_differentials(df, long_df)
    
    # 11. Final Cleanup
    df_final = _final_cleanup(df_final, feature_cols)
    
    print(f"=== PIPELINE COMPLETE: {len(feature_cols)} features generated ===")
    print("Feature layers included: *_ratio, *_dec_adjperf_dec_avg, *_dec_avg, sos_ewm, ELO")
    return df_final, feature_cols


def _clean_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Sorts data chronologically to ensure correct historical reconstruction."""
    df = df.copy()
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values(['event_date', 'fight_id']).reset_index(drop=True)
    df = df.dropna(subset=['fighter_a_name', 'fighter_b_name', 'event_date'])
    
    # LEAKAGE GUARD: Drop pre-computed streaks/wins that might contain current fight info
    drop_cols = [c for c in df.columns if 'recent_wins' in c or 'streak' in c]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"   Dropped potentially leaking columns: {drop_cols}")
        
    print(f"   Sorted {len(df)} fights chronologically")
    return df


def _apply_strict_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Remove Women's fights, Split Decisions, and fighters with < 2 total fights."""
    df = df.copy()
    start_len = len(df)
    
    # Remove Women's division fights - model trained only on men's fights
    if 'weight_class' in df.columns:
        womens_mask = df['weight_class'].astype(str).str.contains(
            r"Women'?s|W\s*Strawweight|W\s*Flyweight|W\s*Bantamweight|W\s*Featherweight",
            case=False, na=False, regex=True
        )
        removed_womens = int(womens_mask.sum())
        df = df[~womens_mask]
        if removed_womens > 0:
            print(f"   Removed {removed_womens} women's division fights (training on men's fights only)")
    
    removed_split = 0
    if 'method' in df.columns:
        mask = df['method'].astype(str).str.contains('Split Decision', case=False, na=False)
        removed_split = int(mask.sum())
        df = df[~mask]
        if removed_split > 0:
            print(f"   Removed {removed_split} split decision fights")
    
    all_fighters = pd.concat([df['fighter_a_name'], df['fighter_b_name']])
    counts = all_fighters.value_counts()
    keep_fighters = set(counts[counts > 2].index)
    
    mask_a = df['fighter_a_name'].isin(keep_fighters)
    mask_b = df['fighter_b_name'].isin(keep_fighters)
    df = df[mask_a & mask_b]
    
    removed_exp = start_len - len(df) - removed_split - removed_womens
    if removed_exp > 0:
        print(f"   Removed {removed_exp} fights involving fighters with <=2 bouts")
    
    return df


def _compute_physical_ratios(long_df: pd.DataFrame) -> pd.DataFrame:
    """Compute physical attribute ratios."""
    long_df = long_df.copy()
    
    def safe_div(a, b):
        return (a / (b + 1e-6)).fillna(1.0)
    
    if 'reach' in long_df.columns:
        opp_reach = long_df[['fight_id', 'fighter_name', 'reach']].copy()
        opp_reach.columns = ['fight_id', 'opponent_name', 'opp_reach']
        long_df = long_df.merge(opp_reach, on=['fight_id', 'opponent_name'], how='left')
        long_df['reach_ratio'] = safe_div(long_df['reach'], long_df['opp_reach'])
        # NOTE: Keep opp_reach for history feature computation (opp_reach_dec_avg, etc.)
    
    # NOTE: height_ratio removed - highly correlated with reach_ratio (r > 0.9)
        
    if 'age' in long_df.columns:
        opp_age = long_df[['fight_id', 'fighter_name', 'age']].copy()
        opp_age.columns = ['fight_id', 'opponent_name', 'opp_age']
        long_df = long_df.merge(opp_age, on=['fight_id', 'opponent_name'], how='left')
        long_df['age_ratio'] = safe_div(long_df['age'], long_df['opp_age'])
        # NOTE: Keep opp_age for history feature computation (opp_age_dec_avg, etc.)
    
    print(f"   Computed physical attribute ratios (reach, age) - height removed")
    return long_df


def _convert_to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Explodes wide dataframe into long format."""
    fighter_a_cols = [col for col in df.columns if col.startswith('fighter_a_')]
    base_cols = set()
    for col in fighter_a_cols:
        base_cols.add(col.replace('fighter_a_', ''))
    base_cols.discard('name')
    base_cols.discard('id')
    
    records = []
    for row in df.itertuples(index=False):
        # Get method for tracking
        method = getattr(row, 'method', 'Unknown') if hasattr(row, 'method') else 'Unknown'
        is_ko_tko = 1.0 if method and 'KO' in str(method).upper() else 0.0
        is_decision = 1.0 if method and 'Decision' in str(method) else 0.0
        
        meta = {
            'fight_id': row.fight_id,
            'event_date': row.event_date,
            'weight_class': getattr(row, 'weight_class', 'Unknown'),
            'result': getattr(row, 'outcome', np.nan),
            'method': method,
        }
        
        # Determine win/loss status safely (handling NaN for future fights)
        if pd.isna(meta['result']):
            win_a = np.nan
            win_b = np.nan
            ko_win_a = 0.0
            ko_win_b = 0.0
            dec_win_a = 0.0
            dec_win_b = 0.0
        elif meta['result'] == 1.0:
            win_a = 1.0
            win_b = 0.0
            ko_win_a = is_ko_tko
            ko_win_b = 0.0
            dec_win_a = is_decision
            dec_win_b = 0.0
        elif meta['result'] == 0.0:
            win_a = 0.0
            win_b = 1.0
            ko_win_a = 0.0
            ko_win_b = is_ko_tko
            dec_win_a = 0.0
            dec_win_b = is_decision
        else:
            # Draw or No Contest
            win_a = 0.5
            win_b = 0.5
            ko_win_a = 0.0
            ko_win_b = 0.0
            dec_win_a = 0.0
            dec_win_b = 0.0
        
        # Get per-min stats data
        a_mins = getattr(row, 'fighter_a_fight_minutes', 1.0) or 1.0
        b_mins = getattr(row, 'fighter_b_fight_minutes', 1.0) or 1.0
        a_sig_landed = getattr(row, 'fighter_a_sig_strikes_landed', 0.0) or 0.0
        b_sig_landed = getattr(row, 'fighter_b_sig_strikes_landed', 0.0) or 0.0

        rec_a = meta.copy()
        rec_a.update({
            'fighter_name': row.fighter_a_name,
            'opponent_name': row.fighter_b_name,
            'is_a': True,
            'win': win_a,
            'ko_tko_win': ko_win_a,
            'decision_win': dec_win_a,  # For decision_win_rate
            'sig_strikes_landed_per_min': a_sig_landed / max(a_mins, 0.1),
            'sig_strikes_absorbed_per_min': b_sig_landed / max(a_mins, 0.1),  # Opponent's landed = absorbed
        })
        for base in base_cols:
            rec_a[base] = getattr(row, f'fighter_a_{base}', np.nan)
        records.append(rec_a)
        
        rec_b = meta.copy()
        rec_b.update({
            'fighter_name': row.fighter_b_name,
            'opponent_name': row.fighter_a_name,
            'is_a': False,
            'win': win_b,
            'ko_tko_win': ko_win_b,
            'decision_win': dec_win_b,  # For decision_win_rate
            'sig_strikes_landed_per_min': b_sig_landed / max(b_mins, 0.1),
            'sig_strikes_absorbed_per_min': a_sig_landed / max(b_mins, 0.1),  # Opponent's landed = absorbed
        })
        for base in base_cols:
            rec_b[base] = getattr(row, f'fighter_b_{base}', np.nan)
        records.append(rec_b)
        
    long_df = pd.DataFrame(records)
    long_df = long_df.sort_values(['fighter_name', 'event_date', 'fight_id']).reset_index(drop=True)
    print(f"   Converted to long format: {len(long_df)} records")
    return long_df


def _compute_ratio_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """Compute ratio-based features using historical averages."""
    long_df = long_df.copy()
    
    def safe_div(a, b):
        return (a / (b + 1e-6)).fillna(0)

    # Striking target ratios
    if all(f'{c}_dec_avg' in long_df.columns for c in ['head_strikes_landed', 'body_strikes_landed', 'leg_strikes_landed']):
        total = long_df['head_strikes_landed_dec_avg'] + long_df['body_strikes_landed_dec_avg'] + long_df['leg_strikes_landed_dec_avg']
        long_df['head_land_ratio'] = safe_div(long_df['head_strikes_landed_dec_avg'], total)
        long_df['body_land_ratio'] = safe_div(long_df['body_strikes_landed_dec_avg'], total)
        long_df['leg_land_ratio'] = safe_div(long_df['leg_strikes_landed_dec_avg'], total)
        
    # Position ratios
    if all(f'{c}_dec_avg' in long_df.columns for c in ['distance_strikes_landed', 'clinch_strikes_landed', 'ground_strikes_landed']):
        total = long_df['distance_strikes_landed_dec_avg'] + long_df['clinch_strikes_landed_dec_avg'] + long_df['ground_strikes_landed_dec_avg']
        long_df['distance_land_ratio'] = safe_div(long_df['distance_strikes_landed_dec_avg'], total)
        long_df['clinch_land_ratio'] = safe_div(long_df['clinch_strikes_landed_dec_avg'], total)
        long_df['ground_land_ratio'] = safe_div(long_df['ground_strikes_landed_dec_avg'], total)
        
    # Accuracy
    if 'sig_strikes_landed_dec_avg' in long_df.columns and 'sig_strikes_attempted_dec_avg' in long_df.columns:
        long_df['sig_str_acc'] = safe_div(long_df['sig_strikes_landed_dec_avg'], long_df['sig_strikes_attempted_dec_avg'])
        
    if 'takedowns_landed_dec_avg' in long_df.columns and 'takedowns_attempted_dec_avg' in long_df.columns:
        long_df['td_acc'] = safe_div(long_df['takedowns_landed_dec_avg'], long_df['takedowns_attempted_dec_avg'])

    # KD per sig strike landed (Power)
    if 'knockdowns_dec_avg' in long_df.columns and 'sig_strikes_landed_dec_avg' in long_df.columns:
        long_df['kd_per_sig_str'] = safe_div(long_df['knockdowns_dec_avg'], long_df['sig_strikes_landed_dec_avg'])
        
    # TD per sig strike attempt (Level Change Threat) - matches td_per_sig_str_att_diff
    if 'takedowns_landed_dec_avg' in long_df.columns and 'sig_strikes_attempted_dec_avg' in long_df.columns:
        long_df['td_per_sig_att'] = safe_div(long_df['takedowns_landed_dec_avg'], long_df['sig_strikes_attempted_dec_avg'])
        long_df['td_per_sig_str_att'] = long_df['td_per_sig_att']  # Alias for requested feature name

    # ===== NEW FEATURES (Requested) =====
    
    # Control time per minute (ctrl_per_min)
    if 'control_time_dec_avg' in long_df.columns and 'fight_minutes_dec_avg' in long_df.columns:
        long_df['ctrl_per_min'] = safe_div(long_df['control_time_dec_avg'], long_df['fight_minutes_dec_avg'])
    
    # Control ratio - control time vs opponent's control time (will be computed as diff anyway)
    # For now, create control time per fight as a proxy
    if 'control_time_per_fight_dec_avg' in long_df.columns:
        long_df['ctrl_ratio'] = long_df['control_time_per_fight_dec_avg']
    elif 'control_time_dec_avg' in long_df.columns:
        long_df['ctrl_ratio'] = long_df['control_time_dec_avg']
    
    # Distance accuracy (distance_acc)
    if 'distance_strikes_landed_dec_avg' in long_df.columns and 'distance_strikes_attempted_dec_avg' in long_df.columns:
        long_df['distance_acc'] = safe_div(long_df['distance_strikes_landed_dec_avg'], long_df['distance_strikes_attempted_dec_avg'])
    
    # Reversal rate (rev_rate) - reversals per fight (renamed from rev_ratio for clarity)
    if 'reversals_dec_avg' in long_df.columns:
        long_df['rev_rate'] = long_df['reversals_dec_avg']
    
    # Leg strikes landed per minute (leg_land_per_min)
    if 'leg_strikes_landed_dec_avg' in long_df.columns and 'fight_minutes_dec_avg' in long_df.columns:
        long_df['leg_land_per_min'] = safe_div(long_df['leg_strikes_landed_dec_avg'], long_df['fight_minutes_dec_avg'])
    
    # Head defense (head_def) - how well we avoid opponent's head strikes
    # Uses OPPONENT's head strike data (same pattern as sig_strike_defense)
    if 'head_strikes_landed_dec_avg' in long_df.columns and 'head_strikes_attempted_dec_avg' in long_df.columns:
        # Get opponent's head striking data
        opp_head = long_df[['fight_id', 'fighter_name', 'head_strikes_landed_dec_avg', 'head_strikes_attempted_dec_avg']].copy()
        opp_head.columns = ['fight_id', 'opponent_name', 'opp_head_landed', 'opp_head_attempted']
        long_df = long_df.merge(opp_head, on=['fight_id', 'opponent_name'], how='left')
        # head_def = 1 - (opponent_head_landed / opponent_head_attempted)
        long_df['head_def'] = 1 - (long_df['opp_head_landed'] / (long_df['opp_head_attempted'] + 1e-6))
        long_df['head_def'] = long_df['head_def'].clip(0, 1).fillna(0.5)
        long_df = long_df.drop(columns=['opp_head_landed', 'opp_head_attempted'], errors='ignore')

    # NEW: Body Defense (body_def) - how well we avoid opponent's body strikes
    if 'body_strikes_landed_dec_avg' in long_df.columns and 'body_strikes_attempted_dec_avg' in long_df.columns:
        opp_body = long_df[['fight_id', 'fighter_name', 'body_strikes_landed_dec_avg', 'body_strikes_attempted_dec_avg']].copy()
        opp_body.columns = ['fight_id', 'opponent_name', 'opp_body_landed', 'opp_body_attempted']
        long_df = long_df.merge(opp_body, on=['fight_id', 'opponent_name'], how='left')
        long_df['body_def'] = 1 - (long_df['opp_body_landed'] / (long_df['opp_body_attempted'] + 1e-6))
        long_df['body_def'] = long_df['body_def'].clip(0, 1).fillna(0.5)
        long_df = long_df.drop(columns=['opp_body_landed', 'opp_body_attempted'], errors='ignore')

    # NEW: Leg Defense (leg_def) - how well we avoid opponent's leg strikes
    if 'leg_strikes_landed_dec_avg' in long_df.columns and 'leg_strikes_attempted_dec_avg' in long_df.columns:
        opp_leg = long_df[['fight_id', 'fighter_name', 'leg_strikes_landed_dec_avg', 'leg_strikes_attempted_dec_avg']].copy()
        opp_leg.columns = ['fight_id', 'opponent_name', 'opp_leg_landed', 'opp_leg_attempted']
        long_df = long_df.merge(opp_leg, on=['fight_id', 'opponent_name'], how='left')
        long_df['leg_def'] = 1 - (long_df['opp_leg_landed'] / (long_df['opp_leg_attempted'] + 1e-6))
        long_df['leg_def'] = long_df['leg_def'].clip(0, 1).fillna(0.5)
        long_df = long_df.drop(columns=['opp_leg_landed', 'opp_leg_attempted'], errors='ignore')

    # NOTE: sub_def feature skipped - complex to compute properly without sub defense data
    # Reliance on takedown_defense is sufficient for grappling defense metrics

    # NEW: Ground Strikes per Control Minute (ground_land_per_ctrl)
    # Damage efficiency on ground
    if 'ground_strikes_landed_dec_avg' in long_df.columns and 'control_time_dec_avg' in long_df.columns:
        long_df['ground_land_per_ctrl'] = safe_div(long_df['ground_strikes_landed_dec_avg'], long_df['control_time_dec_avg'] / 60.0)

    # NEW: Takedowns per Control Minute (td_land_per_ctrl)
    # Chain wrestling efficiency
    if 'takedowns_landed_dec_avg' in long_df.columns and 'control_time_dec_avg' in long_df.columns:
        long_df['td_land_per_ctrl'] = safe_div(long_df['takedowns_landed_dec_avg'], long_df['control_time_dec_avg'] / 60.0)

    # NEW: Reversals per Opponent Control Minute (rev_per_ctrlopp)
    # Scrambling efficiency: How often do you reverse when controlled?
    if 'reversals_dec_avg' in long_df.columns:
        # Need opponent's control time
        opp_ctrl = long_df[['fight_id', 'fighter_name', 'control_time_dec_avg']].copy()
        opp_ctrl.columns = ['fight_id', 'opponent_name', 'opp_control_time']
        long_df = long_df.merge(opp_ctrl, on=['fight_id', 'opponent_name'], how='left')
        long_df['rev_per_ctrlopp'] = safe_div(long_df['reversals_dec_avg'], long_df['opp_control_time'] / 60.0)
        long_df = long_df.drop(columns=['opp_control_time'], errors='ignore')
    
    # Submission attempts per fight (sub_att)
    if 'submission_attempts_dec_avg' in long_df.columns:
        long_df['sub_att'] = long_df['submission_attempts_dec_avg']
    
    # Distance per sig strike (distance_per_sig_str_land) - distance strikes / total sig strikes
    if 'distance_strikes_landed_dec_avg' in long_df.columns and 'sig_strikes_landed_dec_avg' in long_df.columns:
        long_df['distance_per_sig_str'] = safe_div(long_df['distance_strikes_landed_dec_avg'], long_df['sig_strikes_landed_dec_avg'])
    
    # REMOVED: ko_ratio (duplicate of ko_tko_win_rate)
    # REMOVED: sig_str_land_ratio (duplicate of sig_str_acc)
    
    return long_df


def _compute_history_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ALL historical layers internally (career_avg, ewm_avg, roll3_avg).
    These are needed for proper z-score and momentum calculations.
    """
    long_df = long_df.copy()
    
    exclude = {'fight_id', 'event_date', 'fighter_name', 'opponent_name', 'is_a', 'weight_class', 'win', 'result', 
               'height', 'reach'}  # age removed from exclude to get age_dec_avg_diff
    numeric_cols = [c for c in long_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(long_df[c])]
    
    grouped = long_df.groupby('fighter_name', group_keys=False)
    
    for col in numeric_cols:
        # Compute ALL layers (needed internally)
        long_df[f'{col}_career_avg'] = grouped[col].transform(
            lambda x: x.shift(1).expanding().mean()
        ).fillna(0)
        
        long_df[f'{col}_dec_avg'] = grouped[col].transform(
            lambda x: x.shift(1).ewm(alpha=0.15, min_periods=2).mean()
        ).fillna(0)
        
        # roll3 computation REMOVED - not used in final features
    # Win Rate - LEAKAGE SAFE: shift(1) ensures current fight outcome is EXCLUDED
    # win_rate represents fighter's record BEFORE this fight, not including it
    # This feature is used internally for SOS calculation but NOT included in final model features
    long_df['career_wins'] = grouped['win'].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)
    long_df['career_fights'] = grouped['win'].transform(lambda x: x.shift(1).expanding().count()).fillna(0)
    long_df['win_rate'] = (long_df['career_wins'] / long_df['career_fights'].replace(0, 1)).fillna(0.5)
    
    # NEW FEATURES: Additional derived stats from historical data
    # total_fights: Career experience (number of fights before this one)
    long_df['total_fights'] = long_df['career_fights']  # Already computed above
    
    # win_streak: Consecutive wins before this fight (reset on loss)
    def compute_streak(wins):
        shifted = wins.shift(1).fillna(0)
        streak = []
        current_streak = 0
        for w in shifted:
            if w == 1:
                current_streak += 1
            else:
                current_streak = 0
            streak.append(current_streak)
        return pd.Series(streak, index=wins.index)
    
    long_df['win_streak'] = grouped['win'].transform(compute_streak).fillna(0)
    
    # days_since_last_fight: Days between current fight and previous fight (layoff time)
    long_df['event_date_dt'] = pd.to_datetime(long_df['event_date'])
    long_df['days_since_last_fight'] = grouped['event_date_dt'].transform(
        lambda x: x.diff().dt.days
    ).fillna(365)  # Default to 1 year for debuts
    long_df['days_since_last_fight'] = long_df['days_since_last_fight'].clip(0, 1500)  # Cap at ~4 years
    long_df = long_df.drop(columns=['event_date_dt'], errors='ignore')
    
    # recent_wins: REMOVED (user requested - redundant with win_streak)
    # long_df['recent_wins'] = grouped['win'].transform(
    #     lambda x: x.shift(1).rolling(window=5, min_periods=1).sum()
    # ).fillna(0)
    
    # ko_tko_win_rate: Percentage of wins that were by KO/TKO (shifted to exclude current fight)
    if 'ko_tko_win' in long_df.columns:
        long_df['career_ko_wins'] = grouped['ko_tko_win'].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)
        long_df['ko_tko_win_rate'] = (long_df['career_ko_wins'] / long_df['career_wins'].replace(0, 1)).fillna(0)
        long_df = long_df.drop(columns=['career_ko_wins'], errors='ignore')
        print("   Added ko_tko_win_rate")

    # decision_win_rate: Percentage of wins that were by Decision (shifted)
    if 'decision_win' in long_df.columns:
        long_df['career_dec_wins'] = grouped['decision_win'].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)
        long_df['decision_win_rate'] = (long_df['career_dec_wins'] / long_df['career_wins'].replace(0, 1)).fillna(0)
        long_df = long_df.drop(columns=['career_dec_wins'], errors='ignore')
        print("   Added decision_win_rate")
    
    # win_percentage: REMOVED (100% correlated with win_rate - just scaled differently)
    # long_df['win_percentage'] = long_df['win_rate'] * 100
    
    # sig_strike_defense: 1 - (opponent_sig_strikes_landed / opponent_sig_strikes_attempted)
    # Uses already-computed decayed averages from opponent's perspective
    # Since we removed opponent columns, we need to compute this from the opponent's data
    if 'sig_strikes_landed_dec_avg' in long_df.columns and 'sig_strikes_attempted_dec_avg' in long_df.columns:
        # Get opponent's striking data
        opp_striking = long_df[['fight_id', 'fighter_name', 'sig_strikes_landed_dec_avg', 'sig_strikes_attempted_dec_avg']].copy()
        opp_striking.columns = ['fight_id', 'opponent_name', 'opp_sig_landed', 'opp_sig_attempted']
        long_df = long_df.merge(opp_striking, on=['fight_id', 'opponent_name'], how='left')
        # Defense = 1 - (opponent_landed / opponent_attempted) = how well you defend
        long_df['sig_strike_defense'] = 1 - (long_df['opp_sig_landed'] / (long_df['opp_sig_attempted'] + 1e-6))
        long_df['sig_strike_defense'] = long_df['sig_strike_defense'].clip(0, 1).fillna(0.5)
        long_df = long_df.drop(columns=['opp_sig_landed', 'opp_sig_attempted'], errors='ignore')
    
    # takedown_defense: 1 - (opponent_takedowns_landed / opponent_takedowns_attempted)
    if 'takedowns_landed_dec_avg' in long_df.columns and 'takedowns_attempted_dec_avg' in long_df.columns:
        opp_td = long_df[['fight_id', 'fighter_name', 'takedowns_landed_dec_avg', 'takedowns_attempted_dec_avg']].copy()
        opp_td.columns = ['fight_id', 'opponent_name', 'opp_td_landed', 'opp_td_attempted']
        long_df = long_df.merge(opp_td, on=['fight_id', 'opponent_name'], how='left')
        long_df['takedown_defense'] = 1 - (long_df['opp_td_landed'] / (long_df['opp_td_attempted'] + 1e-6))
        long_df['takedown_defense'] = long_df['takedown_defense'].clip(0, 1).fillna(0.5)
        long_df = long_df.drop(columns=['opp_td_landed', 'opp_td_attempted'], errors='ignore')
    
    print(f"   Computed history features (Career, EWM) for {len(numeric_cols)} metrics")
    print(f"   Added: total_fights, win_streak, sig_strike_defense, takedown_defense")
    long_df = long_df.drop(columns=['career_wins', 'career_fights'], errors='ignore')
    # NOTE: 'win' column is kept for Elo calculation - will be dropped after
    return long_df


def _compute_opponent_adjusted_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """Computes Z-scores and Z-score Decayed Averages using opponent allowances."""
    long_df = long_df.copy()
    
    # Get EWM columns but exclude roll3 features (user requested removal)
    ewm_cols = [c for c in long_df.columns if c.endswith('_dec_avg') and 'roll3' not in c]
    base_stats = [c.replace('_dec_avg', '') for c in ewm_cols]
    
    # Add Ratio features to base_stats (EXCLUDING physical attribute ratios)
    # Physical ratios (reach_ratio, height_ratio, age_ratio) should NOT get z-scores
    # because they are static attributes, not performance metrics
    physical_ratios = {'reach_ratio', 'height_ratio', 'age_ratio'}
    ratio_cols = [c for c in long_df.columns 
                  if (c.endswith('_ratio') or c.endswith('_acc') or '_per_' in c)
                  and c not in physical_ratios]
    base_stats.extend(ratio_cols)
    # Remove duplicates just in case
    base_stats = list(set(base_stats))
    
    opp_stats = long_df[['fight_id', 'event_date', 'opponent_name']].copy()
    
    for base in base_stats:
        if base in long_df.columns:
            opp_stats[f'allowed_{base}'] = long_df[base]
            
    opp_stats = opp_stats.sort_values(['opponent_name', 'event_date', 'fight_id'])
    grouped_opp = opp_stats.groupby('opponent_name', group_keys=False)
    
    for base in base_stats:
        col_allowed = f'allowed_{base}'
        
        opp_stats[f'{base}_opp_allowed'] = grouped_opp[col_allowed].transform(
            lambda x: x.shift(1).ewm(alpha=0.15, min_periods=2).mean()
        )
        
        opp_stats[f'{base}_opp_mad'] = grouped_opp[col_allowed].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=2).apply(
                lambda y: np.median(np.abs(y - np.median(y)))
            )
        ).fillna(1.0)
        
        opp_stats[f'{base}_opp_mad'] = opp_stats[f'{base}_opp_mad'].replace(0, 1e-6)
        
    merge_cols = ['fight_id', 'opponent_name'] + \
                 [c for c in opp_stats.columns if c.endswith('_opp_allowed') or c.endswith('_opp_mad')]
                 
    long_df = long_df.merge(opp_stats[merge_cols], on=['fight_id', 'opponent_name'], how='left')
    
    # Compute Z-Scores
    for base in base_stats:
        # Use _dec_avg if available (for raw stats), otherwise use base directly (for ratios)
        if f'{base}_dec_avg' in long_df.columns:
            ewm_col = f'{base}_dec_avg'
        else:
            ewm_col = base
            
        opp_allowed_col = f'{base}_opp_allowed'
        opp_mad_col = f'{base}_opp_mad'
        
        if ewm_col in long_df.columns and opp_allowed_col in long_df.columns:
            z_col = f'{base}_zscore'
            long_df[z_col] = (long_df[ewm_col] - long_df[opp_allowed_col]) / long_df[opp_mad_col]
            long_df[z_col] = long_df[z_col].clip(-7, 7)
    
    # === FINAL LAYER: Adjusted Performance Decayed Average (Z-Score Based) ⭐ ===
    grouped = long_df.groupby('fighter_name', group_keys=False)
    
    for base in base_stats:
        z_col = f'{base}_zscore'
        if z_col in long_df.columns:
            # Decayed average of z-scores (EWM with shift)
            # Renamed to _dec_adjperf_dec_avg to match user preference
            long_df[f'{base}_dec_adjperf_dec_avg'] = grouped[z_col].transform(
                lambda x: x.shift(1).ewm(alpha=0.15, min_periods=2).mean()
            ).fillna(0)
            
            # Winsorize the final layer to prevent extreme values (Khamzat rule)
            long_df[f'{base}_dec_adjperf_dec_avg'] = long_df[f'{base}_dec_adjperf_dec_avg'].clip(-7, 7)
    
    # Strength of Schedule (SOS) - measures quality of opponents faced
    # LEAKAGE SAFE: Uses win_rate which already excludes current fight (shift(1) applied earlier)
    # Then applies another shift(1) to exclude current opponent from the average
    
    # Step 1: Get opponent's win_rate for each fight (opponent's quality)
    # Note: opp's win_rate is their record BEFORE this fight (already shifted)
    opp_quality = long_df[['fight_id', 'fighter_name', 'win_rate']].copy()
    opp_quality.columns = ['fight_id', 'opponent_name', 'opp_win_rate']
    long_df = long_df.merge(opp_quality, on=['fight_id', 'opponent_name'], how='left')
    
    # Step 2: SOS = EWM average of opponent win rates faced (shifted to exclude current fight)
    # Double protection: opp_win_rate is already pre-fight, and we shift(1) again
    long_df = long_df.sort_values(['fighter_name', 'event_date', 'fight_id'])
    grouped = long_df.groupby('fighter_name', group_keys=False)
    long_df['sos_ewm'] = grouped['opp_win_rate'].transform(
        lambda x: x.shift(1).ewm(alpha=0.15, min_periods=2).mean()
    ).fillna(0.5)
    
    # Drop opp_win_rate after SOS calculation - it should NOT be a final feature
    long_df = long_df.drop(columns=['opp_win_rate'], errors='ignore')
    
    print(f"   Computed Z-Scores, Z-Score Decayed Averages (FINAL LAYER [BEST]), and SOS for {len(base_stats)} metrics")
    return long_df


def _compute_elo_ratings(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Elo ratings for each fighter based on their fight history.
    
    Features generated:
    - elo: Fighter's pre-fight Elo rating
    - elo_trend: Change in Elo over last 3 fights (momentum indicator)
    
    These become elo_diff, elo_diff_squared, elo_win_prob when differentials are created.
    """
    long_df = long_df.copy()
    print("   Computing Elo ratings...")
    
    # === ELO PARAMETERS ===
    BASE_ELO = 1500.0
    
    # Finish multipliers - reward decisive victories
    FINISH_MULTIPLIERS = {
        'KO/TKO': 1.5,
        'SUB': 1.4,
        'TKO': 1.3,
        'Decision - Unanimous': 1.0,
        'Decision - Majority': 0.9,
        'Decision - Split': 0.8,  # Less certain skill gap
        'default': 1.0,
    }
    
    def get_k_factor(num_fights: int) -> float:
        """Dynamic K-factor: higher for new fighters, lower for established."""
        if num_fights < 5:
            return 150.0  # New fighters - ratings adjust quickly
        elif num_fights < 15:
            return 100.0  # Developing fighters
        else:
            return 60.0   # Established fighters - more stable ratings
    
    def get_finish_multiplier(method: str) -> float:
        """Get point multiplier based on finish type."""
        if pd.isna(method):
            return 1.0
        method_str = str(method).upper()
        if 'KO' in method_str or 'TKO' in method_str:
            return FINISH_MULTIPLIERS['KO/TKO']
        elif 'SUB' in method_str:
            return FINISH_MULTIPLIERS['SUB']
        elif 'SPLIT' in method_str:
            return FINISH_MULTIPLIERS['Decision - Split']
        elif 'MAJORITY' in method_str:
            return FINISH_MULTIPLIERS['Decision - Majority']
        elif 'DECISION' in method_str:
            return FINISH_MULTIPLIERS['Decision - Unanimous']
        return FINISH_MULTIPLIERS['default']
    
    def expected_score(rating_a: float, rating_b: float) -> float:
        """Calculate expected win probability for fighter A."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    
    # Sort by date to process chronologically
    long_df = long_df.sort_values(['event_date', 'fight_id']).reset_index(drop=True)
    
    # Initialize Elo tracking
    fighter_elo = {}  # Current Elo for each fighter
    fighter_fights = {}  # Number of fights for each fighter (for K-factor)
    fighter_elo_history = {}  # Last N Elo values for trend calculation
    
    # Pre-fight Elo values (what we'll use as features)
    prefight_elo = []
    prefight_elo_trend = []
    
    # Group by fight to process both fighters together
    fight_ids = long_df['fight_id'].unique()
    
    # Build a lookup for each row's position
    row_to_idx = {(row.fight_id, row.fighter_name): idx 
                  for idx, row in long_df.iterrows()}
    
    # Initialize arrays
    n_rows = len(long_df)
    elo_values = np.full(n_rows, np.nan)
    elo_trend_values = np.full(n_rows, np.nan)
    
    for fight_id in fight_ids:
        fight_rows = long_df[long_df['fight_id'] == fight_id]
        
        if len(fight_rows) != 2:
            continue
            
        fighter_a = fight_rows.iloc[0]['fighter_name']
        fighter_b = fight_rows.iloc[1]['fighter_name']
        
        # Get PRE-FIGHT Elo (before this fight happens)
        elo_a = fighter_elo.get(fighter_a, BASE_ELO)
        elo_b = fighter_elo.get(fighter_b, BASE_ELO)
        
        # Compute Elo trend (change over last 3 fights)
        history_a = fighter_elo_history.get(fighter_a, [])
        history_b = fighter_elo_history.get(fighter_b, [])
        
        if len(history_a) >= 3:
            trend_a = elo_a - history_a[-3]  # Current minus 3 fights ago
        elif len(history_a) >= 1:
            trend_a = elo_a - history_a[0]   # Current minus earliest
        else:
            trend_a = 0.0
            
        if len(history_b) >= 3:
            trend_b = elo_b - history_b[-3]
        elif len(history_b) >= 1:
            trend_b = elo_b - history_b[0]
        else:
            trend_b = 0.0
        
        # Store pre-fight values
        idx_a = row_to_idx.get((fight_id, fighter_a))
        idx_b = row_to_idx.get((fight_id, fighter_b))
        
        if idx_a is not None:
            elo_values[idx_a] = elo_a
            elo_trend_values[idx_a] = trend_a
        if idx_b is not None:
            elo_values[idx_b] = elo_b
            elo_trend_values[idx_b] = trend_b
        
        # Now UPDATE Elo based on fight result
        win_a = fight_rows.iloc[0]['win']
        method = fight_rows.iloc[0].get('method', None)
        
        if pd.isna(win_a):
            continue  # No result yet (future fight)
        
        # Get K-factors
        fights_a = fighter_fights.get(fighter_a, 0)
        fights_b = fighter_fights.get(fighter_b, 0)
        k_a = get_k_factor(fights_a)
        k_b = get_k_factor(fights_b)
        
        # Get finish multiplier
        finish_mult = get_finish_multiplier(method)
        
        # Calculate expected scores
        expected_a = expected_score(elo_a, elo_b)
        expected_b = 1.0 - expected_a
        
        # Actual results
        if win_a == 1.0:
            actual_a, actual_b = 1.0, 0.0
        elif win_a == 0.0:
            actual_a, actual_b = 0.0, 1.0
        else:  # Draw
            actual_a, actual_b = 0.5, 0.5
            finish_mult = 0.5  # Draws don't get finish bonus
        
        # Update Elo ratings
        new_elo_a = elo_a + k_a * (actual_a - expected_a) * finish_mult
        new_elo_b = elo_b + k_b * (actual_b - expected_b) * finish_mult
        
        # Store updated values
        fighter_elo[fighter_a] = new_elo_a
        fighter_elo[fighter_b] = new_elo_b
        fighter_fights[fighter_a] = fights_a + 1
        fighter_fights[fighter_b] = fights_b + 1
        
        # Update history for trend calculation
        if fighter_a not in fighter_elo_history:
            fighter_elo_history[fighter_a] = []
        if fighter_b not in fighter_elo_history:
            fighter_elo_history[fighter_b] = []
        fighter_elo_history[fighter_a].append(new_elo_a)
        fighter_elo_history[fighter_b].append(new_elo_b)
        
        # Keep only last 10 for memory efficiency
        if len(fighter_elo_history[fighter_a]) > 10:
            fighter_elo_history[fighter_a] = fighter_elo_history[fighter_a][-10:]
        if len(fighter_elo_history[fighter_b]) > 10:
            fighter_elo_history[fighter_b] = fighter_elo_history[fighter_b][-10:]
    
    # Add to dataframe
    long_df['elo'] = elo_values
    long_df['elo_trend'] = elo_trend_values
    
    # Fill NaN with base Elo for fighters with no history
    long_df['elo'] = long_df['elo'].fillna(BASE_ELO)
    long_df['elo_trend'] = long_df['elo_trend'].fillna(0.0)
    
    # Stats
    valid_elo = long_df['elo'].notna().sum()
    print(f"   Elo ratings computed for {valid_elo} fighter-fight records")
    print(f"   Elo range: {long_df['elo'].min():.0f} - {long_df['elo'].max():.0f}")
    print(f"   Unique fighters with Elo: {len(fighter_elo)}")
    
    # Drop 'win' column now that Elo is computed - prevents leakage
    long_df = long_df.drop(columns=['win'], errors='ignore')
    
    return long_df


def _compute_momentum(long_df: pd.DataFrame) -> pd.DataFrame:
    """Computes Momentum: Trend of improvement."""
    long_df = long_df.copy()
    
    ewm_cols = [c for c in long_df.columns if c.endswith('_dec_avg')]
    base_stats = [c.replace('_dec_avg', '') for c in ewm_cols]
    
    for base in base_stats:
        roll3_col = f'{base}_roll3_avg'
        career_col = f'{base}_career_avg'
        
        if roll3_col in long_df.columns and career_col in long_df.columns:
            long_df[f'{base}_momentum'] = long_df[roll3_col] - long_df[career_col]
            
    print(f"   Computed momentum features for {len(base_stats)} metrics")
    return long_df


def _merge_and_create_differentials(df: pd.DataFrame, long_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Merges back and creates differentials.
    
    CRITICAL FILTER: Only keep features ending with:
    - _ratio
    - _dec_adjperf_dec_avg ⭐ (FINAL LAYER - most predictive)
    - _dec_avg (Volume/Rate baseline)
    - sos_ewm (single best win metric - opponent quality trend)
    - elo, elo_trend, elo_diff_squared, elo_win_prob (NEW - Elo rating features)
    
    This excludes _career_avg, _zscore (intermediate layers), 
    win_rate, and opp_win_rate_hist to reduce multicollinearity.
    """
    
    # Drop all raw per-fight stats and intermediate calculations
    # LEAKAGE GUARD: win_rate is kept (safe due to shift(1))
    # NOTE: sos_ewm REMOVED - had negative importance and hurt predictions
    keep_meta = {'fighter_name', 'opponent_name', 'fight_id', 'event_date', 'weight_class', 'is_a',
                  'total_fights', 'win_streak', 'sig_strike_defense', 'takedown_defense', 
                  'win_rate', 'ko_tko_win_rate', 'decision_win_rate', 'days_since_last_fight',
                  # Ratio features (removed duplicates: ko_ratio, sig_str_land_ratio)
                  'ctrl_per_min', 'ctrl_ratio', 'distance_acc', 'rev_rate', 'leg_land_per_min',
                  'head_def', 'body_def', 'leg_def', 'sub_att', 'distance_per_sig_str', 'td_per_sig_str_att',
                  'ground_land_per_ctrl', 'td_land_per_ctrl', 'rev_per_ctrlopp',
                  # ELO features (NEW)
                  'elo', 'elo_trend'}
    drop_raw = [c for c in long_df.columns if not (
        c.endswith('_ratio') or 
        c.endswith('_acc') or 
        c.endswith('_dec_adjperf_dec_avg') or 
        c.endswith('_dec_avg') or 
        c in keep_meta
    )]
    long_df = long_df.drop(columns=drop_raw, errors='ignore')
    
    # LEAKAGE VERIFICATION: Ensure win_rate is NOT in the data at this point
    # Win_rate is confirmed safe (shifted) and re-enabled by user request
    # if 'win_rate' in long_df.columns:
    #     print("   WARNING: Dropping win_rate to prevent potential leakage - use sos_ewm instead")
    #     long_df = long_df.drop(columns=['win_rate'], errors='ignore')
    
    # Select features to keep - ONLY the specified layers (excluding roll3)
    feature_cols = [c for c in long_df.columns if (
        c.endswith('_ratio') or 
        c.endswith('_acc') or 
        c.endswith('_dec_adjperf_dec_avg') or 
        c.endswith('_dec_avg')
    ) and 'roll3' not in c]  # Exclude roll3 features
    # Keep derived features (sos_ewm REMOVED - negative importance)
    feature_cols += ['total_fights', 'win_streak', 'sig_strike_defense', 'takedown_defense', 
                     'win_rate', 'ko_tko_win_rate', 'decision_win_rate', 'days_since_last_fight',
                     # Ratio features (renamed: rev_ratio -> rev_rate)
                     'ctrl_per_min', 'ctrl_ratio', 'distance_acc', 'rev_rate', 'leg_land_per_min',
                     'head_def', 'body_def', 'leg_def', 'sub_att', 'distance_per_sig_str', 'td_per_sig_str_att',
                     'ground_land_per_ctrl', 'td_land_per_ctrl', 'rev_per_ctrlopp',
                     # ELO features (NEW)
                     'elo', 'elo_trend']
    
    # Filter to only features that exist
    feature_cols = [c for c in feature_cols if c in long_df.columns]
    
    # Remove duplicates and sort
    feature_cols = sorted(list(set(feature_cols)))
    
    # LEAKAGE VERIFICATION: Ensure no raw current-fight stats leaked through (win_rate is now ALLOWED)
    forbidden_patterns = ['career_wins', 'career_fights', 'opp_win_rate']
    leaked_features = [f for f in feature_cols if any(p in f for p in forbidden_patterns)]
    if leaked_features:
        print(f"   ERROR: Potential leakage detected in features: {leaked_features}")
        feature_cols = [f for f in feature_cols if f not in leaked_features]
    
    print(f"   Selected {len(feature_cols)} features from layers: *_ratio, *_dec_adjperf_dec_avg, *_dec_avg, ELO")
    
    # Split
    long_a = long_df[long_df['is_a']][['fight_id'] + feature_cols].copy()
    long_b = long_df[~long_df['is_a']][['fight_id'] + feature_cols].copy()
    
    # Rename
    long_a.columns = ['fight_id'] + [f'A_{c}' for c in feature_cols]
    long_b.columns = ['fight_id'] + [f'B_{c}' for c in feature_cols]
    
    # Merge - include odds columns if they exist
    base_cols = ['fight_id', 'event_date', 'fighter_a_name', 'fighter_b_name', 'outcome']
    odds_cols = ['A_open_odds', 'B_open_odds', 'A_open_prob', 'B_open_prob', 
                 'opening_odds_diff', 'implied_prob_A']
    merge_cols = base_cols + [c for c in odds_cols if c in df.columns]
    df_final = df[merge_cols].copy()
    df_final = df_final.merge(long_a, on='fight_id', how='left')
    df_final = df_final.merge(long_b, on='fight_id', how='left')
    
    # Diff
    final_cols = []
    for col in feature_cols:
        diff_name = f'{col}_diff'
        df_final[diff_name] = df_final[f'A_{col}'] - df_final[f'B_{col}']
        final_cols.append(diff_name)
    
    # === SPECIAL ELO FEATURES ===
    if 'A_elo' in df_final.columns and 'B_elo' in df_final.columns:
        # elo_diff_squared: Captures non-linear mismatch effects
        # A huge mismatch (diff=400) is exponentially different than moderate (diff=100)
        df_final['elo_diff_squared'] = df_final['elo_diff'] ** 2
        # Preserve sign: negative if B is higher rated
        df_final['elo_diff_squared'] = df_final['elo_diff_squared'] * np.sign(df_final['elo_diff'])
        final_cols.append('elo_diff_squared')
        
        # elo_win_prob: Expected win probability for Fighter A based on Elo
        # This is the direct Elo prediction - very interpretable
        df_final['elo_win_prob'] = 1.0 / (1.0 + 10.0 ** ((df_final['B_elo'] - df_final['A_elo']) / 400.0))
        final_cols.append('elo_win_prob')
        
        print(f"   Added Elo-derived features: elo_diff_squared, elo_win_prob")
            
    return df_final, final_cols


def _final_cleanup(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Drops all non-feature, non-target columns to prevent leakage."""
    meta_cols = ['fight_id', 'event_date', 'fighter_a_name', 'fighter_b_name', 'outcome']
    
    # Preserve betting odds columns if they exist (needed for ROI analysis and as features)
    odds_cols = ['A_open_odds', 'B_open_odds', 'A_open_prob', 'B_open_prob', 
                 'opening_odds_diff', 'implied_prob_A']
    for col in odds_cols:
        if col in df.columns:
            meta_cols.append(col)
    
    keep_cols = meta_cols + feature_cols
    # Only keep columns that exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    df_clean = df[keep_cols].copy()
    return df_clean


if __name__ == "__main__":
    print("This module is designed to be imported. Use build_prefight_features(df).")
