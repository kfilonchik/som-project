import pandas as pd
import pm4py
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

class PreProcessing:

    def extract_features(event_log):
        event_log = pm4py.format_dataframe(event_log, case_id='case_id', activity_key='activity', timestamp_key='timestamp')
        event_log =  event_log.sort_values(by=['case_id', 'timestamp', 'SORTKEY'])

        event_log['timestamp'] = pd.to_datetime(event_log['timestamp'])

        #Extract durations
        case_durations = pm4py.get_all_case_durations(event_log, activity_key='activity', case_id_key='case_id', timestamp_key='timestamp')
        #Extract number of process steps
        process_steps_per_case = event_log.groupby('case_id').size().reset_index(name='num_steps')
        unique_activities = event_log['activity'].unique()
        # Number of reworks
        #sum_reworks_per_case = event_log.groupby('case_id')['ISREWORK'].sum().reset_index(name='sum_reworks')
        # Manual steps
        #automation_rate  = event_log.groupby('case_id')['AUTOMATIC'].mean().reset_index()
        manual_steps_df = event_log[event_log['AUTOMATIC'] != 1]
        manual_steps_per_case = manual_steps_df.groupby('case_id').size().reset_index(name='manual_steps')
        rueckabwicklung_count = event_log[event_log['activity'] == 'Rückabwicklung'].groupby('case_id').size().reset_index(name='rueckabwicklung_count')
        mahnstufe_count = event_log[event_log['activity'].str.contains('Mahnstufe')].groupby('case_id').size().reset_index(name='mahnstufe_count')

        # Mark case as open or closed
        #end_activities = ['Ausgleich durch Eingangszahlung', 'Ausgleich durch Ausgangszahlung', 'Ausbuchung', 'Ausgleich durch Sachbearbeitung']
        #event_log['is_closed'] = event_log['activity'].isin(end_activities).astype(int)
       # case_status = event_log.groupby('case_id')['is_closed'].max().reset_index()
        #case_status['is_open'] = 1 - case_status['is_closed']
        #case_status.drop(columns=['is_closed'], inplace=True)

        # Calculate durations between specific process steps
        # Duration between 'Ablesung angelegt' and 'Faktura angelegt'
        # Step 2: Initialize Vectors
        case_ids = event_log['case_id'].unique()
        activity_counts = pd.DataFrame(0, index=case_ids, columns=unique_activities)

        # Step 3: Populate the Vectors
        for case_id, group in event_log.groupby('case_id'):
            counts = group['activity'].value_counts()
            activity_counts.loc[case_id, counts.index] = counts.values

        # Step 4: Normalize the Vectors (Optional)
        #activity_counts_normalized = activity_counts.div(activity_counts.sum(axis=1), axis=0)
        #activity_counts_normalized.index.name = 'case_id'
        #print(activity_counts_normalized)
        activity_counts.index.name = 'case_id'


        #duration_ablesung_to_faktura = calculate_duration(event_log, 'Ablesung angelegt', 'Faktura angelegt')

        # Duration from 'Vorläufiges Ende' to today's date
        #today = pd.to_datetime('now').normalize()
        #duration_vorlaeufiges_ende_to_today = event_log[event_log['activity'] == 'Vorläufiges Ende / kein Ausgleich'].copy()
        #print(duration_vorlaeufiges_ende_to_today)
        #duration_vorlaeufiges_ende_to_today['timestamp'] =  pd.to_datetime(duration_vorlaeufiges_ende_to_today['timestamp']).dt.tz_localize(None)
        #duration_vorlaeufiges_ende_to_today['duration_to_today_days'] = (today - duration_vorlaeufiges_ende_to_today['timestamp']).dt.total_seconds() / (3600 * 24)
        #duration_vorlaeufiges_ende_to_today = duration_vorlaeufiges_ende_to_today[['case_id', 'duration_to_today_days']]

        
        features = pd.DataFrame()
        features['case_id'] = case_ids
        features['case_duration'] = case_durations
        features['case_duration'] = features['case_duration'].astype(int) / 86400
        merged_df = features.merge(process_steps_per_case, on='case_id', how='left')\
                    .merge(manual_steps_per_case, on='case_id', how='left')\
                    .merge(rueckabwicklung_count, on='case_id', how='left')\
                    .merge(mahnstufe_count, on='case_id', how='left')\
                    #.merge(activity_counts, on='case_id', how='left')
                      #.merge(sum_reworks_per_case, on='case_id', how='left')\
                    #.merge(duration_ablesung_to_faktura, on='case_id', how='left')\
                    #.merge(case_status, on='case_id', how='left')\
                   # .merge(duration_vorlaeufiges_ende_to_today, on='case_id', how='left')
        
        # Fill NaN values with appropriate default values
        merged_df.fillna({
            'rueckabwicklung_count': 0,
           'mahnstufe_count': 0,
           'manual_steps' : 0
        #    'duration_Ablesung angelegt_to_Faktura angelegt_days': 0,
        #    'duration_to_today_days': 0
        }, inplace=True)

        print(merged_df.columns)

        #merged_df['manual_steps'].fillna(0, inplace=True)
        #merged_df['manual_steps'] = merged_df['manual_steps'] / merged_df['num_steps']
        #merged_df['num_steps'] = merged_df['num_steps'] / event_log['activity'].nunique()
        #merged_df['rueckabwicklung_count'] = merged_df['rueckabwicklung_count'] /  merged_df['rueckabwicklung_count'].max()
        #merged_df['mahnstufe_count'] =  merged_df['mahnstufe_count'] /  merged_df['mahnstufe_count'].max()
        #merged_df['case_duration'] = (merged_df['case_duration'] / np.mean(merged_df['case_duration'], axis=0)) / np.std(merged_df['case_duration'], axis=0)

        merged_df.to_csv('data/features-cases-reduced3.csv', index=False)

        return merged_df 

    def create_feature_matrix(df):
        df_numeric = df.drop(columns=['case_id'])
        # Normalize the data
        scaler = StandardScaler()
        df_normalized = scaler.fit_transform(df_numeric)

        # Convert to Numpy array for SOM
        data_for_som = df_normalized

        return data_for_som
