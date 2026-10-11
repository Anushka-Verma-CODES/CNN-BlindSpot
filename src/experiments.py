import pandas as pd

def create_empty_log():
    return pd.DataFrame(columns=[
        "Mission", "Target_Class", "Original_Score", 
        "Action", "Cell", "New_Score", "Score_Change"
    ])

def log_action(df, mission_title, target_class, orig_score, action, cell, new_score):
    change = new_score - orig_score
    new_row = pd.DataFrame([{
        "Mission": mission_title,
        "Target_Class": target_class,
        "Original_Score": round(orig_score, 2),
        "Action": action,
        "Cell": cell,
        "New_Score": round(new_score, 2),
        "Score_Change": round(change, 2)
    }])
    return pd.concat([df, new_row], ignore_index=True)

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')
