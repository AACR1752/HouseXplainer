import re
import ast
import pandas as pd

# To covert the following texts columns into maps
def convert_to_json(input):
    result = {}
    content = re.findall(r'\{(.*?)\}', input)
    for item in content:
        key_value = item.split(':', 1)[1].strip('"').split(':', 1)
        if len(key_value) == 2:
            key = key_value[0].strip().lower().replace(' ', '_')
            value = key_value[1].strip()
        result[key] = value
    return result

def clean_data(df_pandas):
    # Load the data
    # df_pandas = pd.read_csv(file)
    df_pandas.columns = df_pandas.columns.str.replace(' ', '_').str.lower()

    # Drop duplicates based on 'listing_id' column
    df_pandas = df_pandas.drop_duplicates(subset='listing_id')

    # Convert the following columns to JSON
    df_pandas['details'] = df_pandas['details'].apply(convert_to_json)
    df_pandas['key_facts'] = df_pandas['key_facts'].apply(convert_to_json)
    # df_pandas['rooms_details'] = df_pandas['rooms'].apply(convert_to_json)

    # Drop specified columns
    columns_to_drop = [
        'web-scraper-order',
        'web-scraper-start-url',
        # 'listing-href',
        'address',
        'comparables',
        'rooms'
    ]

    df_pandas = df_pandas.drop(columns=columns_to_drop)

    df_pandas["description"] = df_pandas["description"].str.extract(r'"description":"(.*?)"')
    df_pandas['ai_summary'] = df_pandas["ai_summary"].apply(lambda x: [item['AI_summary'] for item in ast.literal_eval(x)])

    df_pandas = df_pandas.rename(columns={'listing-href': 'listing_url'})

    # Normalize the JSON column
    json_df_1 = pd.json_normalize(df_pandas['key_facts']).set_index(df_pandas.index)
    json_df_2 = pd.json_normalize(df_pandas['details']).set_index(df_pandas.index)
    # json_df_3 = pd.json_normalize(df_pandas['rooms'])

    # Identify common columns
    common_columns = json_df_1.columns.intersection(json_df_2.columns)

    # Drop common columns from json_df_1
    json_df_1 = json_df_1.drop(columns=common_columns)

    # Combine the new DataFrame with the original one
    df_pandas = df_pandas.drop(columns=['key_facts', 'details']).join([json_df_1, json_df_2])

    # df_pandas.to_csv('house_sigma_raw_exploded.csv', index=False)
    return df_pandas