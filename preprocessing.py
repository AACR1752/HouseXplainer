import re
import pandas as pd
import numpy as np

def transform_address(address):
  """
  Transforms addresses in the following formats:
  - 'Unit A - 586 Mount Anne Drive' to '586A Mount Anne Drive'
  - '431-A Bairstow Crescent' to '431A Bairstow Crescent'
  - '424 a Tealby Cres' to '424a Tealby Cres'
  """

  if re.search(r'pinegrove', address, re.IGNORECASE):  # Check if 'pinegrove' exists
    address = re.sub(r'pinegrove', 'pine grove', address, flags=re.IGNORECASE)

  match1 = re.match(r"unit\s+([a-z])\s*-\s*(\d+)\s*(.*)", address, re.IGNORECASE)
  match2 = re.match(r"(\d+)\s*-\s*([a-z])\s*(.*)", address, re.IGNORECASE)
  match3 = re.match(r"(\d+)\s+([a-z])\s+(.*)", address, re.IGNORECASE)  # New pattern
  match4 = re.match(r"Unit\s+(\d+)\s*-\s*(\d+\s+.*)", address, re.IGNORECASE)

  if match1:
    return f"{match1.group(2)}{match1.group(1).upper()} {match1.group(3)}"
  elif match2:
    return f"{match2.group(1)}{match2.group(2).upper()} {match2.group(3)}"
  elif match3:
    return f"{match3.group(1)}{match3.group(2).upper()} {match3.group(3)}"  # New transformation
  elif match4:
    return f"{match4.group(2)} {match4.group(1)}"
  else:
    return address
  
def process_housing(df_house_sigma, output):
  # Convert the columns to lowercase
    df_house_sigma['address_lower'] = df_house_sigma['address'].str.lower().str.strip()
    output['civic_addr_lower'] = output['civic_addr'].str.lower().str.strip()

    # replace drive with dr and street with st on civic_addr_lower
    output['civic_addr_lower'] = output['civic_addr_lower'].str.replace('drive', 'dr')
    output['civic_addr_lower'] = output['civic_addr_lower'].str.replace('street', 'st')
    df_house_sigma['address_lower'] = df_house_sigma['address_lower'].str.replace('drive', 'dr')
    df_house_sigma['address_lower'] = df_house_sigma['address_lower'].str.replace('street', 'st')

    # only in df_house_sigma, crescent -> cres, avenue -> ave, boulevard -> blvd, road -> rd, place -> pl, court -> crt, circle -> cir
    df_house_sigma['address_lower'] = df_house_sigma['address_lower'].str.replace('crescent', 'cres')
    df_house_sigma['address_lower'] = df_house_sigma['address_lower'].str.replace('avenue', 'ave')
    df_house_sigma['address_lower'] = df_house_sigma['address_lower'].str.replace('boulevard', 'blvd')
    df_house_sigma['address_lower'] = df_house_sigma['address_lower'].str.replace('road', 'rd')
    df_house_sigma['address_lower'] = df_house_sigma['address_lower'].str.replace('place', 'pl')
    df_house_sigma['address_lower'] = df_house_sigma['address_lower'].str.replace('court', 'crt')
    df_house_sigma['address_lower'] = df_house_sigma['address_lower'].str.replace('circle', 'cir')

    # Working on some edge cases
    df_house_sigma['address_lower'] = df_house_sigma['address_lower'].apply(transform_address)
    df_house_sigma['address_lower'] = df_house_sigma['address_lower'].str.lower().str.strip()

    # Perform the merge using the lowercase columns
    merged_df = pd.merge(df_house_sigma, output, left_on='address_lower', right_on='civic_addr_lower', how='left')

    # Drop the temporary lowercase columns if not needed
    # merged_df.drop(columns=['address_lower', 'civic_addr_lower'], inplace=True)

    # Select the specified columns
    result_df = merged_df[['listing_id', 'address' , 'civic_addr', 'latitude', 'longitude', 'geometry', 'neighbourhood', 'type', 'listed', 'sold', 'details', 'key facts']]
    
    return result_df