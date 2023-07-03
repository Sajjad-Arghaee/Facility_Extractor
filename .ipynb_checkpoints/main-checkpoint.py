import pandas as pd
import scrapy
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

train = pd.read_csv('data/input.csv')
validation = pd.read_csv('data/output.csv')

# good article on
# https://medium.com/@mikeyo4800/how-to-build-a-multi-label-text-classification-model-using-nlp-and-machine-learning-2e05f72aad5f

facility_enum = open('data/facility_enum.txt', 'r')
Lines = facility_enum.readlines()
facility_enum.close()

# provide new dataframe with edited tags
filled_train = []
# find name according to code number
code_name_enumeration = {}

for line in Lines:
    code_split_name = list(line.split('\t'))
    code_split_name[1] = code_split_name[1][:-1]
    code_name_enumeration[code_split_name[0]] = code_split_name[1]

for index, row in validation.iterrows():
    validation.at[index, 'facility_code'] = code_name_enumeration[row['facility_code']].lower().strip()
validation.rename(columns={'facility_code': 'facility'})

for index, row in train.iterrows():
    hotel_id = row.hotel_id
    text = row.content
    if type(text) is not str:
        # print(f'There is not any provided content for {hotel_id} Hotel')
        continue
    items = scrapy.Selector(text=text).css('.hotel-description-content::text').extract()
    # location = items[0]
    # facilities = items[1:]
    facilities = items
    merged_facilities = ''
    for facility in facilities:
        merged_facilities += facility + ' '

    merged_facilities.lower().strip()
    prepare_new_row = [hotel_id, merged_facilities]
    filled_train.append(prepare_new_row)

filled_train = pd.DataFrame(filled_train, columns=['hotel_id', 'facilities'])
# print(validation)
# print(filled_train)

facility_codes = validation['facility_code'].tolist()
unique_facility_codes = set(facility_codes)
hotel_ids = filled_train['hotel_id'].tolist()
unique_hotel_ids = set(hotel_ids)

features_of_hotel = []

for hotel_id in unique_hotel_ids:
    hotel = validation[validation.hotel_id == hotel_id]
    features = hotel['facility_code'].tolist()
    features_of_hotel.append([hotel_id, features])

features_df = pd.DataFrame(features_of_hotel, columns=['hotel_id', 'features'])
# print(features_df)
# print(filled_train)

number_of_occurrence = {}
for index, row in features_df.iterrows():
    features_line = row['features']
    for feature in features_line:
        if number_of_occurrence.get(feature):
            number_of_occurrence[feature] += 1
        else:
            number_of_occurrence[feature] = 1

# Creating histogram
# fig, ax = plt.subplots(figsize=(10, 7))
# ax.hist(number_of_occurrence.values())
# plt.show()
# print(number_of_occurrence)

number_of_occurrence_filtered = {}
for key, value in number_of_occurrence.items():
    if value > 10:
        number_of_occurrence_filtered[key] = value

# print(len(number_of_occurrence))
# print(len(number_of_occurrence_filtered))
# result => from 231 to 144 features
selected_features = number_of_occurrence_filtered.keys()
for index, row in features_df.iterrows():
    new_features = []
    for feature in row['features']:
        if feature in selected_features:
            new_features.append(feature)
    features_df.at[index, 'features'] = new_features

preprocessed_dataframe = pd.merge(filled_train['facilities'], features_df['features'], left_index=True,
                                  right_index=True)
X_train, X_test, y_train, y_test = train_test_split(preprocessed_dataframe['facilities'],
                                                    preprocessed_dataframe['features'], test_size=0.2, random_state=42)
