
base_folder = 'data/results/score/Mini'

def read_txt(file_path):
    with open(file_path, 'r') as f:
        data = f.read().splitlines()
    return data

driving_suggestion_txt_path = f'{base_folder}/driving_suggestion/all_score.txt'
general_perception_txt_path = f'{base_folder}/general_perception/all_score.txt'


REGION_CLASS_LST = ['barrier', 'miscellaneous', 'traffic_cone', 'traffic_light', 'traffic_sign', 'vehicle', 'vru']

driving_suggestion_data = read_txt(driving_suggestion_txt_path)
general_perception_data = read_txt(general_perception_txt_path)

region_score_dict = {}
for region_class in REGION_CLASS_LST:
    region_perception_txt_path = f'{base_folder}/region_perception/{region_class}.txt'
    data = read_txt(region_perception_txt_path)
    region_score_dict[region_class] = float(data[0].split(':')[1]) * 10


region_perception_data = read_txt(f'{base_folder}/region_perception/all_score.txt')

driving_suggestion_score = float(driving_suggestion_data[0].split(':')[1]) * 10
region_perception_score = float(region_perception_data[0].split(':')[1]) * 10
general_perception_score = float(general_perception_data[0].split(':')[1]) * 10


# prety print
print('General Perception Score:', general_perception_score)

print('Driving Suggestion Score:', driving_suggestion_score)

print('Region Perception Score:', region_perception_score)
print(region_score_dict)

