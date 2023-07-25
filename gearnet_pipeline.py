import json
from lib.pipeline import Pipeline
from lib.disable_logger import DisableLogger

GPU = 1

# run experiment without storing data
def run_exp_pure(layer, knn_k=10, spatial_radius=10.0, sequential_max_distance=2):
    pipeline = Pipeline(
        model='gearnet',
        dataset='atpbind3d',
        gpus=[GPU],
        model_kwargs={
            'gpu': GPU,
            'input_dim': 21,
            'hidden_dims': [512] * layer,
        },
        graph_knn_k=knn_k,
        graph_spatial_radius=spatial_radius,
        graph_sequential_max_distance=sequential_max_distance,
    )
    
    train_record = pipeline.train_until_fit(patience=3)
    return train_record


def add_to_data(file_data, parameters, trial):
    '''
    file_data: list of {parameters: dict, trials: list}
    parameters: dict
    trial: dict

    This function would find the entry in file_data with the same parameters
    and append the trial to the trials list.
    '''
    result = file_data[::]
    for entry in result:
        if all(entry['parameters'][k] == v for k, v in parameters.items()):
            entry['trials'].append(trial)
            return result
    result.append({
        "parameters": parameters,
        "trials": [trial]
    })
    return result

def read_initial_data(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, IndexError):
        # File does not exist, or it is empty
        data = []
    return data

def main():
    DATA_FILE_PATH = 'gearnet_ablation_pipeline.json'
    data = read_initial_data(DATA_FILE_PATH)
    layer = 4
    for trial in range(3):
        for knn_k in [5, 10, 15]:
            for spatial_radius in [5.0, 10.0, 15.0]:
                for sequential_max_distance in [4, 5, 6]:
                        parameters = {
                            'layer': layer,
                            'knn_k': knn_k,
                            'spatial_radius': spatial_radius,
                            'sequential_max_distance': sequential_max_distance,
                        }
                        print(parameters)
                        train_record = run_exp_pure(layer, knn_k, spatial_radius, sequential_max_distance)
                        data = add_to_data(data, parameters, train_record)
                        with open(DATA_FILE_PATH, 'w') as f:
                            json.dump(data, f, indent=2)



if __name__ == '__main__':
    main()
