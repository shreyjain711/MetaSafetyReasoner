import json

def data_generator(file_path, batch_size):
    with open(file_path, 'r') as f:
        batch = []
        for line in f:
            data = json.loads(line)
            batch.append(data)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch