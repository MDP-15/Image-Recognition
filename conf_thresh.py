thresholds = {
    '1': 0.6,
    '2': 0.78,
    '3': 0.75,
    '4': 0.72,
    '5': 0.75,
    '6': 0.6,
    '7': 0.6,
    '8': 0.75,
    '9': 0.78,
    '10': 0.6,
    '11': 0.7,
    '12': 0.72,
    '13': 0.6,
    '14': 0.7,
    '15': 0.6,
}


def confidence_threshold(label_id):
    return thresholds[label_id]
