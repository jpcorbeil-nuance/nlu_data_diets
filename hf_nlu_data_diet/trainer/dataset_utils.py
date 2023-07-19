def format_nlu_dataset(dataset):
    dataset.set_format(type=None, columns=['id', 'input_ids', 'attention_mask', "intent_label", "slot_label"])
    return dataset

def format_classification_dataset(dataset):
    dataset.set_format(type=None, columns=['id', 'input_ids', 'attention_mask', "label"])
    return dataset
