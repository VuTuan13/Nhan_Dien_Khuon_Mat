from tensorflow.keras.models import load_model, Model

def load_embedding_model(model_path='model/best_resnet50.keras'):
    full_model = load_model(model_path)
    embedding_layer_output = full_model.layers[-3].output  # Dense(512)
    embedding_model = Model(inputs=full_model.input, outputs=embedding_layer_output)
    return embedding_model
