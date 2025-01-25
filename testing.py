import numpy as np
import pickle
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical

# Constants
NUM_CLASSES = 16
MAX_EPOCHS = 50
BATCH_SIZE = 200
PICKLE_FILE = 'dataset/dataset_large.pickle'
MAP_SYMBOLS = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
               '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
               '+': 10, '-': 11, 'times': 12, 'div': 13, '(': 14, ')': 15}
VEC_SYMBOLS = np.vectorize(MAP_SYMBOLS.get)

# Load Dataset
with open(PICKLE_FILE, 'rb') as f:
    data = pickle.load(f)

# Preprocessing
X = np.array(data['img'])
y = VEC_SYMBOLS(np.array(data['label']))

# Convert images to grayscale if not already
X = np.array([cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img for img in X])

# Normalize images
X = X.astype(np.float32) / 255.0

# Expand dimensions for channel (needed for Conv2D)
X = np.expand_dims(X, -1)

# One-hot encode labels
y = to_categorical(y, num_classes=NUM_CLASSES)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute class weights for imbalanced datasets
class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(y, axis=1)), y=np.argmax(y, axis=1))
class_weights = dict(enumerate(class_weights))

# Model Definition
model = Sequential([
    Input(shape=(28, 28, 1)),  # Input Layer
    Conv2D(32, (3, 3), activation='relu'),  # Conv Layer 1
    BatchNormalization(),
    MaxPool2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),  # Conv Layer 2
    BatchNormalization(),
    MaxPool2D((2, 2)),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),  # Fully Connected Layer 1
    Dropout(0.4),
    Dense(64, activation='relu'),  # Fully Connected Layer 2
    Dense(NUM_CLASSES, activation='softmax')  # Output Layer
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
es = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, min_delta=0.01, restore_best_weights=True)
mc = ModelCheckpoint('bestmodel.keras', monitor='val_accuracy', verbose=1, save_best_only=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

callbacks = [es, mc, lr_scheduler]

# Train Model
history = model.fit(
    X_train, y_train,
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    class_weight=class_weights,
    shuffle=True,
    callbacks=callbacks,
    verbose=1
)

# Save the Model
model.save('final_model.keras')

# Evaluate Model on Test Data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Plot Training and Validation Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

# Plot Training and Validation Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.show()

# Inference Example
def predict_expression(image_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    img_resized = cv.resize(img, (28, 28)).astype(np.float32) / 255.0
    img_input = np.expand_dims(np.expand_dims(img_resized, -1), 0)  # Add batch and channel dims
    prediction = model.predict(img_input)
    predicted_class = np.argmax(prediction)
    for symbol, class_id in MAP_SYMBOLS.items():
        if class_id == predicted_class:
            return symbol
    return "Unknown"

# Test Inference
uploaded_image_path = "examples/test.jpeg"  # Replace with the path to your test image
result = predict_expression(uploaded_image_path)
print(f"Predicted Symbol: {result}")

