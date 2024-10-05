import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('ECG.h5')

# Define the test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
x_test = test_datagen.flow_from_directory("C:/Users/shres/Downloads/Project/Arrhythmia Detection/data/test", target_size=(64, 64), batch_size=32, class_mode="categorical", shuffle=False)


# Generate predictions for the test set
y_pred = model.predict(x_test, steps=len(x_test), verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get true labels
y_true = x_test.classes

# Sort class labels based on their indices
class_labels = [k for k, v in sorted(x_test.class_indices.items(), key=lambda item: item[1])]

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45)  # Rotate x-axis labels
plt.yticks(rotation=0)   # Rotate y-axis labels
plt.rc('font', size=8)  # Adjust font size
plt.tight_layout()  # Ensures the plot fits within the figure area
plt.show()

# Print classification report
print(classification_report(y_true, y_pred_classes, target_names=class_labels))
