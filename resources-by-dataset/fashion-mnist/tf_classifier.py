import tensorflow as tf
import vehicle_lang as vcl
from vehicle_lang.loss import tensorflow as loss_tf
from tensorflow.keras import layers, Sequential

MEAN, STD = 0.2860, 0.3530  # mean and std dev of Fashion MNIST
BATCH_SIZE = 64
SUBSET_SIZE = 1024 # ensure SUBSET_SIZE mod BATCH_SIZE = 0

(train_images, train_labels), _ = tf.keras.datasets.fashion_mnist.load_data()

train_images = train_images[:SUBSET_SIZE]
train_labels = train_labels[:SUBSET_SIZE]
train_labels = train_labels.astype("int32")

train_images = train_images.astype("float32") / 255.0
train_images = (train_images - MEAN) / STD
train_images = train_images[..., None]  # add channel dim -> (N, 28, 28, 1)

train_loader = (
    tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    .shuffle(SUBSET_SIZE)
    .batch(BATCH_SIZE)
)

# load Vehicle specification + loss function

spec = loss_tf.load_specification(
    "fmnist-robustness.vcl",
    logic=vcl.VehicleDifferentiableLogic()
)

constraint_loss_fn = spec["robust"]

model = Sequential([
    layers.InputLayer(shape=(1, 28, 28)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(10)
])

@tf.function
def network(x: tf.Tensor) -> tf.Tensor:
    return tf.reshape(model(tf.reshape(x, (1, 1, 28, 28))), (10,))

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

num_epochs = 5
alpha = 0.5

for epoch in range(num_epochs):
    running_total_loss, correct, seen = 0.0, 0, 0

    for step, (images, labels) in enumerate(train_loader):
        with tf.GradientTape() as tape:
            logits = model(images)
            task_loss = cross_entropy(labels, logits)

            constraint_loss = constraint_loss_fn(
                n=BATCH_SIZE,
                classifier=network,
                epsilon=tf.constant(0.005),
                trainingImages=tf.squeeze(images, axis=-1),
                trainingLabels=labels
            )

            constraint_loss = tf.reduce_mean(tf.stack(constraint_loss))
            total_loss = alpha * task_loss + (1 - alpha) * constraint_loss

        print(
            f"Step {step}:\n\t"
            f"task loss:        {task_loss.numpy():.4f}\n\t"
            f"constraint loss:  {constraint_loss.numpy():.4f}\n\t"
            f"total loss:       {total_loss.numpy():.4f}"
        )

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
 
        running_total_loss += total_loss.numpy() * labels.shape[0]
        correct += (tf.argmax(logits, axis=1, output_type=labels.dtype) == labels).numpy().sum()
        seen += labels.shape[0]
        
    print(
        f"Epoch: {epoch + 1}, "
        f"mean total loss: {running_total_loss / seen:.4f}, "
        f"train accuracy: {100 * correct / seen:.1f}%"
    )

model.export("models/tf_simple_classifier")
