import os
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.contrib.session_bundle import exporter
import keras.backend as K

# устанавливаем режим в test time.
K.set_learning_phase(0)

# создаем модель и загружаем веса
model = ResNet50(weights='imagenet')

sess = K.get_session()

# задаем путь сохранения модели и версию модели
export_path_base = './model'
export_version = 1

export_path = os.path.join(
  tf.compat.as_bytes(export_path_base),
  tf.compat.as_bytes(str(export_version)))
print('Exporting trained model to', export_path)
builder = tf.saved_model.builder.SavedModelBuilder(export_path)

# создаем входы и выходы из тензоров
model_input = tf.saved_model.utils.build_tensor_info(model.input)
model_output = tf.saved_model.utils.build_tensor_info(model.output)

# создаем сигнатуру для предсказания, в которой устанавливаем входы и выходы модели
prediction_signature = (
  tf.saved_model.signature_def_utils.build_signature_def(
      inputs={'images': model_input},
      outputs={'scores': model_output},
      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

# добавляем сигнатуры к SavedModelBuilder
legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
builder.add_meta_graph_and_variables(
  sess, [tf.saved_model.tag_constants.SERVING],
  signature_def_map={
      'predict':
          prediction_signature,
  },
  legacy_init_op=legacy_init_op)

builder.save()
