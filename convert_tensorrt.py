from tensorflow.python.compiler.tensorrt import trt_convert as trt
from sklearn.model_selection import train_test_split
import numpy as np

model_path = 'mlmodels/static'

x_train, x_test, y_train, y_test = train_test_split(np.load('dataset/static_X.npy'), np.load('dataset/static_Y.npy'), test_size=0.3, random_state=42)

converter = trt.TrtGraphConverterV2(
   input_saved_model_dir=model_path,
   precision_mode=trt.TrtPrecisionMode.FP32
)

MAX_BATCH_SIZE=128
def input_fn():
   batch_size = MAX_BATCH_SIZE
   x = x_test[0:batch_size]
   yield [x]
   
trt_func = converter.convert() 
converter.build(input_fn=input_fn)

OUTPUT_SAVED_MODEL_DIR = 'mlmodels/static_tftrt'
converter.save(output_saved_model_dir=OUTPUT_SAVED_MODEL_DIR)