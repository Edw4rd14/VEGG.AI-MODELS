# ==================================================
# ST1516 DEVOPS AND AUTOMATION FOR AI CA2 ASSIGNMENT
# ==================================================
# NAME: EDWARD TAN YUAN CHONG
# CLASS: DAAA/FT/2B/04
# ADM NO: 2214407
# ==================================================
# FILENAME: Dockerfile
# ==================================================

# Use the TensorFlow Serving base image
FROM tensorflow/serving

# Create a directory for the model configuration file
RUN mkdir -p /models

# Copy the models to the respective directories
COPY models/conv2d128 /models/conv2d128
COPY models/customwgg31 /models/customwgg31
COPY model_config.conf /models/model_config.conf

# Set the model config file environment variable
ENV MODEL_CONFIG_FILE=/models/model_config.conf

# Expose the REST API port
EXPOSE 8501

# Use the config file option when running TensorFlow Serving
CMD ["tensorflow_model_server", "--port=8500", "--rest_api_port=8501", "--model_config_file=${MODEL_CONFIG_FILE}"]
