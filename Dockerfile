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

# Copy new files or directories and add to file system of container
COPY / / 

# Set the model name as an environment variable
ENV MODEL_NAME=models

# Expose port 8501
EXPOSE 8501

# Run command
RUN echo '#!/bin/bash'