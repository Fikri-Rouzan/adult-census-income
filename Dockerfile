FROM tensorflow/serving:latest

# Salin SavedModel hasil Pusher ke dalam container
COPY serving_model/adult-income-model /models/adult-income-model

# Set variabel environment untuk nama model
ENV MODEL_NAME=adult-income-model

# Expose port REST API dan Prometheus Metrics
EXPOSE 8501

# Jalankan TF Serving dengan opsi pemantauan Prometheus aktif
CMD ["tensorflow_model_server", \
     "--port=8500", \
     "--rest_api_port=8501", \
     "--model_name=adult-income-model", \
     "--model_base_path=/models/adult-income-model", \
     "--monitoring_config_file=/models/adult-income-model/monitoring_config.file"]