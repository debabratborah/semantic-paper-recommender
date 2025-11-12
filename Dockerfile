# Use the official PyTorch CUDA image as a base
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

#Set the working directory inside the container
WORKDIR /app

#Copy the requirements file into the container
COPY requirements.txt .

#Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Copy all project files into the container
COPY . .

#Command to run the script when the container starts
CMD ["python", "hgrec_recommendation.py"]