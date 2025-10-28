
Install MLflow:

Open your terminal or command prompt.
Run the following command to install MLflow using pip:
pip install mlflow

After installation, run the below command to run mlflow server locally

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000


To deploy it in streamlit cloud. Setup a remote ML Flow server and then commit the code to Github and then go to https://streamlit.io/cloud

Choose free version, select the github repo, python file and deploy the streamlit application