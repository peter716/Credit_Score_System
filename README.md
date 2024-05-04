A credit scoring system is designed to predict the creditworthiness of potential borrowers. In a recent project, I explored the deployment and monitoring of such a system using a REST API. DynamoDB was chosen as the database due to its scalability and performance characteristics.

In this lab, I enhanced the functionality by enabling bulk loading of credit scores directly into DynamoDB using a specially designed driver function. This function was updated to align with the REST API design principles, incorporating changes such as revised function parameters and appropriate HTTP status codes.

Additionally, I managed the model training process by scanning the retraining table and appending this data to a copy of the original `credit_train` file. This process created a new training dataset, which was subsequently used in the `do_model_update` function to perform model retraining. To improve model validation, the dataset was partitioned such that the first 20 percent of the data was used for validation purposes, while the remaining 80 percent—including the newly appended retraining data—was used for training.

This structured approach not only streamlined the model training and updating process but also ensured that the system remained robust and responsive to new data.
