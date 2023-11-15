# MLOps_Project02

## Overview

Welcome to MLOps_Project02! This project focuses on streamlining the machine learning operations (MLOps) workflow, making it easier for users to set up and monitor their models using Docker and Weights & Biases (WANDB). Follow the steps below to get started.

## Prerequisites

Make sure you have the following installed on your machine:

- [Docker](https://www.docker.com/get-started)
- [Sourcetree](https://www.sourcetreeapp.com/) (optional, for GitHub interaction)

## Getting Started

1. **Clone the Repository:**
   - If you're not familiar with GitHub, download Sourcetree and create an account.
   - Fork this project and create a clone.

2. **Update Local Repository:**
   - Open the project and run the following command in Sourcetree or terminal:
     ```bash
     git pull
     ```

3. **Build Docker Image:**
   - Navigate to the project folder in the terminal.
   - Build the Docker image using the following command:
     ```bash
     docker build -t image_name .
     ```

4. **Set up WANDB:**
   - If you don't have a WANDB account, [create one](https://wandb.ai/site).
   - Log in to your WANDB account.

5. **Run Docker Image:**
   - Execute the following command to run the Docker image, replacing `your_actual_api_key` with your WANDB API key:
     ```bash
     docker run -e WANDB_API_KEY=your_actual_api_key image_name
     ```

6. **Explore WANDB Dashboard:**
   - Now, you can follow the model training progress on the WANDB dashboard.
7. **Have Fun!**
   - Enjoy exploring and experimenting with MLOps_Project02! If you have any questions or suggestions, don't hesitate to reach out. Happy coding!

