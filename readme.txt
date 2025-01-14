# Context
-----------

In this study, the goal is to segment an image into standard road image categories (human, car, road, sky, vegetation, objects).

To select the most effective approach, we tested two models: U-Net and FPN.

We implemented an API that is accessible through a Streamlit app, which uses five images to evaluate the prediction model.

The Streamlit app and API are automatically deployed using GitHub Actions for continuous integration and continuous deployment (CI/CD).

To use it locally:

1. Install Docker Desktop.
2. Use the `docker-compose.yml` file to set up the environment.

# Technologies Used
----------------------

* **Computer Vision**: U-Net, FPN, EfficientNet, data augmentation
* **CI/CD**: Azure, Docker, GitHub Actions
* **API**: FastAPI
* **Frontend**: Streamlit