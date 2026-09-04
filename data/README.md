# Data

This project uses the **Women's Clothing E-Commerce Reviews** dataset: 23,486 real customer
product reviews, each with free-text review content and a 1-5 star rating.

- **Original source:** [Kaggle - Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) (public domain / CC0).
- **Programmatic copy used by this project:** loaded automatically at notebook runtime from a
  GitHub-hosted CSV mirror, so no manual download or Kaggle API key is required:

  ```
  https://raw.githubusercontent.com/nethajinirmal13/Training-datasets/main/Womens%20Clothing%20E-Commerce%20Reviews.csv
  ```

- **Columns used by this project:**
  - `Review Text` — free-text customer review (model input).
  - `Rating` — 1-5 star rating, converted into a 3-class sentiment label:

    ```
    1-2 stars -> Negative
    3   stars -> Neutral
    4-5 stars -> Positive
    ```

The raw CSV is **not** committed to this repository (see `.gitignore`) — it is downloaded and
cached to `data/` automatically the first time the notebook runs, making the whole pipeline
reproducible from a fresh clone without any manual data setup.
